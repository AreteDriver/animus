"""API-based faucet driver for direct HTTP claims."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from animus_faucet.drivers.base import BaseDriver
from animus_faucet.models import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig
    from animus_faucet.models import FaucetDefinition

logger = logging.getLogger("animus_faucet.drivers.api")


class APIDriver(BaseDriver):
    """Driver for faucets with direct HTTP API endpoints.

    These are the highest-ROI targets: no captcha, no browser overhead.
    Config expects `driver_config` with:
        - method: GET or POST (default POST)
        - address_field: JSON field name for wallet address (default "address")
        - headers: optional extra headers dict
        - payload_template: optional dict template with {address} placeholder
        - success_field: JSON response field to check for success
        - amount_field: JSON response field for payout amount
    """

    def __init__(self, faucet: FaucetDefinition, config: FaucetConfig):
        super().__init__(faucet, config)
        dc = self.faucet.driver_config
        self.method = dc.get("method", "POST").upper()
        self.address_field = dc.get("address_field", "address")
        self.extra_headers = dc.get("headers", {})
        self.payload_template = dc.get("payload_template")
        self.success_field = dc.get("success_field")
        self.amount_field = dc.get("amount_field")

    async def claim(self, wallet_address: str) -> ClaimResult:
        proxy_url = self.config.proxy.proxy_url if self.faucet.requires_proxy else None

        transport = httpx.AsyncHTTPTransport(proxy=proxy_url) if proxy_url else None
        async with httpx.AsyncClient(
            transport=transport,
            timeout=30.0,
            headers={
                "User-Agent": self.config.user_agent,
                **self.extra_headers,
            },
        ) as client:
            try:
                if self.payload_template:
                    payload = _interpolate(self.payload_template, wallet_address)
                else:
                    payload = {self.address_field: wallet_address}

                if self.method == "GET":
                    resp = await client.get(self.faucet.url, params=payload)
                else:
                    resp = await client.post(self.faucet.url, json=payload)

                if resp.status_code == 429:
                    return self._failure(ClaimStatus.RATE_LIMITED, "HTTP 429")
                if resp.status_code == 403:
                    return self._failure(ClaimStatus.BANNED, "HTTP 403")
                if resp.status_code >= 500:
                    return self._failure(ClaimStatus.FAUCET_DOWN, f"HTTP {resp.status_code}")

                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data, proxy_used=proxy_url)

            except httpx.ConnectError as e:
                return self._failure(ClaimStatus.NETWORK_ERROR, str(e))
            except httpx.TimeoutException:
                return self._failure(ClaimStatus.NETWORK_ERROR, "Request timed out")
            except httpx.HTTPStatusError as e:
                return self._failure(ClaimStatus.UNKNOWN_ERROR, str(e))

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head(self.faucet.url)
                return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _parse_response(self, data: dict[str, Any], proxy_used: str | None = None) -> ClaimResult:
        """Parse API response into ClaimResult."""
        if self.success_field:
            if not data.get(self.success_field):
                error = data.get("error", data.get("message", "Unknown API error"))
                return self._failure(ClaimStatus.UNKNOWN_ERROR, str(error), proxy_used=proxy_used)

        amount = None
        if self.amount_field:
            amount = data.get(self.amount_field)
            if amount is not None:
                amount = float(amount)

        tx_hash = data.get("tx_hash") or data.get("txHash") or data.get("hash")

        return self._success(
            amount=amount,
            tx_hash=tx_hash,
            proxy_used=proxy_used,
        )


def _interpolate(template: dict, address: str) -> dict:
    """Replace {address} placeholders in a payload template."""
    result = {}
    for key, value in template.items():
        if isinstance(value, str):
            result[key] = value.replace("{address}", address)
        elif isinstance(value, dict):
            result[key] = _interpolate(value, address)
        else:
            result[key] = value
    return result
