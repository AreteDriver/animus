"""FastAPI integration for Animus Contracts runtime validation.

Provides a dependency factory and route decorator so API handlers can gate
incoming payloads against the canonical JSON schemas in
``packages/contracts/``.

Usage::

    from animus_bootstrap.dashboard.contract_validation import ValidatedBody

    @router.post("/api/actions")
    async def create_action(body: dict = Depends(ValidatedBody("action"))):
        ...

Or with the decorator::

    from animus_bootstrap.dashboard.contract_validation import validate_contract

    @router.post("/api/actions")
    @validate_contract("action")
    async def create_action(request: Request):
        body = await request.json()
        ...
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

try:
    from animus_contracts import ValidationError, validate

    _HAS_CONTRACTS = True
except ImportError:  # pragma: no cover
    _HAS_CONTRACTS = False


class ValidatedBody:
    """FastAPI dependency factory — validates the request body against a schema.

    Args:
        schema_name: Basename of the contract schema (e.g. ``"action"``).
    """

    def __init__(self, schema_name: str) -> None:
        self.schema_name = schema_name

    async def __call__(self, request: Request) -> dict[str, Any]:
        if not _HAS_CONTRACTS:
            raise HTTPException(
                status_code=503,
                detail="Contracts package is not installed.",
            )

        body = await request.json()
        try:
            validate(body, self.schema_name)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"errors": exc.errors, "schema": exc.schema_name},
            ) from exc
        return body


def validate_contract(schema_name: str):
    """Decorator that validates the request body before the handler runs.

    The decorated handler **must** accept a ``request: Request`` parameter
    (either positionally or by name) so the decorator can read the body.

    Example::

        @router.post("/api/actions")
        @validate_contract("action")
        async def create_action(request: Request, ...):
            body = await request.json()
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _HAS_CONTRACTS:
                raise HTTPException(
                    status_code=503,
                    detail="Contracts package is not installed.",
                )

            request: Request | None = kwargs.get("request")
            if request is None:
                request = next(
                    (a for a in args if isinstance(a, Request)),
                    None,
                )
            if request is None:
                raise HTTPException(
                    status_code=500,
                    detail="validate_contract decorator requires a Request parameter",
                )

            body = await request.json()
            try:
                validate(body, schema_name)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"errors": exc.errors, "schema": exc.schema_name},
                ) from exc

            return await func(*args, **kwargs)

        return wrapper

    return decorator
