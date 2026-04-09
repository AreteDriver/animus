"""Captcha solver using Ollama vision models and Whisper audio fallback."""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from animus_faucet.models import CaptchaType

if TYPE_CHECKING:
    pass

logger = logging.getLogger("animus_faucet.vision.captcha")


@dataclass
class SolveResult:
    """Result of a captcha solve attempt."""

    solved: bool
    method: str  # "vision", "audio", "checkbox"
    confidence: float = 0.0
    error: str | None = None


class CaptchaSolver:
    """Solves captchas using local vision models and audio transcription.

    Strategy priority:
    1. Checkbox click (reCAPTCHA v2) — often passes without image challenge
    2. Audio fallback + Whisper — highest accuracy (~85-90%)
    3. Vision model (Ollama llava) — for image grid challenges (~70-85%)
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llava"):
        self.ollama_url = ollama_url
        self.model = model

    async def solve(self, page, captcha_type: CaptchaType) -> SolveResult:
        """Attempt to solve a captcha on the given page."""
        if captcha_type == CaptchaType.NONE:
            return SolveResult(solved=True, method="none", confidence=1.0)

        if captcha_type == CaptchaType.RECAPTCHA_V2:
            return await self._solve_recaptcha_v2(page)
        elif captcha_type == CaptchaType.HCAPTCHA:
            return await self._solve_hcaptcha(page)
        elif captcha_type == CaptchaType.TURNSTILE:
            return await self._solve_turnstile(page)
        else:
            return SolveResult(
                solved=False,
                method="unsupported",
                error=f"Unsupported captcha type: {captcha_type.value}",
            )

    async def _solve_recaptcha_v2(self, page) -> SolveResult:
        """Attempt reCAPTCHA v2: checkbox first, then audio fallback."""
        # Step 1: Try clicking the checkbox
        try:
            checkbox = page.frame_locator("iframe[src*='recaptcha']").locator(
                ".recaptcha-checkbox-border"
            )
            await checkbox.click()

            # Wait to see if it passes without a challenge
            import asyncio

            await asyncio.sleep(2.0)

            # Check if the checkbox is now checked
            checked = (
                await page.frame_locator("iframe[src*='recaptcha']")
                .locator(".recaptcha-checkbox-checked")
                .count()
            )

            if checked > 0:
                return SolveResult(solved=True, method="checkbox", confidence=0.95)

        except Exception as e:
            logger.debug(f"Checkbox click failed: {e}")

        # Step 2: Try audio challenge
        return await self._try_audio_fallback(page, "recaptcha")

    async def _solve_hcaptcha(self, page) -> SolveResult:
        """Attempt hCaptcha using vision model on image grid."""
        try:
            # Look for hCaptcha iframe
            challenge_frame = page.frame_locator("iframe[src*='hcaptcha']")
            prompt_el = challenge_frame.locator(".prompt-text")
            prompt_text = await prompt_el.text_content(timeout=5000)

            if not prompt_text:
                return SolveResult(
                    solved=False, method="vision", error="Could not read hCaptcha prompt"
                )

            # Screenshot the challenge grid
            grid = challenge_frame.locator(".challenge-container")
            screenshot = await grid.screenshot()

            # Ask vision model
            positions = await self._vision_identify(screenshot, prompt_text)

            if not positions:
                return SolveResult(
                    solved=False, method="vision", error="Vision model returned no positions"
                )

            # Click identified cells
            cells = challenge_frame.locator(".task-image")
            for pos in positions:
                if pos < await cells.count():
                    await cells.nth(pos).click()
                    import asyncio

                    await asyncio.sleep(0.3)

            # Submit
            verify_btn = challenge_frame.locator(".button-submit")
            await verify_btn.click()

            import asyncio

            await asyncio.sleep(2.0)
            return SolveResult(solved=True, method="vision", confidence=0.7)

        except Exception as e:
            logger.debug(f"hCaptcha vision solve failed: {e}")
            return SolveResult(solved=False, method="vision", error=str(e))

    async def _solve_turnstile(self, page) -> SolveResult:
        """Cloudflare Turnstile — usually auto-passes in a real browser."""
        import asyncio

        try:
            await asyncio.sleep(3.0)
            # Turnstile typically auto-solves if browser fingerprint looks human
            turnstile = page.locator("iframe[src*='turnstile']")
            if await turnstile.count() > 0:
                await turnstile.click()
                await asyncio.sleep(2.0)
            return SolveResult(solved=True, method="turnstile_passthrough", confidence=0.6)
        except Exception as e:
            return SolveResult(solved=False, method="turnstile_passthrough", error=str(e))

    async def _try_audio_fallback(self, page, captcha_system: str) -> SolveResult:
        """Download audio challenge and transcribe with Whisper."""
        try:
            import httpx

            # Click the audio button
            if captcha_system == "recaptcha":
                audio_btn = page.frame_locator("iframe[src*='recaptcha']").locator(
                    "#recaptcha-audio-button"
                )
                await audio_btn.click()

                import asyncio

                await asyncio.sleep(2.0)

                # Get audio source
                audio_src = (
                    await page.frame_locator("iframe[src*='recaptcha']")
                    .locator("#audio-source")
                    .get_attribute("src")
                )

                if not audio_src:
                    return SolveResult(solved=False, method="audio", error="No audio source found")

                # Download audio
                async with httpx.AsyncClient() as client:
                    resp = await client.get(audio_src)
                    audio_bytes = resp.content

                # Transcribe
                text = await self._transcribe_audio(audio_bytes)
                if not text:
                    return SolveResult(solved=False, method="audio", error="Whisper returned empty")

                # Type the answer
                input_el = page.frame_locator("iframe[src*='recaptcha']").locator("#audio-response")
                await input_el.fill(text)

                verify_btn = page.frame_locator("iframe[src*='recaptcha']").locator(
                    "#recaptcha-verify-button"
                )
                await verify_btn.click()

                await asyncio.sleep(2.0)
                return SolveResult(solved=True, method="audio", confidence=0.85)

        except Exception as e:
            logger.debug(f"Audio fallback failed: {e}")

        return SolveResult(solved=False, method="audio", error="Audio fallback failed")

    async def _vision_identify(self, screenshot_bytes: bytes, prompt: str) -> list[int]:
        """Use Ollama vision model to identify which grid cells match the prompt."""
        try:
            import httpx

            b64_image = base64.b64encode(screenshot_bytes).decode()

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": (
                            f"This is a captcha image grid. {prompt} "
                            "Return ONLY the cell numbers (0-indexed, left-to-right, "
                            "top-to-bottom) that match, as comma-separated integers. "
                            "Example: 0,3,5"
                        ),
                        "images": [b64_image],
                        "stream": False,
                    },
                )
                data = resp.json()
                response_text = data.get("response", "")

                # Parse comma-separated integers
                numbers = re.findall(r"\d+", response_text)
                return [int(n) for n in numbers if int(n) < 16]

        except Exception as e:
            logger.debug(f"Vision identify failed: {e}")
            return []

    async def _transcribe_audio(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio captcha using Whisper."""
        try:
            import tempfile

            import whisper

            model = whisper.load_model("base")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                f.write(audio_bytes)
                f.flush()
                result = model.transcribe(f.name)
                text = result.get("text", "").strip().lower()
                # Remove punctuation
                text = re.sub(r"[^\w\s]", "", text)
                return text if text else None

        except ImportError:
            logger.warning("Whisper not installed — pip install openai-whisper")
            return None
        except Exception as e:
            logger.debug(f"Whisper transcription failed: {e}")
            return None
