"""Model pin store — expected-digest verification for local Ollama models.

E12 (Qwen #5): closes the "backdoored open-weight model" vector by verifying
that a model's on-disk digest matches a previously-recorded pin before
running inference.

Usage:
    store = ModelPinStore()
    store.pin_model("qwen2.5:14b", "sha256:abc123...")
    if not store.verify_pin("qwen2.5:14b", current_digest):
        raise ProviderError("Model digest mismatch — possible tampering")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]  # Optional import

logger = logging.getLogger("model_pin")


def _default_pin_path() -> Path:
    return Path.home() / ".config" / "animus" / "model-pins.json"


class ModelPinStore:
    """Persistent store of expected model digests.

    Pins are stored as a JSON mapping:
        {"model_name": "digest", ...}

    The digest is the sha256 manifest digest returned by Ollama's ``/api/tags``
    endpoint (``models[N].digest``).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_pin_path()
        self._pins: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text())
                if isinstance(data, dict):
                    self._pins = data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load model pins from %s: %s", self._path, e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._pins, indent=2, sort_keys=True) + "\n")

    def pin_model(self, model_name: str, digest: str) -> None:
        """Record the expected digest for a model."""
        self._pins[model_name] = digest
        self._save()

    def unpin_model(self, model_name: str) -> None:
        """Remove a model pin."""
        self._pins.pop(model_name, None)
        self._save()

    def get_pin(self, model_name: str) -> str | None:
        """Return the pinned digest, or None if not pinned."""
        return self._pins.get(model_name)

    def verify_pin(self, model_name: str, current_digest: str) -> bool:
        """Check whether ``current_digest`` matches the stored pin.

        Returns ``True`` when:
        - the model is pinned and digests match, OR
        - the model is not pinned (no policy = no violation).

        Returns ``False`` only when the model IS pinned and the digest differs.
        """
        expected = self._pins.get(model_name)
        if expected is None:
            return True
        return expected == current_digest

    def list_pins(self) -> dict[str, str]:
        """Return a copy of all pins."""
        return dict(self._pins)


def fetch_ollama_digest(model_name: str, base_url: str = "http://localhost:11434") -> str | None:
    """Query Ollama ``/api/tags`` for the digest of a specific model.

    Returns the digest string (e.g. ``sha256:fa0eaa50...``) or None if
    the model is not found or Ollama is unreachable.
    """
    if httpx is None:
        logger.warning("httpx not installed; cannot fetch Ollama digests")
        return None

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("Ollama /api/tags query failed: %s", e)
        return None

    for model in data.get("models", []):
        if model.get("model") == model_name or model.get("name") == model_name:
            return model.get("digest")
    return None
