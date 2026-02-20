"""Field-level encryption utility using Fernet symmetric encryption.

Provides authenticated encryption for sensitive dict fields (API keys,
tokens, PII) using a key derived from Settings.secret_key via PBKDF2.
"""

import base64
import functools
import logging

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

_ENCRYPTED_PREFIX = "enc:"
_PBKDF2_SALT = b"gorgon-field-encryption-v1"
_PBKDF2_ITERATIONS = 480_000


class FieldEncryptor:
    """Encrypts and decrypts string values using Fernet authenticated encryption.

    The encryption key is derived from a secret string via PBKDF2-HMAC-SHA256,
    producing a URL-safe base64-encoded 32-byte key suitable for Fernet.

    Args:
        secret_key: The secret used to derive the Fernet key.
    """

    def __init__(self, secret_key: str) -> None:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_PBKDF2_SALT,
            iterations=_PBKDF2_ITERATIONS,
        )
        derived = kdf.derive(secret_key.encode("utf-8"))
        self._fernet = Fernet(base64.urlsafe_b64encode(derived))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Base64-encoded ciphertext prefixed with ``enc:``.
        """
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return f"{_ENCRYPTED_PREFIX}{token.decode('utf-8')}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string previously produced by :meth:`encrypt`.

        Args:
            ciphertext: The ``enc:``-prefixed ciphertext to decrypt.

        Returns:
            The original plaintext string.

        Raises:
            ValueError: If the ciphertext is not properly prefixed or is invalid.
        """
        if not ciphertext.startswith(_ENCRYPTED_PREFIX):
            raise ValueError("Ciphertext missing 'enc:' prefix — not an encrypted value")
        raw = ciphertext[len(_ENCRYPTED_PREFIX) :]
        try:
            return self._fernet.decrypt(raw.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Decryption failed — invalid token or wrong key") from exc

    def encrypt_dict_fields(self, data: dict, fields: list[str]) -> dict:
        """Return a copy of *data* with the specified fields encrypted.

        Fields that are already encrypted (``enc:`` prefix) or missing from the
        dict are silently skipped.

        Args:
            data: The source dictionary.
            fields: Keys whose values should be encrypted.

        Returns:
            A shallow copy of *data* with the requested fields encrypted.
        """
        result = dict(data)
        for field in fields:
            value = result.get(field)
            if isinstance(value, str) and not value.startswith(_ENCRYPTED_PREFIX):
                result[field] = self.encrypt(value)
        return result

    def decrypt_dict_fields(self, data: dict, fields: list[str]) -> dict:
        """Return a copy of *data* with the specified fields decrypted.

        Fields that are not encrypted (no ``enc:`` prefix) or missing from the
        dict are silently skipped.

        Args:
            data: The source dictionary.
            fields: Keys whose values should be decrypted.

        Returns:
            A shallow copy of *data* with the requested fields decrypted.
        """
        result = dict(data)
        for field in fields:
            value = result.get(field)
            if isinstance(value, str) and value.startswith(_ENCRYPTED_PREFIX):
                result[field] = self.decrypt(value)
        return result


@functools.lru_cache(maxsize=1)
def get_field_encryptor() -> FieldEncryptor:
    """Create or return a cached :class:`FieldEncryptor` using application settings.

    Returns:
        A :class:`FieldEncryptor` instance keyed from ``Settings.secret_key``.
    """
    from animus_forge.config.settings import get_settings

    settings = get_settings()
    return FieldEncryptor(settings.secret_key)
