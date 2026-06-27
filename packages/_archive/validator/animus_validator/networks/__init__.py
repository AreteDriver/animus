"""Network adapters for Animus Validator."""

from animus_validator.networks.base import BaseNetwork
from animus_validator.networks.ethereum import EthereumNetwork
from animus_validator.networks.solana import SolanaNetwork
from animus_validator.networks.sui import SuiNetwork

__all__ = [
    "BaseNetwork",
    "EthereumNetwork",
    "SolanaNetwork",
    "SuiNetwork",
]
