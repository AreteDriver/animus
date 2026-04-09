"""Base class for referral program definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from animus_referral.models import ReferralProgram


class BaseProgram(ABC):
    """Abstract base for a category of referral programs.

    Subclasses provide a registry of known programs and know how to
    generate referral links from a referral code.
    """

    @abstractmethod
    def get_programs(self) -> list[ReferralProgram]:
        """Return all known programs in this category."""
        raise NotImplementedError

    @abstractmethod
    def generate_link(self, program: ReferralProgram, referral_code: str) -> str:
        """Generate a referral link for the given program and code."""
        raise NotImplementedError

    def get_program_by_name(self, name: str) -> ReferralProgram | None:
        """Look up a program by name."""
        return next((p for p in self.get_programs() if p.name == name), None)
