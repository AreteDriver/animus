"""Animus Citizens — Mind Foundation layer.

Long-lived specialist citizens that improve the Animus Mind itself
before any domain-specialist citizens are built.

Phase 0 Citizens:
- Architect (Citizen 001): observes, analyzes, proposes improvements
- Conversation Designer (Citizen 002): reduces cognitive effort
- Knowledge Curator (Citizen 003): maintains knowledge accuracy

Every citizen operates under the Constitution layer:
- No autonomous architectural changes
- No autonomous memory changes
- No autonomous deployment
- Evidence before action
- Human approval required for all commissions
"""

from __future__ import annotations

from animus.citizens.architect import ArchitectCitizen
from animus.citizens.commissioner import ForgeCommissioner
from animus.citizens.citizen_council import CitizenCouncil, RankedProposal
from animus.citizens.conversation_designer import ConversationDesignerCitizen
from animus.citizens.abstraction import AbstractionCitizen
from animus.citizens.harvester import HarvesterCitizen
from animus.citizens.intelligence import IntelligenceCitizen
from animus.citizens.knowledge_curator import KnowledgeCuratorCitizen
from animus.citizens.pattern import PatternCitizen
from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalConfidence,
    ProposalStatus,
    RiskAssessment,
)
from animus.citizens.proposal_queue import ProposalQueue, QueuedProposal
from animus.citizens.session_steward import SessionStewardCitizen
from animus.citizens.test_oracle import TestOracleCitizen

__all__ = [
    "AbstractionCitizen",
    "ArchitectCitizen",
    "CitizenCouncil",
    "ConversationDesignerCitizen",
    "ForgeCommissioner",
    "HarvesterCitizen",
    "IntelligenceCitizen",
    "KnowledgeCuratorCitizen",
    "PatternCitizen",
    "ProposalQueue",
    "QueuedProposal",
    "RankedProposal",
    "SessionStewardCitizen",
    "TestOracleCitizen",
    "EvidenceItem",
    "ImprovementProposal",
    "ProposalConfidence",
    "ProposalStatus",
    "RiskAssessment",
]
