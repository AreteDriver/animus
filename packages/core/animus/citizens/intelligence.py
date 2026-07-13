"""Citizen 006 — The Intelligence Officer.

The sensory and reconnaissance system of Animus.

Responsibilities:
- Extract entities from documents, code, and external sources
- Perform open-source intelligence (OSINT) on public information
- Detect secrets, credentials, and sensitive data in artifacts
- Build entity co-occurrence graphs for relationship analysis
- Monitor external sources for changes relevant to tracked entities
- Produce evidence-backed intelligence reports and proposals

Never:
- Access private data without authorization
- Bypass authentication or terms of service
- Modify external systems or profiles
- Deploy autonomous actions based on intelligence findings

Instead:
    Observe → Extract → Correlate → Report → Human Approval → Action
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalConfidence,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("citizens.intelligence")


# ═══════════════════════════════════════════════════════════════════
# Data Structures — Ported from RedOPS + Dossier
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ExtractedEntities:
    """Container for entities extracted from text."""

    emails: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)
    ipv4_addresses: set[str] = field(default_factory=set)
    ipv6_addresses: set[str] = field(default_factory=set)
    phone_numbers: set[str] = field(default_factory=set)
    md5_hashes: set[str] = field(default_factory=set)
    sha1_hashes: set[str] = field(default_factory=set)
    sha256_hashes: set[str] = field(default_factory=set)
    social_handles: set[str] = field(default_factory=set)
    credit_cards: set[str] = field(default_factory=set)
    aws_keys: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, list[str]]:
        return {k: sorted(v) for k, v in {
            "emails": self.emails,
            "domains": self.domains,
            "urls": self.urls,
            "ipv4_addresses": self.ipv4_addresses,
            "ipv6_addresses": self.ipv6_addresses,
            "phone_numbers": self.phone_numbers,
            "md5_hashes": self.md5_hashes,
            "sha1_hashes": self.sha1_hashes,
            "sha256_hashes": self.sha256_hashes,
            "social_handles": self.social_handles,
            "credit_cards": self.credit_cards,
            "aws_keys": self.aws_keys,
        }.items()}

    def merge(self, other: ExtractedEntities) -> None:
        self.emails.update(other.emails)
        self.domains.update(other.domains)
        self.urls.update(other.urls)
        self.ipv4_addresses.update(other.ipv4_addresses)
        self.ipv6_addresses.update(other.ipv6_addresses)
        self.phone_numbers.update(other.phone_numbers)
        self.md5_hashes.update(other.md5_hashes)
        self.sha1_hashes.update(other.sha1_hashes)
        self.sha256_hashes.update(other.sha256_hashes)
        self.social_handles.update(other.social_handles)
        self.credit_cards.update(other.credit_cards)
        self.aws_keys.update(other.aws_keys)

    def total_count(self) -> int:
        return sum(len(v) for v in self.to_dict().values())

    def is_empty(self) -> bool:
        return self.total_count() == 0


@dataclass
class SecretFinding:
    """A detected secret or credential in code/text."""

    pattern_name: str
    description: str
    severity: str  # critical, high, medium, low
    matched_text: str
    file_path: str = ""
    line_number: int = 0
    confidence: float = 0.9
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlatformProfile:
    """A discovered public profile on a platform."""

    username: str
    platform: str
    url: str
    category: str  # social, professional, code, forum, other
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceReport:
    """Report produced by the Intelligence Officer after analysis."""

    source: str
    extracted: ExtractedEntities = field(default_factory=ExtractedEntities)
    secrets: list[SecretFinding] = field(default_factory=list)
    profiles: list[PlatformProfile] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════
# Compiled Patterns — Ported from RedOPS entity_extract + code_artifacts
# ═══════════════════════════════════════════════════════════════════

_ENTITY_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
    ),
    "url": re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*", re.IGNORECASE),
    "domain": re.compile(
        r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
        r"(?:com|org|net|edu|gov|mil|io|co|dev|app|xyz|info|biz|us|uk|de|fr|jp|cn|ru|br|in|au|ca)\b",
        re.IGNORECASE,
    ),
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "ipv6": re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|"
        r"\b\+?[1-9]\d{1,14}\b"
    ),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "social": re.compile(r"@[A-Za-z0-9_]{1,30}\b"),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"  # Visa
        r"5[1-5][0-9]{14}|"  # MasterCard
        r"3[47][0-9]{13}|"  # Amex
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"  # Discover
    ),
    "aws_key": re.compile(r"\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b"),
}

_SECRET_PATTERNS = {
    "aws_access_key": {
        "pattern": re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}"),
        "description": "AWS Access Key ID",
        "severity": "critical",
    },
    "github_token": {
        "pattern": re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}"),
        "description": "GitHub Personal Access Token",
        "severity": "critical",
    },
    "github_oauth": {
        "pattern": re.compile(r"gho_[A-Za-z0-9_]{36,}"),
        "description": "GitHub OAuth Token",
        "severity": "critical",
    },
    "generic_api_key": {
        "pattern": re.compile(
            r"(?:api[_-]?key|apikey)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"
        ),
        "description": "Generic API Key",
        "severity": "high",
    },
    "generic_secret": {
        "pattern": re.compile(
            r"(?:secret|password|passwd|pwd)\s*[=:]\s*['\"]([^'\"]{8,})['\"]"
        ),
        "description": "Generic Secret/Password",
        "severity": "high",
    },
    "private_key": {
        "pattern": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIV" + r"ATE KEY-----"),
        "description": "Private Key",
        "severity": "critical",
    },
    "slack_token": {
        "pattern": re.compile(
            r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,}"
        ),
        "description": "Slack Token",
        "severity": "high",
    },
    "stripe_key": {
        "pattern": re.compile(r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}"),
        "description": "Stripe API Key",
        "severity": "critical",
    },
    "google_api_key": {
        "pattern": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
        "description": "Google API Key",
        "severity": "high",
    },
    "jwt_token": {
        "pattern": re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "description": "JWT Token",
        "severity": "medium",
    },
    "database_url": {
        "pattern": re.compile(
            r"(?:mysql|postgres|postgresql|mongodb|redis)://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+"
        ),
        "description": "Database Connection String with Credentials",
        "severity": "critical",
    },
}

# Social platforms for OSINT URL generation
_PLATFORMS: dict[str, dict[str, Any]] = {
    "github": {
        "name": "GitHub",
        "url": "https://github.com/{username}",
        "category": "code",
        "format": r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$",
        "max_length": 39,
    },
    "gitlab": {"name": "GitLab", "url": "https://gitlab.com/{username}", "category": "code"},
    "twitter": {
        "name": "Twitter/X",
        "url": "https://twitter.com/{username}",
        "category": "social",
        "format": r"^[a-zA-Z0-9_]+$",
        "max_length": 15,
    },
    "linkedin": {
        "name": "LinkedIn",
        "url": "https://linkedin.com/in/{username}",
        "category": "professional",
        "format": r"^[a-zA-Z0-9-]+$",
    },
    "reddit": {
        "name": "Reddit",
        "url": "https://reddit.com/user/{username}",
        "category": "social",
        "format": r"^[a-zA-Z0-9_-]+$",
        "max_length": 20,
    },
    "hackernews": {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/user?id={username}",
        "category": "forum",
    },
    "medium": {
        "name": "Medium",
        "url": "https://medium.com/@{username}",
        "category": "social",
    },
    "dev_to": {
        "name": "DEV.to",
        "url": "https://dev.to/{username}",
        "category": "code",
    },
    "keybase": {
        "name": "Keybase",
        "url": "https://keybase.io/{username}",
        "category": "code",
    },
    "npm": {"name": "npm", "url": "https://npmjs.com/~{username}", "category": "code"},
    "pypi": {"name": "PyPI", "url": "https://pypi.org/user/{username}", "category": "code"},
    "dockerhub": {
        "name": "Docker Hub",
        "url": "https://hub.docker.com/u/{username}",
        "category": "code",
    },
}

_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".pdf",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".pyc", ".pyo",
}

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", "target", ".idea", ".vscode",
}


# ═══════════════════════════════════════════════════════════════════
# Intelligence Citizen
# ═══════════════════════════════════════════════════════════════════


class IntelligenceCitizen:
    """Citizen 006 — The Intelligence Officer.

    Extracts entities, detects secrets, generates OSINT profiles,
    and builds intelligence reports. NEVER acts on findings directly;
    always produces reports for human review.
    """

    def __init__(
        self,
        memory_layer: MemoryLayer | None = None,
        evidence_dir: Path | str | None = None,
        codebase_path: Path | str = ".",
    ):
        self.memory = memory_layer
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None
        self.codebase_path = Path(codebase_path).expanduser()
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_codebase(self) -> list[dict[str, Any]]:
        """Scan codebase for secrets and return findings as observations.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if not self.codebase_path.exists():
            logger.warning("Codebase path does not exist: %s", self.codebase_path)
            return findings

        secrets = self.scan_directory_secrets(self.codebase_path)
        for s in secrets:
            findings.append({
                "source": "codebase",
                "description": f"{s.description}: {s.matched_text}",
                "severity": s.severity,
                "context": {
                    "pattern": s.pattern_name,
                    "file": s.file_path,
                    "line": s.line_number,
                },
            })

        entities = self.extract_from_directory(self.codebase_path)
        for file_path, extracted in entities.items():
            if extracted.total_count() > 0:
                findings.append({
                    "source": "codebase",
                    "description": f"Extracted {extracted.total_count()} entities from {file_path}",
                    "severity": "info",
                    "context": {
                        "file": file_path,
                        "entities": extracted.to_dict(),
                    },
                })

        logger.info("Intelligence observe_codebase: %d findings", len(findings))
        return findings

    # ------------------------------------------------------------------
    # Entity Extraction (ported from RedOPS entity_extract)
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> ExtractedEntities:
        """Extract all entity types from text using regex patterns.

        Args:
            text: Source text to analyze.

        Returns:
            ExtractedEntities containing all found entities.
        """
        entities = ExtractedEntities()
        if not text:
            return entities

        entities.emails = set(_ENTITY_PATTERNS["email"].findall(text))
        entities.urls = set(_ENTITY_PATTERNS["url"].findall(text))
        entities.domains = set(_ENTITY_PATTERNS["domain"].findall(text))
        entities.ipv4_addresses = set(_ENTITY_PATTERNS["ipv4"].findall(text))
        entities.ipv6_addresses = set(_ENTITY_PATTERNS["ipv6"].findall(text))
        entities.phone_numbers = set(_ENTITY_PATTERNS["phone"].findall(text))
        entities.md5_hashes = set(_ENTITY_PATTERNS["md5"].findall(text))
        entities.sha1_hashes = set(_ENTITY_PATTERNS["sha1"].findall(text))
        entities.sha256_hashes = set(_ENTITY_PATTERNS["sha256"].findall(text))
        entities.social_handles = set(_ENTITY_PATTERNS["social"].findall(text))
        entities.credit_cards = set(_ENTITY_PATTERNS["credit_card"].findall(text))
        entities.aws_keys = set(_ENTITY_PATTERNS["aws_key"].findall(text))

        return entities

    def extract_from_file(self, file_path: Path | str) -> ExtractedEntities:
        """Extract entities from a file.

        Args:
            file_path: Path to file to analyze.

        Returns:
            ExtractedEntities from file contents.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return ExtractedEntities()

        if path.suffix.lower() in _SKIP_EXTENSIONS:
            logger.debug(f"Skipping binary file: {path}")
            return ExtractedEntities()

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return self.extract_entities(text)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return ExtractedEntities()

    def extract_from_directory(
        self, dir_path: Path | str, pattern: str = "**/*"
    ) -> dict[str, ExtractedEntities]:
        """Extract entities from all files in a directory.

        Args:
            dir_path: Directory to scan.
            pattern: Glob pattern for file matching.

        Returns:
            Mapping of file path -> ExtractedEntities.
        """
        root = Path(dir_path)
        if not root.exists():
            logger.warning(f"Directory not found: {root}")
            return {}

        results: dict[str, ExtractedEntities] = {}
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in _SKIP_EXTENSIONS:
                continue

            entities = self.extract_from_file(path)
            if not entities.is_empty():
                results[str(path.relative_to(root))] = entities

        return results

    # ------------------------------------------------------------------
    # Secret Detection (ported from RedOPS code_artifacts)
    # ------------------------------------------------------------------

    def scan_secrets(self, text: str, source: str = "") -> list[SecretFinding]:
        """Scan text for secrets and credentials.

        Args:
            text: Source text to scan.
            source: Identifier for the source (filename, URL, etc.).

        Returns:
            List of secret findings.
        """
        findings: list[SecretFinding] = []
        if not text:
            return findings

        for name, config in _SECRET_PATTERNS.items():
            for match in config["pattern"].finditer(text):
                findings.append(
                    SecretFinding(
                        pattern_name=name,
                        description=config["description"],
                        severity=config["severity"],
                        matched_text=match.group()[:50],  # Truncate for safety
                        file_path=source,
                        confidence=0.9,
                    )
                )

        return findings

    def scan_file_secrets(self, file_path: Path | str) -> list[SecretFinding]:
        """Scan a file for secrets with line numbers."""
        path = Path(file_path)
        if not path.exists():
            return []

        findings: list[SecretFinding] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return findings

        for line_num, line in enumerate(lines, 1):
            line_findings = self.scan_secrets(line, source=str(path))
            for finding in line_findings:
                finding.line_number = line_num
            findings.extend(line_findings)

        return findings

    def scan_directory_secrets(
        self, dir_path: Path | str, pattern: str = "**/*"
    ) -> list[SecretFinding]:
        """Scan all files in a directory for secrets."""
        root = Path(dir_path)
        if not root.exists():
            return []

        all_findings: list[SecretFinding] = []
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in _SKIP_EXTENSIONS:
                continue

            findings = self.scan_file_secrets(path)
            all_findings.extend(findings)

        return all_findings

    # ------------------------------------------------------------------
    # OSINT Profile Generation (ported from RedOPS social_osint)
    # ------------------------------------------------------------------

    def generate_profile_urls(self, username: str) -> list[PlatformProfile]:
        """Generate public profile URLs for a username across platforms.

        Args:
            username: Username to check.

        Returns:
            List of PlatformProfile objects with generated URLs.
            (Does NOT verify existence — generates candidate URLs only.)
        """
        profiles: list[PlatformProfile] = []
        for platform_id, config in _PLATFORMS.items():
            # Validate username format if regex provided
            fmt = config.get("format")
            max_len = config.get("max_length", 30)
            if len(username) > max_len:
                continue
            if fmt and not re.match(fmt, username):
                continue

            url = config["url"].format(username=username)
            profiles.append(
                PlatformProfile(
                    username=username,
                    platform=config["name"],
                    url=url,
                    category=config["category"],
                    confidence=0.3,  # Low confidence — URLs not verified
                )
            )

        return profiles

    def extract_usernames(self, text: str) -> set[str]:
        """Extract potential usernames from text.

        Args:
            text: Source text.

        Returns:
            Set of candidate usernames.
        """
        usernames: set[str] = set()
        if not text:
            return usernames

        # @username patterns
        for match in _ENTITY_PATTERNS["social"].finditer(text):
            handle = match.group().lstrip("@")
            if 3 <= len(handle) <= 30:
                usernames.add(handle)

        # GitHub/GitLab URL extraction
        github_pattern = re.compile(r"github\.com/([a-zA-Z0-9-]+)", re.IGNORECASE)
        for match in github_pattern.finditer(text):
            usernames.add(match.group(1))

        return usernames

    def generate_osint_report(self, text: str) -> IntelligenceReport:
        """Generate a comprehensive OSINT report from text.

        Args:
            text: Source text (e.g., document, web page, conversation).

        Returns:
            IntelligenceReport with extracted entities, secrets, and profiles.
        """
        report = IntelligenceReport(source="text_analysis")
        report.extracted = self.extract_entities(text)
        report.secrets = self.scan_secrets(text)

        # Generate profile URLs from social handles
        for handle in report.extracted.social_handles:
            username = handle.lstrip("@")
            profiles = self.generate_profile_urls(username)
            report.profiles.extend(profiles)

        # Also extract usernames from URLs
        usernames = self.extract_usernames(text)
        for username in usernames:
            existing = {p.username for p in report.profiles}
            if username not in existing:
                profiles = self.generate_profile_urls(username)
                report.profiles.extend(profiles)

        # Build summary
        parts = []
        if report.extracted.total_count() > 0:
            parts.append(f"{report.extracted.total_count()} entities extracted")
        if report.secrets:
            parts.append(f"{len(report.secrets)} secret(s) detected")
        if report.profiles:
            parts.append(f"{len(report.profiles)} profile URL(s) generated")
        report.summary = "; ".join(parts) if parts else "No findings"

        return report

    # ------------------------------------------------------------------
    # NER Integration (uses dossier.ner)
    # ------------------------------------------------------------------

    def extract_named_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities using the Dossier NER engine.

        Args:
            text: Source text.

        Returns:
            List of entity dicts with name, type, confidence.
        """
        try:
            from animus.dossier.ner import NEREngine

            engine = NEREngine()
            result = engine.extract(text)
            entities = []
            for category in ("people", "places", "orgs"):
                for item in getattr(result, category, []):
                    entities.append({
                        "name": item.get("name", ""),
                        "type": category,
                        "confidence": 0.7,
                        "canonical": item.get("name", "").lower().strip(),
                    })
            return entities
        except ImportError:
            logger.warning("NEREngine not available — install with 'animus[dossier]'")
            return []

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def analyze(self, text: str | None = None, file_path: Path | str | None = None) -> IntelligenceReport:
        """Analyze text or file and produce an intelligence report.

        Args:
            text: Direct text to analyze.
            file_path: File to analyze (alternative to text).

        Returns:
            IntelligenceReport with all findings.
        """
        if text:
            report = self.generate_osint_report(text)
            report.entities = self.extract_named_entities(text)
        elif file_path:
            path = Path(file_path)
            text_content = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
            report = self.generate_osint_report(text_content)
            report.entities = self.extract_named_entities(text_content)
            report.source = f"file:{path}"
        else:
            report = IntelligenceReport(source="empty")
            report.summary = "No input provided"

        return report

    # ------------------------------------------------------------------
    # Proposal Generation (follows Architect pattern)
    # ------------------------------------------------------------------

    def generate_proposal(self, report: IntelligenceReport | None = None) -> ImprovementProposal | None:
        """Generate an improvement proposal from intelligence findings.

        Args:
            report: IntelligenceReport to base proposal on. If None, scans codebase
                and builds a report automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if report is None:
            # Autonomous-loop path: scan codebase and build report
            if not self.codebase_path.exists():
                logger.info("No codebase path available — no proposal generated")
                return None
            logger.info("Scanning codebase for intelligence report: %s", self.codebase_path)
            # Sample a subset of files to avoid excessive scanning
            sample_texts: list[str] = []
            for path in self.codebase_path.rglob("*.py"):
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                try:
                    sample_texts.append(path.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                if len(sample_texts) >= 20:
                    break

            combined = "\n\n".join(sample_texts)
            if not combined:
                logger.info("No source files found to analyze — no proposal generated")
                return None
            report = self.generate_osint_report(combined)
            report.entities = self.extract_named_entities(combined)

        # Critical secrets are always actionable
        critical_secrets = [s for s in report.secrets if s.severity == "critical"]
        high_secrets = [s for s in report.secrets if s.severity == "high"]

        if critical_secrets:
            problem = f"{len(critical_secrets)} critical secret(s) exposed in {report.source}"
            recommendation = (
                "Rotate exposed credentials immediately. Review secret storage practices "
                "and implement pre-commit hooks (e.g., gitleaks) to prevent future leakage."
            )
            severity = "critical"
        elif high_secrets:
            problem = f"{len(high_secrets)} high-severity secret(s) detected in {report.source}"
            recommendation = (
                "Audit and remove exposed credentials. Consider using secret management "
                "solutions (e.g., HashiCorp Vault, AWS Secrets Manager)."
            )
            severity = "high"
        elif report.extracted.credit_cards:
            problem = f"{len(report.extracted.credit_cards)} credit card number(s) in {report.source}"
            recommendation = "Remove payment card data from source immediately. PCI compliance violation risk."
            severity = "critical"
        elif report.extracted.aws_keys:
            problem = f"{len(report.extracted.aws_keys)} AWS key(s) in {report.source}"
            recommendation = "Rotate AWS credentials and audit IAM policies."
            severity = "high"
        else:
            # No actionable security findings — skip proposal
            logger.info("No critical security findings — no proposal generated")
            return None

        evidence = [
            EvidenceItem(
                source=report.source,
                description=s.description,
                data={
                    "pattern": s.pattern_name,
                    "severity": s.severity,
                    "line": s.line_number,
                    "file": s.file_path,
                },
                timestamp=datetime.now(),
            )
            for s in report.secrets
        ]

        proposal = ImprovementProposal(
            id=f"INTEL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Intelligence Alert: {problem[:60]}",
            problem=problem,
            evidence=evidence,
            root_cause="Credentials or sensitive data detected in source material through automated pattern matching",
            recommendation=recommendation,
            alternatives_considered=[
                "Ignore findings (not recommended for critical secrets)",
                "Manual audit only (slower, may miss patterns)",
            ],
            expected_benefits="Reduced attack surface and compliance risk",
            potential_risks=[
                RiskAssessment(
                    description="Credential rotation may disrupt services",
                    severity="medium",
                    mitigation="Stagger rotation during low-traffic windows",
                    probability=0.3,
                ),
            ],
            confidence_score=0.85,
            estimated_effort_hours=2.0,
            affected_components=["Security", "Infrastructure"],
            evaluation_plan="Re-scan after remediation to confirm secrets removed",
            rollback_plan="Restore from backup if rotation causes failures",
            success_metrics=["Zero secrets in re-scan", "Services operational post-rotation"],
            status=ProposalStatus.DRAFT,
        )

        logger.info(f"Generated proposal {proposal.id}: {proposal.title}")
        return proposal

    def store_proposal(self, proposal: ImprovementProposal) -> bool:
        """Store a proposal in Animus memory (autonomous-loop compatibility).

        Args:
            proposal: Proposal to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — proposal not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"{proposal.title}\n\n{proposal.problem}\n\nRecommendation: {proposal.recommendation}",
                memory_type=MemoryType.PROCEDURAL,
                tags=["intelligence", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info(f"Proposal {proposal.id} stored in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to store proposal: {e}")
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_report(self, report: IntelligenceReport) -> bool:
        """Store an intelligence report in Animus memory.

        Args:
            report: Report to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — report not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=report.summary,
                memory_type=MemoryType.SEMANTIC,
                tags=["intelligence", "osint", report.source],
                metadata={
                    "entities": report.extracted.to_dict(),
                    "secrets_count": len(report.secrets),
                    "profiles_count": len(report.profiles),
                    "timestamp": report.timestamp.isoformat(),
                },
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store report: {e}")
            return False
