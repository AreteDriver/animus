"""JSONL persistence for articles and earnings."""

from __future__ import annotations

import logging
from pathlib import Path

from .models import Article, ContentPlatform, ContentStatus, EarningsRecord

logger = logging.getLogger(__name__)


class ContentStore:
    """Persists articles and earnings to JSONL files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._articles_path = self.data_dir / "articles.jsonl"
        self._earnings_path = self.data_dir / "earnings.jsonl"

    def save_article(self, article: Article) -> None:
        """Append or update an article in the store."""
        articles = self._load_articles_dict()
        articles[article.id] = article
        self._write_articles(list(articles.values()))

    def get_article(self, article_id: str) -> Article | None:
        """Get a single article by ID."""
        articles = self._load_articles_dict()
        return articles.get(article_id)

    def list_articles(
        self,
        platform: ContentPlatform | None = None,
        status: ContentStatus | None = None,
    ) -> list[Article]:
        """List articles with optional filters."""
        articles = list(self._load_articles_dict().values())
        if platform is not None:
            articles = [a for a in articles if a.platform == platform]
        if status is not None:
            articles = [a for a in articles if a.status == status]
        return articles

    def delete_article(self, article_id: str) -> bool:
        """Remove an article from the store."""
        articles = self._load_articles_dict()
        if article_id not in articles:
            return False
        del articles[article_id]
        self._write_articles(list(articles.values()))
        return True

    def save_earnings(self, record: EarningsRecord) -> None:
        """Append an earnings record."""
        with open(self._earnings_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def list_earnings(
        self,
        article_id: str | None = None,
        platform: ContentPlatform | None = None,
    ) -> list[EarningsRecord]:
        """List earnings records with optional filters."""
        records = self._load_earnings()
        if article_id is not None:
            records = [r for r in records if r.article_id == article_id]
        if platform is not None:
            records = [r for r in records if r.platform == platform]
        return records

    def article_count(self) -> int:
        """Return total number of articles."""
        return len(self._load_articles_dict())

    def _load_articles_dict(self) -> dict[str, Article]:
        """Load articles into a dict keyed by ID."""
        if not self._articles_path.exists():
            return {}
        result: dict[str, Article] = {}
        for line in self._articles_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                article = Article.model_validate_json(line)
                result[article.id] = article
            except Exception:
                logger.warning("Skipping malformed article line")
        return result

    def _write_articles(self, articles: list[Article]) -> None:
        """Rewrite the articles file."""
        with open(self._articles_path, "w") as f:
            for article in articles:
                f.write(article.model_dump_json() + "\n")

    def _load_earnings(self) -> list[EarningsRecord]:
        """Load all earnings records."""
        if not self._earnings_path.exists():
            return []
        records: list[EarningsRecord] = []
        for line in self._earnings_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                records.append(EarningsRecord.model_validate_json(line))
            except Exception:
                logger.warning("Skipping malformed earnings line")
        return records
