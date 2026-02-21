"""SQLite-backed persona profile persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from animus_bootstrap.personas.engine import PersonaProfile
from animus_bootstrap.personas.voice import VoiceConfig


class PersonaStorage:
    """SQLite-backed persona profile persistence."""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                system_prompt TEXT DEFAULT '',
                voice_config TEXT DEFAULT '{}',
                knowledge_domains TEXT DEFAULT '[]',
                excluded_topics TEXT DEFAULT '[]',
                channel_bindings TEXT DEFAULT '{}',
                active BOOLEAN DEFAULT 1,
                is_default BOOLEAN DEFAULT 0
            );
        """)
        self._conn.commit()

    def save(self, persona: PersonaProfile) -> None:
        """Insert or update a persona."""
        self._conn.execute(
            """INSERT OR REPLACE INTO personas
            (id, name, description, system_prompt, voice_config,
             knowledge_domains, excluded_topics, channel_bindings,
             active, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                persona.id,
                persona.name,
                persona.description,
                persona.system_prompt,
                json.dumps(self._voice_to_dict(persona.voice)),
                json.dumps(persona.knowledge_domains),
                json.dumps(persona.excluded_topics),
                json.dumps(persona.channel_bindings),
                persona.active,
                persona.is_default,
            ),
        )
        self._conn.commit()

    def load(self, persona_id: str) -> PersonaProfile | None:
        """Load a persona by ID."""
        cur = self._conn.execute("SELECT * FROM personas WHERE id = ?", (persona_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_persona(row)

    def load_all(self) -> list[PersonaProfile]:
        """Load all personas."""
        cur = self._conn.execute("SELECT * FROM personas ORDER BY name")
        return [self._row_to_persona(row) for row in cur.fetchall()]

    def delete(self, persona_id: str) -> bool:
        """Delete a persona. Returns True if found."""
        cur = self._conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _voice_to_dict(voice: VoiceConfig) -> dict:
        """Serialize VoiceConfig to a JSON-safe dict."""
        return asdict(voice)

    @staticmethod
    def _dict_to_voice(data: dict) -> VoiceConfig:
        """Deserialize a dict to VoiceConfig."""
        return VoiceConfig(**data)

    def _row_to_persona(self, row: sqlite3.Row) -> PersonaProfile:
        """Convert a database row to a PersonaProfile."""
        voice_data = json.loads(row["voice_config"])
        return PersonaProfile(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            system_prompt=row["system_prompt"],
            voice=self._dict_to_voice(voice_data),
            knowledge_domains=json.loads(row["knowledge_domains"]),
            excluded_topics=json.loads(row["excluded_topics"]),
            channel_bindings=json.loads(row["channel_bindings"]),
            active=bool(row["active"]),
            is_default=bool(row["is_default"]),
        )
