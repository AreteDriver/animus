"""Ogma — reverse-engineering synthesis persona.

Companion to Lugh: where Lugh harvests, Ogma deciphers. Reads external work
(papers, podcasts, HN, blogs) and our own projects, produces implementation-
ready synthesis written to ``~/projects/notes/ogma/YYYY-MM-DD-<slug>.md``.

v0 scope (per ``docs/OGMA.md``): ``read.read()`` for one cached lugh item;
``brief``/``gap``/``audit``/``sweep`` ship as CLI stubs in this package and
fill in in v0.1+.
"""

from animus.ogma.grounding import GapResult, GroundingError, verify_animus_gap
from animus.ogma.models import OgmaOutput, OgmaParseError
from animus.ogma.read import OgmaSynthesisError, synthesize

__all__ = [
    "GapResult",
    "GroundingError",
    "OgmaOutput",
    "OgmaParseError",
    "OgmaSynthesisError",
    "synthesize",
    "verify_animus_gap",
]
