"""Pagination utility with an intentional off-by-one bug.

The bug: ``page_size`` is treated as inclusive, so requesting page 1 with
page_size=10 returns 11 items (indices 0–10 instead of 0–9).
"""

from __future__ import annotations

from typing import Any


def paginate(
    items: list[Any],
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    """Return a paginated slice of *items*.

    Args:
        items: The full list to paginate.
        page: 1-based page number.
        page_size: Number of items per page.

    Returns:
        Dict with ``items``, ``page``, ``page_size``, ``total``, ``pages``.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    total = len(items)
    # BUG: inclusive end index causes off-by-one (page_size=10 returns 11 items)
    start = (page - 1) * page_size
    end = start + page_size + 1  # ← BUG: should be start + page_size

    page_items = items[start:end]
    pages = (total + page_size - 1) // page_size

    return {
        "items": page_items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    }
