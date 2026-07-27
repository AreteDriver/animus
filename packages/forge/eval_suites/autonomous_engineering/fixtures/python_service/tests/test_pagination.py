"""Tests for pagination utility.

One test intentionally fails because of the off-by-one bug in ``paginate``.
"""

from __future__ import annotations

import pytest

from python_service.pagination import paginate


def test_paginate_first_page():
    items = list(range(25))
    result = paginate(items, page=1, page_size=10)
    assert result["page"] == 1
    assert result["page_size"] == 10
    assert result["total"] == 25
    assert result["pages"] == 3
    # This assertion FAILS because of the bug: returns 11 items instead of 10
    assert len(result["items"]) == 10


def test_paginate_second_page():
    items = list(range(25))
    result = paginate(items, page=2, page_size=10)
    assert result["items"] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_paginate_last_partial_page():
    items = list(range(25))
    result = paginate(items, page=3, page_size=10)
    assert result["items"] == [20, 21, 22, 23, 24]


def test_paginate_invalid_page():
    with pytest.raises(ValueError, match="page must be >= 1"):
        paginate([], page=0)


def test_paginate_invalid_page_size():
    with pytest.raises(ValueError, match="page_size must be >= 1"):
        paginate([], page=1, page_size=0)
