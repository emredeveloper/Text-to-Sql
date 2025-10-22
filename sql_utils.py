"""Utility helpers for SQL validation and safety checks."""
from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)

_INSERT_PREFIX = re.compile(r"^insert\s+into\s+", re.IGNORECASE)
_COMMENT_PATTERNS = (
    re.compile(r"--"),   # inline comments
    re.compile(r"/\*"),  # block comments
)


def _contains_semicolon_outside_quotes(statement: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    escape = False

    for char in statement:
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if char == ';' and not in_single_quote and not in_double_quote:
            return True

    return False


def normalize_sql_statement(statement: str) -> str:
    """Return the SQL statement stripped from leading/trailing whitespace."""
    return statement.strip()


def is_single_insert(statement: str) -> bool:
    """Return True when the statement represents a single INSERT command."""
    if not _INSERT_PREFIX.match(statement):
        return False

    body = statement.rstrip(";")
    if _contains_semicolon_outside_quotes(body):
        return False

    for pattern in _COMMENT_PATTERNS:
        if pattern.search(body):
            return False

    return True


def validate_insert_statement(statement: str) -> bool:
    """Validate that the provided SQL contains a single INSERT statement.

    The function is intentionally conservative. Any deviation from a simple
    INSERT command results in rejection. This minimises the risk of executing
    arbitrary SQL originating from user input.
    """
    normalised = normalize_sql_statement(statement)
    if not normalised:
        return False

    if not is_single_insert(normalised):
        LOGGER.warning("Rejected SQL payload: %s", statement)
        return False

    return True
