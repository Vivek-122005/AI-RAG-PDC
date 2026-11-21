"""Regex PII detection and redaction.

Deliberately dependency-free: a model-based detector (e.g. Presidio + spaCy) is more
thorough but costs ~0.5 GB of RAM, which this project's 8 GB laptop target can't spare.
The patterns cover the identifiers whose leak would do real harm (contact details,
payment cards, government IDs, secrets); names and addresses are out of scope.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIFinding:
    kind: str
    start: int
    end: int


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _phone_ok(candidate: str) -> bool:
    """Reject runs of numbers from tables ("79169 8758 3611"): a number separated only by
    spaces must carry a country code (+) or a bracketed area code to count as a phone."""
    digits = sum(c.isdigit() for c in candidate)
    if not 10 <= digits <= 15:
        return False
    spaces_only = not any(c in "-." for c in candidate)
    return not spaces_only or candidate.lstrip().startswith(("+", "(")) or "(" in candidate


# (kind, pattern, extra validation). Order matters: earlier kinds win overlapping spans.
_PATTERNS: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    (
        "API_KEY",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
            r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
        None,
    ),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _luhn_ok),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    (
        "PHONE",
        re.compile(
            r"(?<![\w.(])(?:\+?\d{1,3}[ .-]?)?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]\d{3,4}(?!\w|\.\d)"
        ),
        _phone_ok,
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\.?\d)"
        ),
        None,
    ),
)


def find_pii(text: str) -> list[PIIFinding]:
    """Non-overlapping PII spans in `text`, in order of appearance."""
    found: list[PIIFinding] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern, valid in _PATTERNS:
        for m in pattern.finditer(text):
            if valid and not valid(m.group()):
                continue
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append(PIIFinding(kind, m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def redact_pii(text: str) -> tuple[str, list[PIIFinding]]:
    """Replace each PII span with a typed placeholder such as `[EMAIL]`."""
    findings = find_pii(text)
    out, cursor = [], 0
    for f in findings:
        out += [text[cursor : f.start], f"[{f.kind}]"]
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out), findings
