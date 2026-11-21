"""Guardrails: input checks, untrusted-evidence screening and output checks.

Rule-based and dependency-free so they run on the 8 GB laptop target; each detector sits
behind a small function so a model-based one can replace it later.
"""

from agentic_rag.guardrails.checks import InputDecision, InputGuard, OutputDecision, OutputGuard
from agentic_rag.guardrails.injection import InjectionVerdict, detect_injection, is_safe_passage
from agentic_rag.guardrails.pii import PIIFinding, find_pii, redact_pii

__all__ = [
    "InjectionVerdict",
    "InputDecision",
    "InputGuard",
    "OutputDecision",
    "OutputGuard",
    "PIIFinding",
    "detect_injection",
    "find_pii",
    "is_safe_passage",
    "redact_pii",
]
