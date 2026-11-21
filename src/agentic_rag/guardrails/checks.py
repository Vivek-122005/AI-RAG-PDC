"""Input and output guards: the checks that wrap any answer pipeline.

    question ─► InputGuard ─► pipeline ─► OutputGuard ─► answer
                  │ limits, injection,        │ citations valid, no prompt leak,
                  │ PII redaction             │ PII redaction
                  └─► blocked                 └─► refused / cleaned

Both return plain decisions; what to do with them is the caller's job
(see `pipelines.guarded.GuardedPipeline`).
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic_rag.guardrails.injection import detect_injection
from agentic_rag.guardrails.pii import redact_pii

_CITATION = re.compile(r"\[(\d+)\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    question: str  # cleaned and PII-redacted: the only version that reaches the LLM and logs
    reason: str | None = None  # why it was blocked
    redacted: list[str] = field(default_factory=list)  # PII kinds removed
    injection_rules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputGuard:
    min_chars: int = 3
    max_chars: int = 1000
    block_injection: bool = True
    redact: bool = True

    def check(self, question: str) -> InputDecision:
        text = _CONTROL.sub("", unicodedata.normalize("NFKC", question)).strip()
        if len(text) < self.min_chars:
            return InputDecision(False, text, "too_short")
        if len(text) > self.max_chars:
            return InputDecision(False, text[: self.max_chars], "too_long")
        verdict = detect_injection(text)
        if self.block_injection and verdict.flagged:
            return InputDecision(False, text, "prompt_injection", injection_rules=verdict.matched)
        redacted: list[str] = []
        if self.redact:
            text, findings = redact_pii(text)
            redacted = sorted({f.kind for f in findings})
        return InputDecision(True, text, None, redacted, verdict.matched)


@dataclass(frozen=True)
class OutputDecision:
    text: str
    refuse: bool = False  # the caller should replace the answer with a refusal
    reason: str | None = None
    removed_citations: list[int] = field(default_factory=list)
    redacted: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutputGuard:
    """Checks a drafted answer against the evidence it was given."""

    system_prompts: Sequence[str] = ()  # an answer quoting these is a prompt leak
    require_citations: bool = True
    redact: bool = True
    leak_window: int = 60  # characters of a system prompt that count as a leak

    def check(self, text: str, n_contexts: int, *, refused: bool) -> OutputDecision:
        # Leak check first: a leaked prompt contains the refusal sentence, so it would
        # otherwise pass as a harmless refusal.
        if self._leaks_prompt(text):
            return OutputDecision(text, refuse=True, reason="system_prompt_leak")
        if refused:
            return OutputDecision(text)

        invalid = sorted({int(n) for n in _CITATION.findall(text) if not 1 <= int(n) <= n_contexts})
        if invalid:  # a citation to a passage that was never shown is a fabricated source
            text = _CITATION.sub(lambda m: "" if int(m[1]) in invalid else m[0], text)
            text = re.sub(r"[ \t]+([.,;])", r"\1", text).strip()
        if self.require_citations and not _CITATION.search(text):
            return OutputDecision(text, True, "no_valid_citations", invalid)

        redacted: list[str] = []
        if self.redact:
            text, findings = redact_pii(text)
            redacted = sorted({f.kind for f in findings})
        return OutputDecision(text, False, None, invalid, redacted)

    def _leaks_prompt(self, text: str) -> bool:
        norm = " ".join(text.split()).lower()
        for prompt in self.system_prompts:
            p = " ".join(prompt.split()).lower()
            step = max(1, self.leak_window // 2)
            for i in range(0, max(1, len(p) - self.leak_window + 1), step):
                if p[i : i + self.leak_window] in norm:
                    return True
        return False
