"""Heuristic prompt-injection detection for user questions and retrieved documents.

Two threats, one detector:
- direct injection: the user's question tries to override the system instructions;
- indirect injection: a document in the corpus carries instructions aimed at the model
  ("ignore previous instructions and ..."), which arrive inside retrieved evidence.

Rules are weighted; a text is flagged when the summed weight reaches a threshold, so a
single weak signal (the word "ignore", a role-play phrase) does not block a normal
question. A trained classifier would catch paraphrases these rules miss; it can replace
`detect_injection` behind the same signature when RAM allows.
"""

import re
import unicodedata
from dataclasses import dataclass, field

BLOCK_THRESHOLD = 1.0


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    weight: float


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_TARGET = r"(?:instructions?|prompts?|rules|directions|guidelines|context|messages?)"
RULES: tuple[Rule, ...] = (
    Rule(
        "override_instructions",
        _rx(
            r"\b(?:ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,40}?"
            rf"\b(?:previous|prior|above|earlier|all|any|your|the|system|these)\b[^.\n]{{0,20}}?\b{_TARGET}"
        ),
        1.0,
    ),
    Rule("new_instructions", _rx(r"\b(?:new|updated|real|actual)\s+instructions?\s*:"), 1.0),
    Rule(
        "reveal_prompt",
        _rx(
            r"\b(?:reveal|show|print|repeat|output|leak|tell me)\b[^.\n]{0,30}?"
            r"\b(?:system|hidden|initial|original|developer)\s+(?:prompt|instructions?|message)"
        ),
        1.0,
    ),
    Rule(
        "role_override",
        _rx(
            r"\byou are (?:now|no longer)\b|\bfrom now on,? you\b|\bact as (?:if you (?:are|were) )?"
            r"(?:an? )?(?:unrestricted|unfiltered|evil|jailbroken)"
        ),
        1.0,
    ),
    Rule(
        "jailbreak_terms",
        _rx(r"\b(?:jailbreak|DAN mode|developer mode|do anything now|god mode)\b"),
        1.0,
    ),
    # Chat-template control tokens never occur in a genuine question or paper text.
    Rule(
        "chat_template_token",
        _rx(r"<\|im_(?:start|end)\|>|\[/?INST\]|<\|?(?:system|assistant|user)\|>|<<SYS>>"),
        1.0,
    ),
    Rule("role_line", _rx(r"(?:^|\n)\s*(?:#{2,}\s*)?(?:system|assistant)\s*:"), 0.6),
    Rule(
        "constraint_removal",
        _rx(
            r"\bwithout (?:any )?(?:citations|restrictions|rules|limits|filters|censorship|guardrails)\b"
        ),
        0.5,
    ),
    Rule("evidence_breakout", _rx(r"</?\s*evidence\b"), 0.6),
    Rule(
        "directive_to_model",
        _rx(
            r"\b(?:AI|assistant|language model|LLM|chatbot)s?\b[^.\n]{0,40}?\b(?:must|should|will)\b"
            r"[^.\n]{0,20}?\b(?:say|respond|reply|answer|output|print)\b"
        ),
        0.5,
    ),
    Rule(
        "refuse_answer_trick", _rx(r"\b(?:instead|rather),?\s+(?:say|respond|reply|output)\b"), 0.5
    ),
)


@dataclass(frozen=True)
class InjectionVerdict:
    score: float
    matched: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.score >= BLOCK_THRESHOLD


def _normalise(text: str) -> str:
    """Undo common obfuscation: full-width letters, zero-width characters, odd spacing."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[​-‏⁠﻿]", "", text)
    return re.sub(r"[ \t]+", " ", text)


def detect_injection(text: str) -> InjectionVerdict:
    norm = _normalise(text)
    matched = [r.name for r in RULES if r.pattern.search(norm)]
    weights = {r.name: r.weight for r in RULES}
    return InjectionVerdict(round(sum(weights[m] for m in matched), 2), matched)


def is_safe_passage(text: str) -> bool:
    """True unless a retrieved passage looks like it carries instructions for the model."""
    return not detect_injection(text).flagged
