"""Small helpers for reading token usage from OpenAI-compatible responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported for one completed model request."""

    input_tokens: int
    output_tokens: int


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def extract_token_usage(response: object) -> TokenUsage | None:
    """Return usage only when both provider token counters are trustworthy.

    OpenAI-compatible SDKs may expose ``usage`` either as an object or a
    mapping.  Missing or malformed counters are treated as unavailable rather
    than silently written as zero in the experiment results.
    """

    usage = _field(response, "usage")
    if usage is None:
        return None
    input_tokens = _non_negative_int(_field(usage, "input_tokens"))
    output_tokens = _non_negative_int(_field(usage, "output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
