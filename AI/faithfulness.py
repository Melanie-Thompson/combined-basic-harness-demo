"""Response faithfulness and pre-execution relevance checker.

Two-stage verification pipeline:
1. Pre-execution relevance (`check_relevance`) — verifies that selected tools
   match the user's intent via tool descriptions before execution.
2. Post-execution faithfulness (`check_faithfulness`) — verifies that the AI's
   response accurately reflects tool outputs using per-tool deterministic
   extractors (numeric, temporal, exact with word boundaries and negation guards)
   with semantic similarity fallback for dynamic/unschematized MCP tools.
"""

from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Callable

from dateutil import parser as dtparser

from AI.embeddings import semantic_similarity
from AI.models.message import FaithfulnessResult, ToolExecutionRecord
from AI.tools import TOOL_SCHEMAS, Extractor

log = logging.getLogger("faithfulness")

SEMANTIC_THRESHOLD = 0.4
RELEVANCE_THRESHOLD = 0.25

# Default rounding tolerance allows natural LLM rounding (e.g., 11.4 -> 11)
DEFAULT_NUMERIC_ABS_TOL = 0.5
DEFAULT_NUMERIC_REL_TOL = 0.01

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Extract candidate time/date phrases instead of feeding entire paragraphs to dateutil
_TEMPORAL_CANDIDATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?\s*(?:[aApP]\.?[mM]\.?|UTC|GMT)?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:,?\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)

_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:no|not|never|without|zero|none|neither|nor|n't)\b\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)


def _check_numeric(
    value: str,
    ai_text: str,
    abs_tol: float = DEFAULT_NUMERIC_ABS_TOL,
    rel_tol: float = DEFAULT_NUMERIC_REL_TOL,
) -> bool:
    """Check if a numeric value appears in ai_text within rounding or percentage tolerance."""
    try:
        expected = float(value.replace(",", ""))
    except ValueError:
        return False

    # Support ratio-to-percentage representation (e.g., tool "0.84" -> response "84%")
    candidates = [expected]
    if 0.0 < abs(expected) < 1.0:
        candidates.append(expected * 100.0)

    for match in _NUM_RE.finditer(ai_text):
        try:
            found = float(match.group().replace(",", ""))
        except ValueError:
            continue

        for target in candidates:
            tol = max(abs_tol, abs(target) * rel_tol)
            if abs(found - target) <= tol:
                return True

    return False


def _to_utc_naive(dt: datetime) -> datetime:
    """Convert timezone-aware datetimes to UTC before stripping tzinfo."""
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _check_temporal(value: str, ai_text: str) -> bool:
    """Check if a date or timestamp from tool output is faithfully represented in ai_text."""
    stripped_val = value.strip()
    is_date_only = bool(_DATE_ONLY_RE.match(stripped_val))

    try:
        expected_raw = dtparser.parse(stripped_val)
    except (ValueError, OverflowError):
        return False

    expected_utc = _to_utc_naive(expected_raw)

    # Extract localized temporal substrings rather than fuzzy-parsing the entire paragraph
    for match in _TEMPORAL_CANDIDATE_RE.finditer(ai_text):
        snippet = match.group().strip()
        try:
            parsed_raw = dtparser.parse(snippet, default=expected_raw)
            parsed_utc = _to_utc_naive(parsed_raw)
        except (ValueError, OverflowError):
            continue

        if is_date_only:
            if parsed_utc.date() == expected_utc.date():
                return True
        else:
            if abs(parsed_utc - expected_utc) <= timedelta(minutes=2):
                return True

    # Cross-platform fallback for explicit HH:MM (24h and 12h) strings
    if not is_date_only:
        hm_24 = expected_utc.strftime("%H:%M")
        hour_12 = expected_utc.hour % 12 or 12
        hm_12 = f"{hour_12}:{expected_utc.minute:02d}"
        if re.search(rf"\b(?:{re.escape(hm_24)}|{re.escape(hm_12)})\b", ai_text):
            return True

    return False


def _check_exact(value: str, ai_text: str) -> bool:
    """Case-insensitive whole-word match with basic negation awareness."""
    cleaned = value.strip()
    if not cleaned:
        return True

    pattern = re.compile(rf"\b{re.escape(cleaned)}\b", re.IGNORECASE)
    for match in pattern.finditer(ai_text):
        prefix = ai_text[max(0, match.start() - 32) : match.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return True

    return False


_CHECKERS: dict[str, Callable[[str, str], bool]] = {
    "exact": _check_exact,
    "numeric": _check_numeric,
    "temporal": _check_temporal,
}


def check_relevance(
    user_message: str,
    tool_names: list[str],
    tool_descriptions: dict[str, str],
    allow_chained_helpers: bool = True,
) -> tuple[bool, list[str]]:
    """Verify selected tools against the user's question using tool descriptions.

    When `allow_chained_helpers` is True and multiple tools are invoked in a
    trajectory (e.g., `geocode` -> `weather_fetch`), relevance passes if at least
    the primary/terminal tool aligns with the user's question so intermediate
    lookup helpers are not falsely blocked.
    """
    if not tool_names or not user_message or not user_message.strip():
        return True, []

    mismatches: list[str] = []
    relevant_count = 0

    for name in tool_names:
        desc = (tool_descriptions.get(name) or "").strip()
        if not desc:
            mismatches.append(f"tool '{name}' has no description — cannot verify relevance")
            continue

        desc_score = semantic_similarity(user_message, desc)
        name_score = semantic_similarity(user_message, name.replace("_", " "))
        score = max(desc_score, name_score)
        log.info(
            "relevance: '%s' vs tool '%s' -> %.3f (desc=%.3f, name=%.3f, threshold %.2f)",
            user_message[:60],
            name,
            score,
            desc_score,
            name_score,
            RELEVANCE_THRESHOLD,
        )
        if score >= RELEVANCE_THRESHOLD:
            relevant_count += 1
        else:
            mismatches.append(
                f"tool '{name}' ({desc[:60]}) not relevant to question "
                f"(similarity {score:.3f} < {RELEVANCE_THRESHOLD})"
            )

    if allow_chained_helpers and len(tool_names) > 1 and relevant_count >= 1:
        return True, []

    return len(mismatches) == 0, mismatches


def check_faithfulness(
    ai_text: str,
    records: list[ToolExecutionRecord],
    custom_extractors: dict[str, Extractor] | None = None,
) -> dict[str | None, FaithfulnessResult]:
    """Evaluate faithfulness for each tool execution record against the final AI response.

    Can be invoked directly by both runtime guardrails and `run_benchmark`
    to guarantee identical verification behavior across CI and production.
    """
    results: dict[str | None, FaithfulnessResult] = {}
    cleaned_response = (ai_text or "").strip()

    for record in records:
        tool_out = (record.tool_output or "").strip()

        if not cleaned_response:
            results[record.tool_call_id] = FaithfulnessResult(
                semantic_score=0.0,
                contained=False,
                faithful=False,
                mismatches=["empty_response"],
            )
            continue

        if not tool_out:
            results[record.tool_call_id] = FaithfulnessResult(
                semantic_score=0.0,
                contained=False,
                faithful=False,
                mismatches=["empty_tool_output"],
            )
            continue

        score = semantic_similarity(tool_out, cleaned_response)
        contained = True
        mismatches: list[str] = []

        extractor: Extractor | None = None
        if custom_extractors and record.tool_name in custom_extractors:
            extractor = custom_extractors[record.tool_name]
        else:
            schema = TOOL_SCHEMAS.get(record.tool_name)
            extractor = schema.get("extractor") if schema else None

        if extractor is not None:
            extracted_items = list(extractor(tool_out))
            for value, kind in extracted_items:
                checker = _CHECKERS.get(kind, _check_exact)
                if not checker(str(value), cleaned_response):
                    contained = False
                    mismatches.append(f"{kind}:{value}")
        else:
            # Fallback for dynamic MCP tools without registered extractors
            if score < SEMANTIC_THRESHOLD:
                contained = False
                mismatches.append(f"semantic:{score:.2f}<{SEMANTIC_THRESHOLD}")

        results[record.tool_call_id] = FaithfulnessResult(
            semantic_score=score,
            contained=contained,
            faithful=contained,
            mismatches=mismatches,
        )

    return results