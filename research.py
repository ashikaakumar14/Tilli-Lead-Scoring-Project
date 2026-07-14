import json
import os
import re
from pathlib import Path

import anthropic

from langfuse_tracing import (
    init_tracing,
    observe,
    propagate_attributes,
    update_current_span,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
CACHE_PATH = OUTPUT_DIR / "cache.json"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

init_tracing()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

cache = {}
if CACHE_PATH.exists():
    with CACHE_PATH.open("r") as f:
        cache = json.load(f)


def is_valid_research_result(entry):
    return (
        isinstance(entry, dict)
        and "score" in entry
        and "conversion_likelihood" in entry
    )


def _school_input(row):
    return {
        "school_name": row["school_name"],
        "location": row["location"],
        "size": row["size"],
        "board_affiliation": row["board_affiliation"],
    }


def trace_output(result):
    """Minimal fields sent to Langfuse trace output."""
    return {
        "score": result.get("score"),
        "conversion_likelihood": result.get("conversion_likelihood"),
        "confidence": result.get("confidence"),
    }


def _school_output(result):
    return {
        **trace_output(result),
        "_from_cache": result.get("_from_cache", False),
        "_web_search_used": result.get("_web_search_used", False),
    }


def _collect_response_text(response) -> str:
    """Join all text blocks from a Claude response (web-search replies may span several)."""
    parts = [
        block.text
        for block in response.content
        if hasattr(block, "text") and block.text
    ]
    return "\n".join(parts) if parts else ""


def _extract_balanced_json(text: str) -> str | None:
    """Return the first top-level JSON object found via brace matching."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)

    return None


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _json_candidates(raw: str) -> list[str]:
    """Build ordered parse candidates from a Claude response."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    add(raw)

    stripped = _strip_markdown_fences(raw)
    add(stripped)

    for marker in ("```json", "```"):
        search_from = 0
        while True:
            start = raw.find(marker, search_from)
            if start == -1:
                break

            content_start = start + len(marker)
            if marker == "```":
                remainder = raw[content_start : content_start + 4]
                if remainder.lower() == "json":
                    content_start += 4
            if content_start < len(raw) and raw[content_start] == "\n":
                content_start += 1

            end_fence = raw.find("```", content_start)
            inner = raw[content_start:end_fence] if end_fence != -1 else raw[content_start:]
            add(_strip_markdown_fences(inner))
            add(_extract_balanced_json(inner))
            search_from = start + len(marker)

    for source in (raw, stripped):
        add(_extract_balanced_json(source))

    last_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if last_match:
        add(last_match.group())

    return candidates


def _parse_research_json(raw: str, school_name: str) -> dict:
    """Parse Claude's research output into a JSON object."""
    if not raw or not raw.strip():
        print(f"[research] Empty Claude response for {school_name!r}")
        raise ValueError(f"Empty response from Claude for {school_name}")

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if isinstance(parsed, dict):
            return parsed

    print(f"[research] Failed to parse JSON for {school_name!r}. Raw Claude response:\n{raw}")
    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not find JSON object in Claude response for {school_name}")


@observe(name="research-school")
def research_school(row, force_refresh=False):
    school_name = row["school_name"]
    school_meta = _school_input(row)

    with propagate_attributes(
        tags=["tilli", "school-research"],
        metadata={"board_affiliation": row["board_affiliation"], "location": row["location"]},
    ):
        update_current_span(input=school_meta)

        if not force_refresh and school_name in cache:
            cached = cache[school_name]
            if is_valid_research_result(cached):
                result = cached.copy()
                result["_from_cache"] = True
                update_current_span(
                    output=_school_output(result),
                    metadata={"cache_hit": True},
                )
                return result

        prompt = f"""You are a school research analyst for an EdTech sales team at Tilli, a company that provides Social-Emotional Learning (SEL) programmes to schools across India.

Your task is to research the following school using web search and produce a structured assessment of how likely they are to convert into a Tilli partner school.

SCHOOL DETAILS:
- Name: {row['school_name']}
- Location: {row['location']}
- Size: {row['size']} students
- Board Affiliation: {row['board_affiliation']}

RESEARCH INSTRUCTIONS:
Search the web for this specific school. Look for:
1. Evidence of existing SEL, wellbeing, or counselling programmes
2. Technology adoption — devices, edtech platforms, digital initiatives
3. Leadership profile — principal's public statements, vision, awards
4. School reputation — rankings, press mentions, parent reviews
5. Any signals of openness to innovation or external partnerships

SCORING RULES:
- Base your score primarily on what you find via web search
- If specific evidence is not available, you may make reasonable inferences based on board type, location, size, and school type — but clearly flag these as inferences in your reasoning
- Inferences should never appear in the evidence list — only verified findings go there
- Confidence should be low whenever the score relies heavily on inference rather than verified findings

CONVERSION LIKELIHOOD TIERS:
- 75 to 100 = High
- 50 to 74 = Medium  
- 25 to 49 = Low
- 0 to 24 = Very Low

EXAMPLES OF GOOD OUTPUT:

Example 1 — High scoring school:
School: Inventure Academy, Bangalore, 800 students, IB board

{{
  "score": 82,
  "conversion_likelihood": "High",
  "reasoning": "Inventure Academy has a documented focus on social-emotional learning through their advisory programme and explicitly mentions student wellbeing on their website. The school has adopted multiple edtech platforms including Google Workspace and has a 1:1 device policy. The principal has spoken publicly about holistic education at TEDx Bangalore. Score is based primarily on verified web findings with high confidence.",
  "evidence": [
    "School website has a dedicated Student Wellbeing page describing advisory periods and counselling support",
    "Principal featured in Bangalore Mirror discussing importance of emotional intelligence in education",
    "School listed as Google Reference School indicating high tech adoption",
    "IB curriculum framework explicitly includes social-emotional learning components"
  ],
  "opportunities": "Strong alignment with Tilli's SEL offering. Principal is publicly vocal about wellbeing — direct outreach referencing their existing programmes would resonate well.",
  "concerns": "IB schools often have existing SEL frameworks — need to position Tilli as complementary rather than replacement.",
  "confidence": "high"
}}

Example 2 — Low scoring school with inferences:
School: Rajendra Prasad Government School, Jaipur, 1200 students, CBSE board

{{
  "score": 22,
  "conversion_likelihood": "Very Low",
  "reasoning": "No publicly available evidence of SEL programmes or wellbeing initiatives found via web search. No school website or leadership profile found. Score is primarily based on inference — government schools in Tier 2 cities typically have limited budgets for third-party EdTech programmes and lower likelihood of proactive SEL adoption. These are inferences, not verified findings.",
  "evidence": [
    "No school website found",
    "No mentions in local news or education publications"
  ],
  "opportunities": "Large student population could be valuable if budget constraints are addressed through government partnerships or subsidised pricing.",
  "concerns": "No digital presence, no evidence of SEL awareness, likely budget constrained. Score is largely inferred due to limited public information — recommend verifying before deprioritising entirely.",
  "confidence": "low"
}}

Example 3 — Medium scoring school with mixed evidence:
School: DAV Public School, Chandigarh, 1500 students, CBSE board

{{
  "score": 58,
  "conversion_likelihood": "Medium",
  "reasoning": "DAV Public School Chandigarh has a functional website with some mention of student development activities but no specific SEL programme documented. Tech readiness appears moderate based on mention of a computer lab and participation in a state-level science exhibition. Leadership profile not found via web search — inferred as neutral based on school size and establishment year. DAV schools nationally have a reputation for academic rigour which may or may not translate to SEL openness.",
  "evidence": [
    "School website found with details on extracurricular activities and student clubs",
    "School mentioned in Tribune India for science exhibition participation",
    "Computer lab and library facilities listed on website"
  ],
  "opportunities": "Established school with active student programmes — likely receptive to structured SEL if positioned as complementary to academic goals.",
  "concerns": "No explicit SEL or wellbeing focus found. Leadership stance on SEL unknown. May require more education-focused outreach before conversion.",
  "confidence": "mid"
}}

Now research {row['school_name']} in {row['location']} and return your assessment.

Return ONLY a JSON object with exactly these fields — no prose before or after, no markdown backticks:
{{
  "score": <number between 0 and 100>,
  "conversion_likelihood": "<High/Medium/Low/Very Low>",
  "reasoning": "<paragraph explaining the score, clearly distinguishing between verified findings and inferences>",
  "evidence": ["<specific thing found via web search>", "<another specific finding>"],
  "opportunities": "<what makes this school a good target or how to approach them>",
  "concerns": "<what might make conversion difficult>",
  "confidence": "<low/mid/high>"
}}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        raw = _collect_response_text(response)
        result = _parse_research_json(raw, school_name)

        result["_web_search_used"] = any(
            hasattr(block, "type") and block.type == "web_search_tool_result"
            for block in response.content
        )
        result["_from_cache"] = False

        cache[school_name] = result
        OUTPUT_DIR.mkdir(exist_ok=True)
        with CACHE_PATH.open("w") as f:
            json.dump(cache, f, indent=2)

        update_current_span(
            output=_school_output(result),
            metadata={"cache_hit": False, "web_search_used": result["_web_search_used"]},
        )
        return result


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df
