import json
import os
import re

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Output"
ENV_PATH = BASE_DIR / ".env"
EVALUATED_CSV_PATH = OUTPUT_DIR / "evaluated_schools.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(ENV_PATH)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

JUDGE_COLUMNS = [
    "ev_verifiability",
    "ev_no_hallucination",
    "ev_intent_alignment",
    "ev_edge_case",
    "ev_clarity",
    "ev_overall",
    "ev_critique",
]


def _failed_judge_result():
    return {
        "ev_verifiability": 0,
        "ev_no_hallucination": 0,
        "ev_intent_alignment": 0,
        "ev_edge_case": 0,
        "ev_clarity": 0,
        "ev_overall": 0,
        "ev_critique": "Judge call failed",
    }


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def llm_evaluation(result):
    """Judge a single research result with Claude using the EVIDENCE framework."""
    try:
        prompt = f"""You are evaluating the quality of an AI-generated school research output for a lead scoring tool. The tool helps a sales team at Tilli, an EdTech company selling SEL programmes to schools in India, prioritise which schools to approach.

Evaluate the following research output using the EVIDENCE framework. Score each dimension from 1 to 5 where 1 is very poor and 5 is excellent.

School: {result.get("school_name", "")}
Score given: {result.get("score", "")}/100
Conversion likelihood: {result.get("conversion_likelihood", "")}
Reasoning: {result.get("reasoning", "")}
Evidence items: {result.get("evidence", "")}
Confidence: {result.get("confidence", "")}
Opportunities: {result.get("opportunities", "")}
Concerns: {result.get("concerns", "")}

Dimensions to evaluate:

Verifiability (1-5): Can the evidence items be checked against real sources, or are they vague and unverifiable?
No hallucination (1-5): Does the evidence list contain only specific school-level facts, or does it include generic inferences presented as verified facts?
Intent alignment (1-5): Does the output give a salesperson what they need to prioritise and approach this school, or does it describe the school generically without actionable direction?
Edge case handling (1-5): If the school could not be found or data was ambiguous, was this clearly flagged rather than covered up with confident-sounding inferences?
Clarity (1-5): Is the reasoning concise and actionable for a non-technical sales user?

Return ONLY a JSON object with no extra text:
{{
"ev_verifiability": <1-5>,
"ev_no_hallucination": <1-5>,
"ev_intent_alignment": <1-5>,
"ev_edge_case": <1-5>,
"ev_clarity": <1-5>,
"ev_overall": <average of the five scores, rounded to 1 decimal place>,
"ev_critique": "<one sentence identifying the single biggest weakness in this research output>"
}}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = next(
            block.text for block in reversed(response.content) if hasattr(block, "text")
        )
        return _extract_json(raw)
    except Exception:
        return _failed_judge_result()


def _resolve_csv_path(scored_csv_path):
    path = Path(scored_csv_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def run_evaluation(scored_csv_path, conversion_data=None):
    """Evaluate every school in a scored CSV and write Output/evaluated_schools.csv."""
    csv_path = _resolve_csv_path(scored_csv_path)
    df = pd.read_csv(csv_path)

    judge_rows = []
    for _, row in df.iterrows():
        evaluation = llm_evaluation(row.to_dict())
        judge_rows.append(evaluation)
        print(f"{row.get('school_name', 'Unknown')}: ev_overall={evaluation.get('ev_overall')}")

    judge_df = pd.DataFrame(judge_rows)
    for column in JUDGE_COLUMNS:
        if column not in judge_df.columns:
            judge_df[column] = 0 if column != "ev_critique" else "Judge call failed"

    evaluated_df = pd.concat([df.reset_index(drop=True), judge_df[JUDGE_COLUMNS]], axis=1)

    score_cols = [
        "ev_verifiability",
        "ev_no_hallucination",
        "ev_intent_alignment",
        "ev_edge_case",
        "ev_clarity",
        "ev_overall",
    ]
    for column in score_cols:
        evaluated_df[column] = pd.to_numeric(evaluated_df[column], errors="coerce")

    avg_overall = evaluated_df["ev_overall"].mean()
    dimension_avgs = {
        column: evaluated_df[column].mean()
        for column in [
            "ev_verifiability",
            "ev_no_hallucination",
            "ev_intent_alignment",
            "ev_edge_case",
            "ev_clarity",
        ]
    }
    weakest_dimension = min(dimension_avgs, key=dimension_avgs.get)

    print(f"\nAverage ev_overall: {avg_overall:.2f}")
    print(
        f"Lowest-scoring dimension: {weakest_dimension} "
        f"(avg={dimension_avgs[weakest_dimension]:.2f})"
    )

    # Accuracy evaluation (optional ground truth)
    if conversion_data:
        evaluated_df["converted"] = evaluated_df["school_name"].map(conversion_data)
        labeled = evaluated_df.dropna(subset=["converted"]).copy()
        labeled["converted"] = labeled["converted"].astype(int)

        if labeled.empty:
            print("\nAccuracy evaluation skipped: no matching school names in conversion_data.")
        else:
            ranked = labeled.sort_values("score", ascending=False)
            top_n = max(1, int(len(ranked) * 0.25))
            top_quartile = ranked.head(top_n)
            precision = top_quartile["converted"].mean()

            converted_avg = labeled.loc[labeled["converted"] == 1, "score"].mean()
            not_converted_avg = labeled.loc[labeled["converted"] == 0, "score"].mean()

            print(f"\nPrecision @ top quartile: {precision:.1%} ({top_n} schools)")
            print(f"Average score (converted): {converted_avg:.2f}")
            print(f"Average score (not converted): {not_converted_avg:.2f}")
    else:
        print(
            "\nAccuracy evaluation skipped: no conversion_data provided.\n"
            "Pass a dict of school_name -> 1/0, e.g.\n"
            "  run_evaluation('Output/scored_schools.csv', "
            "conversion_data={'School A': 1, 'School B': 0})"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    evaluated_df.to_csv(EVALUATED_CSV_PATH, index=False)
    print(f"\nSaved evaluations to {EVALUATED_CSV_PATH}")
    return evaluated_df


if __name__ == "__main__":
    run_evaluation("Output/scored_schools.csv")
