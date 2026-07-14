from pathlib import Path

import pandas as pd
import streamlit as st
from langfuse_tracing import (
    flush_tracing,
    observe,
    propagate_attributes,
    update_current_span,
)
from research import normalize_columns, research_school, trace_output

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
SCORED_CSV_PATH = OUTPUT_DIR / "scored_schools.csv"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def show_research_source(research):
    if research.get("_from_cache"):
        web_search_used = research.get("_web_search_used", False)
        if web_search_used:
            st.info("Loaded from cache (originally researched with web search).")
        else:
            st.warning("Loaded from cache (originally researched without web search).")
    elif research.get("_web_search_used"):
        st.success("Fresh API result — web search was used.")
    elif research.get("error"):
        st.error("Research failed — default values used.")
    else:
        st.warning("Fresh API result — web search was not used.")


@observe(name="score-single-school")
def run_single_school(row, force_refresh=False):
    update_current_span(
        input={
            "school_name": row["school_name"],
            "location": row["location"],
            "board_affiliation": row["board_affiliation"],
        }
    )
    with propagate_attributes(tags=["streamlit", "single-school"]):
        result = research_school(row, force_refresh=force_refresh)
    update_current_span(output=trace_output(result))
    flush_tracing()
    return result


@observe(name="batch-score-schools")
def run_batch_scoring(df, force_refresh=False, session_id=None, run_llm_evaluation=False):
    update_current_span(
        input={
            "school_count": len(df),
            "force_refresh": force_refresh,
            "run_llm_evaluation": run_llm_evaluation,
        }
    )

    results = []
    web_search_count = 0
    cache_count = 0

    with propagate_attributes(
        session_id=session_id,
        tags=["streamlit", "batch-upload"],
    ):
        for _, row in df.iterrows():
            result = research_school(row, force_refresh=force_refresh)
            if result.get("_from_cache"):
                cache_count += 1
            elif result.get("_web_search_used"):
                web_search_count += 1
            results.append(
                {
                    "school_name": row["school_name"],
                    "score": result["score"],
                    "conversion_likelihood": result["conversion_likelihood"],
                    "reasoning": result.get("reasoning", ""),
                    "evidence": result.get("evidence", []),
                    "confidence": result.get("confidence", ""),
                    "opportunities": result.get("opportunities", ""),
                    "concerns": result.get("concerns", ""),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)

    if run_llm_evaluation:
        from evaluate import llm_evaluation

        overall_scores = []
        critiques = []
        for _, row in results_df.iterrows():
            judgment = llm_evaluation(row.to_dict())
            overall_scores.append(judgment.get("ev_overall"))
            critiques.append(judgment.get("ev_critique"))
        results_df["ev_overall"] = overall_scores
        results_df["ev_critique"] = critiques

    display_columns = [
        "rank",
        "school_name",
        "score",
        "conversion_likelihood",
        "reasoning",
    ]
    if run_llm_evaluation:
        display_columns.extend(["ev_overall", "ev_critique"])
    results_df = results_df[display_columns]

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(SCORED_CSV_PATH, index=False)

    update_current_span(
        output={
            "schools_scored": len(results_df),
            "web_search_count": web_search_count,
            "cache_count": cache_count,
            "llm_evaluation": run_llm_evaluation,
        }
    )
    flush_tracing()
    return results_df, web_search_count, cache_count


st.set_page_config(page_title="Tilli Lead Scoring", layout="wide")
st.title("Tilli Lead Scoring")

single_tab, batch_tab = st.tabs(["Single School", "Batch Upload"])

with single_tab:
    st.subheader("Score a single school")

    school_name = st.text_input("School name")
    location = st.text_input("Location")
    size = st.number_input("Size (students)", min_value=1, value=500, step=1)
    board_affiliation = st.text_input(
        "Board affiliation", placeholder="e.g. CBSE, ICSE, IB, IGCSE"
    )
    force_refresh = st.checkbox("Force refresh (bypass cache and run fresh web search)")

    if st.button("Score this school", type="primary"):
        if not school_name or not location or not board_affiliation:
            st.error("Please fill in school name, location, and board affiliation.")
        else:
            row = pd.Series(
                {
                    "school_name": school_name,
                    "location": location,
                    "size": size,
                    "board_affiliation": board_affiliation,
                }
            )

            with st.spinner(f"Researching {school_name}..."):
                result = run_single_school(row, force_refresh=force_refresh)

            show_research_source(result)

            col1, col2 = st.columns(2)
            col1.metric("Score", f"{result['score']}/100")
            col2.metric("Conversion likelihood", result["conversion_likelihood"])

            st.markdown("**Reasoning**")
            st.write(result.get("reasoning", ""))

            st.markdown("**Evidence**")
            st.write(result.get("evidence", []))

            st.markdown("**Opportunities**")
            st.write(result.get("opportunities", ""))

            st.markdown("**Concerns**")
            st.write(result.get("concerns", ""))

with batch_tab:
    st.subheader("Score schools from CSV")
    st.caption(
        "Upload a CSV with school_name, location, board_affiliation, and size. "
        "Each row is researched and scored via the API — existing score columns in the file are ignored."
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    force_refresh = st.checkbox(
        "Force refresh (bypass cache for all schools)", key="batch_force_refresh"
    )
    run_llm_evaluation = st.checkbox(
        "Run LLM evaluation", key="batch_run_llm_evaluation"
    )

    if uploaded_file is not None:
        df = normalize_columns(pd.read_csv(uploaded_file))

        required_columns = ["school_name", "location", "board_affiliation", "size"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        else:
            df = df[required_columns].copy()
            df["size"] = pd.to_numeric(df["size"], errors="coerce")

            file_key = (
                f"{uploaded_file.name}:{uploaded_file.size}:"
                f"refresh={force_refresh}:eval={run_llm_evaluation}"
            )

            if st.session_state.get("batch_file_key") != file_key:
                spinner_text = (
                    "Running batch scoring and LLM evaluation..."
                    if run_llm_evaluation
                    else "Running batch scoring..."
                )
                with st.spinner(spinner_text):
                    results_df, web_search_count, cache_count = run_batch_scoring(
                        df,
                        force_refresh=force_refresh,
                        session_id=file_key,
                        run_llm_evaluation=run_llm_evaluation,
                    )

                st.session_state.batch_file_key = file_key
                st.session_state.batch_results = results_df
                st.session_state.batch_web_search_count = web_search_count
                st.session_state.batch_cache_count = cache_count

            results_df = st.session_state.get("batch_results")
            if results_df is not None:
                web_search_count = st.session_state.get("batch_web_search_count", 0)
                cache_count = st.session_state.get("batch_cache_count", 0)
                success_msg = (
                    f"Scored {len(results_df)} schools. "
                    f"Web search used for {web_search_count}; loaded from cache for {cache_count}."
                )
                if "ev_overall" in results_df.columns:
                    success_msg += " LLM evaluation columns included."
                st.success(success_msg)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                download_name = (
                    "evaluated_schools.csv"
                    if "ev_overall" in results_df.columns
                    else "scored_schools.csv"
                )
                st.download_button(
                    label="Download results CSV",
                    data=results_df.to_csv(index=False),
                    file_name=download_name,
                    mime="text/csv",
                )
