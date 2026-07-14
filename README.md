# Tilli AI Lead Scoring Pipeline

An AI-powered pipeline that researches prospective partner schools, scores them by conversion likelihood, and ranks them so Tilli's sales team knows which leads to focus on.

---

## The Problem

Tilli's sales team was identifying partner schools manually, which is a process that isn't easily scalable. This project automates school research and lead scoring using Claude, web search, and a structured framework to evaluate Claude's responses.

---

## How It Works

1. **Ingests** a CSV of prospective schools (name, location, size, board affiliation)
2. **Researches** each school using Claude with live web search — pulling signals on SEL programmes, tech readiness, leadership profile, and reputation
3. **Scores** each school from 0–100 and assigns a conversion likelihood tier (High / Medium / Low / Very Low)
4. **Ranks** all schools and exports a scored spreadsheet for the sales team
5. **Evaluates** output quality using an LLM as a judge, and implements a framework that checks the responses' exactness, verifiability, determinism, edge case handling, clarity, etc.

---

## Tech Stack

- **Language:** Python
- **AI:** Anthropic Claude (claude-haiku-4-5-20251001) with web search
- **UI:** Streamlit
- **Libraries:** pandas, anthropic, python-dotenv, openpyxl
- **Notebook:** Jupyter

---

## Project Structure

```
TAI-Tilli-LeadScoringProject/
├── app.py              # Streamlit UI
├── research.py         # Core pipeline logic
├── evaluate.py         # LLM-as-judge evaluation
├── Data/               # Input school data
├── Output/             # Scored and evaluated results
└── Notebooks/          # Development and testing notebook
```

---

## Setup

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Add your Anthropic API key to a `.env` file: `ANTHROPIC_API_KEY=your_key_here`
3. Run the UI: `streamlit run app.py`

---

## Evaluation

Output quality is assessed using the EVIDENCE framework — a structured method that scores each research result across verifiability, no hallucination, intent alignment, edge case handling, and clarity. Each dimension is scored 1–5 by a second Claude call acting as a judge.

When Tilli provides historical conversion data, the pipeline can also measure accuracy — specifically, what percentage of schools Claude ranked in the top quartile actually converted.

---

## Limitations

- Research quality depends on publicly available information. Smaller schools with limited web presence will score with lower confidence.
- Scoring weights have not yet been backtested against real conversion data — calibration is pending Tilli's historical records.

