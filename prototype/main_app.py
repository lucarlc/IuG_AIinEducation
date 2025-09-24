from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
import os
from prompts import QUESTION_GENERATION_PROMPT
from rag_utils import setup_vectorstores, get_generation_contexts, load_specs_for_evaluation
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from llm_judge import evaluate_question, export_results_to_csv
from rag_utils import load_specs_for_evaluation
from typing import Dict


load_dotenv(find_dotenv(), override=True)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
DATA_ROOT = os.getenv("DATA_ROOT", "data")
RUNS_DIR = os.getenv("RUNS_DIR", "runs")

def _is_valid_specs(s) -> bool:
    # akzeptiere int-keys 1..5 ODER string-keys "1".."5", Werte: nicht-leere Strings
    if not isinstance(s, dict) or not s:
        return False
    keys = set(s.keys())
    ok_int = keys == {1,2,3,4,5}
    ok_str = keys == {"1","2","3","4","5"}
    if not (ok_int or ok_str):
        return False
    return all(isinstance(v, str) and v.strip() for v in s.values())

def ensure_specs_snapshot():
    """Baut einen gültigen specs-Snapshot, falls er fehlt/inkonsistent ist."""
    if "stores" not in st.session_state or not st.session_state.stores:
        return  # stores werden an anderer Stelle gebaut
    if "specs" not in st.session_state or not _is_valid_specs(st.session_state.specs):
        s = load_specs_for_evaluation(st.session_state.stores)
        if _is_valid_specs(s):
            st.session_state.specs = s

st.title("AbiBuddy – Abituraufgaben Generator & Evaluator")

# Initialisiere Session State für Robustheit bei Refresh
if "stores" not in st.session_state:
    st.session_state.stores = setup_vectorstores()
if "generated_question" not in st.session_state:
    st.session_state.generated_question = ""
    st.session_state.generated_origin = ""
if "external_question" not in st.session_state:
    st.session_state.external_question = ""
if "external_origin" not in st.session_state:
    st.session_state.external_origin = ""
if "evaluated_questions" not in st.session_state:
    st.session_state.evaluated_questions = []
if "contexts" not in st.session_state:
    st.session_state.contexts = None

llm = ChatOpenAI(temperature=OPENAI_TEMPERATURE, top_p=1.0, model=OPENAI_MODEL)
MODEL_TAG = f"{getattr(llm, 'model_name', OPENAI_MODEL)}_t{OPENAI_TEMPERATURE}_p{getattr(llm, 'top_p', 1.0)}"

st.subheader("🧠 Eigene Abituraufgabe generieren")
if st.button("Neue Abituraufgabe generieren"):
    ensure_specs_snapshot()  # vor Generierung absichern
    contexts = get_generation_contexts(st.session_state.stores)
    st.session_state.contexts = contexts

# Baue vollständigen Prompt mit Template
    full_prompt = QUESTION_GENERATION_PROMPT.format(
    specs=contexts["specs"],
    pool=contexts["pool"],
    evals=contexts["evals"]
        )
    st.session_state.full_prompt = full_prompt

# Schicke an LLM 
    response = llm.invoke(full_prompt)
    st.session_state.generated_question = response.content.strip()
    st.session_state.generated_origin = "AbiBuddy"
    st.success("Neue Aufgabe wurde generiert.")

# Audit-Log: Kontexte + Prompt + Params + Specs-Snapshot
    import json, hashlib, time, os
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = str(int(time.time()))
    # speichere specs_map (Snapshot)
    with open(f"{RUNS_DIR}/{run_id}_specs_map.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.specs, f, ensure_ascii=False, indent=2)
    # speichere Kontexte + Prompt + Hash
    blob = {
        "model": MODEL_TAG,
        "prompt_template": "QUESTION_GENERATION_PROMPT",
        "contexts": {k: st.session_state.contexts[k] for k in ["specs","pool","evals"]},
        "full_prompt": full_prompt,
    }
    blob["prompt_hash"] = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
    with open(f"{RUNS_DIR}/{run_id}_generation_context.json", "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    st.session_state.last_run_id = run_id

if st.session_state.generated_question:
    st.text_area("Generierte Abituraufgabe", st.session_state.generated_question, height=300)

st.divider()
st.subheader("📥 Externe Abituraufgabe eingeben ...")
external_question = st.text_area("Frage eines anderen Systems", "", height=200)
external_origin = st.text_input("Herkunft (z.B. 'mistral', 'gemini', 'gpt-4 raw')")

if st.button("Externe Frage zur Bewertung übernehmen"):
    if not external_question.strip() or not external_origin.strip():
        st.warning("Bitte sowohl Frage als auch Herkunft angeben.")
    else:
        st.session_state.generated_question = external_question
        st.session_state.generated_origin = external_origin
        st.success(f"Frage aus '{external_origin}' übernommen.")

st.divider()

if "generated_question" in st.session_state and st.session_state.generated_question:
    st.subheader("📊 Frage bewerten und exportieren")
    if st.button("Evaluieren & CSV exportieren"):
        with st.spinner("Bewertung läuft..."):
            ensure_specs_snapshot()  # vor Bewertung absichern
            # LLM-as-a-Judge Bewertung durchführen
            results = evaluate_question(
                question=st.session_state.generated_question,
                specs=st.session_state.specs
            )
            # Audit: Rohantworten des Judges sichern
            import json, os
            os.makedirs(RUNS_DIR, exist_ok=True)
            with open(f"{RUNS_DIR}/{st.session_state.last_run_id}_judge_raw.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            # Resultate (Scores) in Session-State speichern
            eval_dict = {
                "question": st.session_state.generated_question,
                "origin": st.session_state.generated_origin
            }
            for crit_id, eval_result in results[0]["evaluations"].items():
                eval_dict[f"Kriterium {crit_id}"] = eval_result["score"]
                eval_dict[f"Kriterium {crit_id} Begründung"] = eval_result["reason"]

            st.session_state.evaluated_questions.append(eval_dict)
            # In CSV-Datei schreiben (anhängen)
            export_results_to_csv(
                question=st.session_state.generated_question,
                results=results,
                origin=st.session_state.generated_origin
            )
        st.success("Bewertung abgeschlossen & gespeichert als evaluation_results.csv")

# Ergebnisse anzeigen (Tabelle mit Scores pro Kriterium)
if st.session_state.evaluated_questions:
    st.subheader("Evaluationsergebnisse")
    df = pd.DataFrame([{
    "Origin": eq.get("origin", ""),
    "Frage": eq.get("question", ""),
    "Kriterium 1 (curricular content)": eq.get("Kriterium 1", ""),
    "Begründung 1": eq.get("Kriterium 1 Begründung", ""),
    "Kriterium 2 (operator correctness)": eq.get("Kriterium 2", ""),
    "Begründung 2": eq.get("Kriterium 2 Begründung", ""),
    "Kriterium 3 (format compliance)": eq.get("Kriterium 3", ""),
    "Begründung 3": eq.get("Kriterium 3 Begründung", ""),
    "Kriterium 4 (task design)": eq.get("Kriterium 4", ""),
    "Begründung 4": eq.get("Kriterium 4 Begründung", ""),
    "Kriterium 5 (expectation horizon)": eq.get("Kriterium 5", ""),
    "Begründung 5": eq.get("Kriterium 5 Begründung", "")
} for eq in st.session_state.evaluated_questions])
    st.dataframe(df)
