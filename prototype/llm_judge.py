import pandas as pd
import json, re, time, os
from pathlib import Path
from typing import List, Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from rag_utils import load_specs_for_evaluation
from prompts import EVAL_PROMPTS

def get_llm_evaluator():
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    temp = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    return ChatOpenAI(temperature=temp, top_p=1.0, model=model)

def evaluate_question(
    question: str,
    specs: Optional[Dict[int, str]] = None,
    stores: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Entweder 'specs' (fertige Rubrik-Kontexte) übergeben ODER 'stores' angeben,
    damit die specs on-the-fly gebaut werden. Kein stiller Fallback mehr.
    """
    llm_evaluator = get_llm_evaluator()
    if specs is not None:
        documents = specs
    elif stores is not None:
        documents = load_specs_for_evaluation(stores)
    else:
        raise RuntimeError("evaluate_question: 'specs' fehlt. Übergib entweder 'specs' oder 'stores'.")
    result = {"question": question, "evaluations": {}}

    for crit_id, doc in documents.items():
        prompt = EVAL_PROMPTS[crit_id] \
            .replace("{{context}}", doc) \
            .replace("{{question}}", question)

        response = llm_evaluator.invoke(prompt)
        text = response.content if isinstance(response, AIMessage) else str(response)
        # Erwartet JSON {"score": int, "rationale": str}; robuster Fallback bei Freitext
        score, reason = 0, ""
        try:
            payload = json.loads(text)
            score = int(payload.get("score", 0))
            reason = str(payload.get("rationale", "")).strip()
        except Exception:
            m = re.search(r'"score"\s*:\s*(\d+)', text)
            if m:
                score = int(m.group(1))
            m2 = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, flags=re.S)
            if m2:
                reason = m2.group(1).strip()
            # letzter Fallback: Heuristik
            if reason == "":
                reason_line = next((line for line in text.splitlines() if "Begründung:" in line), "")
                reason = reason_line.replace("Begründung:", "").strip() if reason_line else ""
        score = max(0, min(5, score))

        # Score + Begründung speichern
        result["evaluations"][str(crit_id)] = {
            "score": score,
            "reason": reason,
            "text": text  
        }

    mdl = os.getenv("OPENAI_MODEL", "gpt-4")
    tmp = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    result["_meta"] = {"model": f"{mdl}_t{tmp}_p1.0", "ts": int(time.time())}
    return [result]


def export_results_to_csv(question, results, origin="AbiBuddy", path="evaluation_results.csv"):
    """
    Hängt die Bewertungsergebnisse einer Frage an eine CSV-Datei an.
    Falls die Datei noch nicht existiert, wird sie neu erstellt.
    Enthält Origin, Frage, Kriteriums-Scores, Kriteriums-Begründungen und Average-Score.
    """
    score_data = {"origin": origin, "question": question}
    total_score = 0
    
    for crit_id, evaluation in results[0]["evaluations"].items():
        score_data[f"Kriterium {crit_id}"] = evaluation["score"]
        score_data[f"Kriterium {crit_id} Begründung"] = evaluation["reason"]
        
        total_score += evaluation["score"]
        score_data[f"Kriterium {crit_id} Text"] = evaluation["text"].replace("\n", " ").strip()
    score_data["average_score"] = round(total_score / len(results[0]["evaluations"]), 2) if results[0]["evaluations"] else 0

    df_new = pd.DataFrame([score_data])
    file_path = Path(path)
    if file_path.exists():
        try:
            df_old = pd.read_csv(file_path)
        except Exception as e:
            df_old = pd.DataFrame() 
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(file_path, index=False)
