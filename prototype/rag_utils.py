import os
from typing import Dict, List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pdf_extract import extract_documents_from_dir
from indexing import build_faiss, HybridEnsemble


# 1) Datenaufnahme aus PDFs

def _load_all_pdfs() -> Dict[str, List[Document]]:
    """
    Erwartete Verzeichnisstruktur (rekursiv):
      data/pool/**.pdf
      data/evaluation/**.pdf
      data/spezifikationen/**.pdf
    """
    base = os.getenv("DATA_ROOT", "data")
    roots = {
        "pool": os.path.join(base, "pool"),
        "evaluation": os.path.join(base, "evaluation"),
        "specs": os.path.join(base, "spezifikationen"),
    }
    out: Dict[str, List[Document]] = {}
    for label, base in roots.items():
        if os.path.isdir(base):
            out[label] = extract_documents_from_dir(base, label)
        else:
            out[label] = []
    return out



# Chunking (domänenspezifisch)

def _chunk_docs(docs: List[Document], chunk_size: int, overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    out: List[Document] = []
    for d in docs:
        for chunk in splitter.split_text(d.page_content):
            out.append(Document(page_content=chunk, metadata=d.metadata.copy()))
    return out


def _prepare_corpora(raw: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
    specs = _chunk_docs(raw.get("specs", []), chunk_size=550, overlap=100)     # fein
    pool  = _chunk_docs(raw.get("pool", []),  chunk_size=1000, overlap=120)    # größer
    evals = _chunk_docs(raw.get("evaluation", []), chunk_size=1000, overlap=120)
    return {"specs": specs, "pool": pool, "evaluation": evals}


# Getrennte Vectorstores

def setup_vectorstores() -> Dict[str, HybridEnsemble]:
    raw = _load_all_pdfs()
    corpora = _prepare_corpora(raw)
    vs_specs = build_faiss(corpora["specs"])
    vs_pool  = build_faiss(corpora["pool"])
    vs_eval  = build_faiss(corpora["evaluation"])
    return {
        "specs": HybridEnsemble(vs_specs, corpora["specs"]),
        "pool":  HybridEnsemble(vs_pool,  corpora["pool"]),
        "eval":  HybridEnsemble(vs_eval,  corpora["evaluation"]),
    }



# Retrieval-Utilities

def _concat(docs: List[Document], limit_chars: int) -> str:
    out, total = [], 0
    for d in docs:
        t = d.page_content.strip()
        if not t:
            continue
        if total + len(t) > limit_chars:
            out.append(t[: max(0, limit_chars - total)])
            break
        out.append(t)
        total += len(t)
    return "\n\n".join(out)


def retrieve_specs(query: str, stores: Dict[str, HybridEnsemble], k: int = 6,
                   section_whitelist: Optional[set[str]] = None) -> List[Document]:
    docs = stores["specs"].search(query, k=k, mmr=True)
    if section_whitelist:
        allow = {s.lower() for s in section_whitelist}
        docs = [d for d in docs if (d.metadata.get("section") or "").lower() in allow]
    return docs


def retrieve_pool(query: str, stores: Dict[str, HybridEnsemble], k: int = 6) -> List[Document]:
    return stores["pool"].search(query, k=k, mmr=True)


def retrieve_eval(query: str, stores: Dict[str, HybridEnsemble], k: int = 6) -> List[Document]:
    return stores["eval"].search(query, k=k, mmr=True)


def _short(text: str, limit: int = 2200) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def build_specs_canon(stores: Dict[str, HybridEnsemble]) -> str:
    # „Always-on“-Kurzkanon (Operatoren + Format)
    op = retrieve_specs("Operatoren Liste Definitionen Deutsch Abitur", stores, k=3)
    fm = retrieve_specs("Formatvorgaben Struktur Aufgaben Aufgabentypen", stores, k=3)
    return _short(
        "### Operatoren (Kurzkanon)\n" + _concat(op, 1200) +
        "\n\n### Formatvorgaben (Kurzkanon)\n" + _concat(fm, 1200),
        2200
    )



# High-level Kontexte (Generierung)

def get_generation_contexts(stores: Dict[str, HybridEnsemble]) -> Dict[str, str]:
    canon = build_specs_canon(stores)
    spec_docs = retrieve_specs(
        "Deutsch Abitur Aufgabenbau Operatoren Kompetenzbereiche Erwartungshorizont",
        stores, k=6
    )
    pool_docs = retrieve_pool(
        "Beispiele Abituraufgaben Deutsch Interpretation Erörterung materialgestützt",
        stores, k=6
    )
    eval_docs = retrieve_eval(
        "Statistische Auswertung Auswahlhäufigkeit Themen Aufgabenwahl Schulen",
        stores, k=6
    )
    return {
        "specs": canon + "\n\n" + _concat(spec_docs, 4000),
        "pool": _concat(pool_docs, 2800),
        "evals": _concat(eval_docs, 1200),
    }



# Bewertung: rubrik-spezifische Specs

def load_specs_for_evaluation(stores: Dict[str, HybridEnsemble]) -> Dict[int, str]:
    """
    Ruft pro Kriterium gezielt passende Spezifikationssegmente ab, ausschließlich
    aus den PDF-basierten Vectorstores.
    Pro Rubrik wird strikt nur ein vordefiniertes PDF als Quelle akzeptiert.
    """
    if not stores or "specs" not in stores:
        raise RuntimeError("Specs-Store nicht initialisiert. Bitte zunächst die PDF-basierten Vectorstores erstellen.")

    queries = {
        1: "Prüfungsschwerpunkte Deutsch 2025 Leistungskurs Aufgabenarten Bewertung Dauer der Prüfung",
        2: "Operatoren Definitionen Beispiele Anforderungsbereiche Deutsch Abitur",
        3: "Beschreibung der Struktur Arbeitszeit Auswahlzeit Erwartungshorizont Bewertungshinweise",
        4: "Erläuterungen zur Konstruktion Aufgabenarten Prinzipien Varianten materialgestützt",
        5: "Kriterien für Aufgaben Erwartungshorizont Bewertungshinweise Domänenspezifik Materialgrundlage",
    }
    REQUIRED_DOCS: Dict[int, str] = {
        1: "ps_deutsch_2025_lk.pdf",
        2: "D_Grundstock_von_Operatoren.pdf",
        3: "D_Beschreibung_der_Struktur_der_Aufgaben.pdf",
        4: "D_Erlaeuterungen_zur_Konstruktion_der_Aufgaben.pdf",
        5: "D_Kriterien_fuer_Aufgaben_Erwartungshorizonte_und_Bewertungshinweise.pdf",
    }
    
    def _only_from_required_file(docs: List[Document], req_file: str) -> List[Document]:
        """Akzeptiert nur Segmente, deren 'file'-Basename exakt dem geforderten PDF entspricht."""
        rf = (req_file or "").strip().lower()
        out = []
        for d in docs:
            f = str((d.metadata or {}).get("file", "")).strip()
            base = os.path.basename(f).lower()
            if base == rf:
                out.append(d)
        return out
    out: Dict[int, str] = {}
    for cid, q in queries.items():
        segs = retrieve_specs(q, stores, k=48)  # etwas höher, damit das Ziel-PDF sicher in den Treffern ist
        req_file = REQUIRED_DOCS.get(cid, "")
        if not req_file:
            raise RuntimeError(f"REQUIRED_DOCS fehlt für Rubrik {cid}.")
        filtered = _only_from_required_file(segs, req_file)
        if not filtered:
            
            raise RuntimeError(
                f"Keine Segmente für Rubrik {cid} aus '{req_file}'. "
                "Prüfe REQUIRED_DOCS (Basename) & metadata['file'] beim Indexing."
            )
        out[cid] = _concat(filtered, 3500)

    return out