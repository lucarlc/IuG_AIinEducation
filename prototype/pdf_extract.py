# Direct PDF parsing (PyMuPDF) mit Abschnitts-/Seiten-Metadaten
from pathlib import Path
from typing import Iterable, List, Tuple
from langchain.schema import Document

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("Bitte 'pymupdf' installieren: pip install pymupdf") from e


def _iter_pdf_sections(pdf_path: Path) -> Iterable[Tuple[str, str, int, int]]:
    """
    Baseline-Heuristik:
    - liest alle Seiten der PDF
    - bildet pro Seite einen Abschnitt (Section)
    - liefert (section_title, text, page_start, page_end)
    Später kannst du hier Header-/Listen-Erkennung verfeinern.
    """
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if not text:
                continue
            title = f"Seite {i+1}"
            yield (title, text, i + 1, i + 1)
    finally:
        doc.close()


def extract_documents_from_dir(base_dir: str, source_label: str) -> List[Document]:
    """
    Liest *rekursiv* alle PDFs unter base_dir und erzeugt LangChain-Documents mit Metadaten.
    metadata:
      - source: 'pool' | 'evaluation' | 'specs'
      - file:   Pfad zur PDF
      - section: Titel (hier: 'Seite X' – kann später echter Header sein)
      - page_start / page_end: Seitenbereich des Abschnitts
    """
    docs: List[Document] = []
    for pdf_path in Path(base_dir).rglob("*.pdf"):
        for section_title, text, p0, p1 in _iter_pdf_sections(pdf_path):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_label,
                        "file": pdf_path.name,
                        "section": section_title,
                        "page_start": p0,
                        "page_end": p1,
                    },
                )
            )
    return docs
