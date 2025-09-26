# AbiBuddy – Automatisierte Abituraufgabengenerierung mit GPT-4

## 🧠 Projektbeschreibung
**AbiBuddy** ist ein prototypisches System zur automatisierten Generierung und Bewertung von Abituraufgaben im Fach Deutsch mithilfe von GPT-4. Es nutzt **Retrieval-Augmented Generation (RAG)** zur kontextbasierten Aufgabenformulierung und ein **LLM-as-a-Judge**-Modul zur Bewertung auf Basis offizieller Bewertungskriterien.

## 📂 Projektstruktur
```
Neuer Ordner/
│
.
├── prototype/                # App & Pipeline (main_app.py, rag_utils.py, ...)
│   ├── indexing.py
│   ├── llm_judge.py
│   ├── main_app.py
│   ├── pdf_extract.py
│   ├── prompts.py
│   └── runs/                 # Laufprotokolle
├── data/                     # Eingabedaten
│   ├── pool/                 # offizielle Aufgabenpools (PDF)
│   ├── evaluation/           # Evaluationsmaterial (PDF)
│   └── spezifikationen/      # offizielle Spezifikationen (PDF)                                                        
├── README.md
├── requirements.txt          # oder pyproject.toml
├── .env.example              # Beispiel-Variablen (ohne echte Keys)
├── .gitignore
└── .pre-commit-config.yaml

```

## ⚙️ Installation & Ausführung

### 1. Virtuelle Umgebung aktivieren und Abhängigkeiten installieren
```bash
.\venv\Scripts\Activate
pip install -r requirements.txt
```
### 2. Konfiguration (.env)
Lege eine Datei .env im Repo-Root an (siehe .env.example) und setze:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4          # optional, Standard: gpt-4
OPENAI_TEMPERATURE=0.0      # optional, Standard: 0.0
DATA_ROOT=data              # optional, Standard: data
RUNS_DIR=prototype/runs     # optional, Standard: runs

### 3. Anwendung starten
```bash
streamlit run prototype/main_app.py
```

## 💡 Features
- **Zufällige Abituraufgabe generieren** mit GPT-4 + RAG
- **Externe Aufgaben bewerten lassen** (z. B. aus anderen Modellen)
- **Evaluieren & Exportieren** nach 5 offiziellen Kriterien inkl. CSV-Speicherung
- **Übersichtliche GUI** via Streamlit

## 🧪 Bewertungskriterien
Die Bewertung basiert auf einer domänenspezifischen Bewertungsmatrix mit diesen sechs Dimensionen:
1. Curriculare Passung
2. Operatorenkorrektheit
3. Formatkonformität
4. Aufgabenstruktur
5. Erwartungshorizont-Bezugsfähigkeit

Jede Bewertung erfolgt automatisiert über GPT-4 mit Retrieval-Unterstützung und Chain-of-Thought-Reasoning.

## 🔐 Sicherheit & Datenschutz
- Die Datei `.env` enthält deinen **persönlichen API-Key** und darf **nicht veröffentlicht** werden.
- Es werden **keine personenbezogenen Daten** verarbeitet.

## Reproduzierbarkeit
- Datenpfad via `DATA_ROOT` (Default: `data/`)
- Modell via `OPENAI_MODEL` (Default: `gpt-4`), Temperatur via `OPENAI_TEMPERATURE` (Default: `0.0`)
- Alle Generierungs- und Bewertungsruns werden mit Prompt-Hash unter `runs/` protokolliert.

## 📄 Lizenz
Dieses Projekt kann unter der MIT- oder CC-BY 4.0-Lizenz veröffentlicht werden (je nach Datenquelle und Code). Bitte im Zweifel mit den Betreuenden abstimmen.
