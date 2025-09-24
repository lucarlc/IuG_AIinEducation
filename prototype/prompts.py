from langchain.prompts import PromptTemplate

# Prompt für Fragegenerierung via RAG + CoT

QUESTION_GENERATION_PROMPT = PromptTemplate(
    input_variables=["specs", "pool", "evals"],
    template="""
Du bist Lehrer:in und für die Erstellung einer Aufgabe für das Deutschabitur in Deutschland verantwortlich.

Nutze die bereitgestellten Informationen aus:

1. **Offizielle Spezifikationen**:
{{specs}} 
2. **Aufgabenpools vergangener Jahre**:
{{pool}} 
3. **Statistische Evaluationen zur Auswahlhäufigkeit von Aufgaben**:
{{evals}}

Die **Offizielle Spezifikationen** enthalten Informationen z.B. zu Textsorten, Operatoren, Kompetenzbereichen und Prüfungsschwerpunkten, die **Aufgabenpools vergangener Jahre** enthalten alte aufgabenpools die als stilistisches und strukturelles Vorbild dienen können, und die **Statistische Evaluationen zur Auswahlhäufigkeit von Aufgaben** enthalten Informationen zur Orientierung an tatsächlich im Abitur genutzten Aufgabentypen und thematischen Schwerpunkten aus den pools.
Richte dich primär nach 1. specs, um die formalen Anforderungen und inhaltliche Passung sicherzustellen. Nutze 2. pool als stilistische Orientierung und berücksichtige 3. evals zur Plausibilisierung der Aufgabenwahl.

Ziel ist die **Generierung einer vollständigen, authentischen Abituraufgabe**, wie sie im zentralen Abitur (erhöhtes Anforderungsniveau) gestellt wird.

### Anforderungen:
- Verwende **eine geeignete Textgrundlage** (z.B. Gedicht, Kurzgeschichte, Romanauszug oder Sachtext), wie sie auch in realen Abiturprüfungen verwendet wird.
- Berücksichtige bei der Formulierung der Aufgaben die **Operatoren**, **Kompetenzbereiche**, **Anforderungsniveaus** und **Formatvorgaben**, wie in den Spezifikationen definiert.
- Achte darauf, dass die Aufgabenstellung **sachlich, präzise und standardisiert** formuliert ist - orientiere dich sprachlich an den Aufgabenpools.
- **Vermeide jede Form von Meta-Informationen** wie Titel, Kurzbeschreibung, Jahr, Anforderungsniveau oder formale Angaben wie „Hilfsmittel: Wörterbuch“.

### Format:
Die Aufgabe kann eine oder mehrere Teilaufgaben enthalten, typischerweise nummeriert (z.B. 1., 2., ggf. 1.1, 1.2), aber orientiere dich an den echten Aufgabenformaten.

Beginne **direkt mit der Aufgabenstellung**.

Du generierst eine Abituraufgabe, die jedoch in Inhalt und Struktur nicht von den offiziellen Aufgaben unterscheidbar ist."""
)

# Bewertungsrubriken & CoT-Prompts pro Rubrik
EVAL_PROMPTS = {
    1: """
Du bist Fachprüfer:in im Deutschabitur.
Das folgende Dokument enthält **inhaltliche Vorgaben und Themenfelder** aus den offiziellen Abiturrichtlinien (curriculare Passung).

{{context}}

Aufgabe:
{{question}}

Beurteile die curriculare Passung dieser Aufgabe anhand der offiziellen Prüfungsschwerpunkte und Kerncurricula für das Deutschabitur.

**Denke schrittweise:**
1. Ermittle, welches Thema, welche Epoche oder Gattung die Aufgabe behandelt.
2. Vergleiche dies mit den zentralen Themenfeldern laut Abiturrichtlinien.
3. Beurteile, ob sie curricular vorgeschrieben, optional oder irrelevant ist.


Bitte gib am Ende deiner Analyse eine Bewertung auf einer Skala von 1 bis 5 ab:

1 = nicht erfüllt  
2 = unzureichend erfüllt  
3 = teilweise erfüllt  
4 = weitgehend erfüllt  
5 = vollständig erfüllt  

Gib die Ausgabe **ausschließlich** als JSON-Objekt im Format:
{"score": <0-5 Ganzzahl>, "rationale": "<kurze Begründung>"}
""",
    2: """
Du bist Fachprüfer:in im Deutschabitur.
Das folgende Dokument enthält die **offiziellen Operatoren** mit Definitionen und Beispielen.

{{context}}

Aufgabe:
{{question}}

Analysiere die verwendeten Operatoren in dieser Abituraufgabe anhand der offiziellen Operatorenliste.

**Denke schrittweise:**
1. Identifiziere die verwendeten Operatoren (z.B. analysieren, erörtern).
2. Ordne sie dem Anforderungsbereich (I-III) korrekt zu.
3. Beurteile, ob sie funktional und leistungsgerecht eingesetzt sind.


Bitte gib am Ende deiner Analyse eine Bewertung auf einer Skala von 1 bis 5 ab:

1 = nicht erfüllt  
2 = unzureichend erfüllt  
3 = teilweise erfüllt  
4 = weitgehend erfüllt  
5 = vollständig erfüllt  

Gib die Ausgabe **ausschließlich** als JSON-Objekt im Format:
{"score": <0-5 Ganzzahl>, "rationale": "<kurze Begründung>"}
""",
    3: """
Du bist Fachprüfer:in im Deutschabitur.
Das folgende Dokument enthält **formale Strukturvorgaben** für Abituraufgaben (Format, Umfang, Hilfsmittelangaben etc.).

{{context}}

Aufgabe:
{{question}}

Überprüfe die formale Struktur der Aufgabe auf Übereinstimmung mit offiziellen Abiturvorgaben.

**Denke schrittweise:**
1. Prüfe die äußere Gliederung (Nummerierung, Teilaufgaben).
2. Kontrolliere die Angabe oder Verfügbarkeit von Quellen und Materialien.
3. Bewerte, ob Aufbau, Sprache und Umfang dem Prüfungsformat entsprechen.


Bitte gib am Ende deiner Analyse eine Bewertung auf einer Skala von 1 bis 5 ab:

1 = nicht erfüllt  
2 = unzureichend erfüllt  
3 = teilweise erfüllt  
4 = weitgehend erfüllt  
5 = vollständig erfüllt  

Format der Ausgabe am Schluss:
Gib die Ausgabe **ausschließlich** als JSON-Objekt im Format:
{"score": <0-5 Ganzzahl>, "rationale": "<kurze Begründung>"}
""",
    4: """
Du bist Fachprüfer:in im Deutschabitur.
Das folgende Dokument beschreibt **Qualitätsanforderungen an die Aufgabenformulierung**.

{{context}}

Aufgabe:
{{question}}

Bewerte die innere Stimmigkeit und Verständlichkeit der Aufgabe.

**Denke schrittweise:**
1. Analysiere, ob die Aufgabe in sich logisch aufgebaut ist.
2. Prüfe, ob alle Teilaufgaben zusammenpassen und verständlich formuliert sind.
3. Achte auf potenzielle Missverständnisse oder widersprüchliche Anforderungen.


Bitte gib am Ende deiner Analyse eine Bewertung auf einer Skala von 1 bis 5 ab:

1 = nicht erfüllt  
2 = unzureichend erfüllt  
3 = teilweise erfüllt  
4 = weitgehend erfüllt  
5 = vollständig erfüllt  

Gib die Ausgabe **ausschließlich** als JSON-Objekt im Format:
{"score": <0-5 Ganzzahl>, "rationale": "<kurze Begründung>"}
""",
    5: """
Du bist Fachprüfer:in im Deutschabitur.
Das folgende Dokument erklärt **Anforderungen an Erwartungshorizonte** (Bewertungskriterien und Leistungsbeschreibung).

{{context}}

Aufgabe:
{{question}}

Beurteile, ob sich für diese Aufgabe ein transparenter und differenzierter Erwartungshorizont erstellen ließe.

**Denke schrittweise:**
1. Überprüfe, ob alle drei Anforderungsbereiche (I-III) abgedeckt sind.
2. Bewerte, ob Teilaufgaben bewertbar sind und ein differenziertes Notenspektrum ermöglichen.
3. Leite ab, ob die Aufgabenformulierung klare Leistungsanforderungen zulässt.


Bitte gib am Ende deiner Analyse eine Bewertung auf einer Skala von 1 bis 5 ab:

1 = nicht erfüllt  
2 = unzureichend erfüllt  
3 = teilweise erfüllt  
4 = weitgehend erfüllt  
5 = vollständig erfüllt  

Gib die Ausgabe **ausschließlich** als JSON-Objekt im Format:
{"score": <0-5 Ganzzahl>, "rationale": "<kurze Begründung>"}
"""
}