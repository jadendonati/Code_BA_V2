# Notebook-Dokumentation: FF2 – JSON vs. Natürlichsprachlicher Text

**Datei:** `notebooks/BA_audio_analysis_FF2.ipynb`  
**Erstellt auf Basis von:** `BA_audio_analysis_v2.ipynb`  
**Zweck:** Experiment zu Forschungsfrage 2 (FF2)  
**Datum:** 2026-05-13

---

## Forschungsfrage

> **FF2:** Welches Übergabeformat – strukturiertes JSON oder natürlichsprachliche Beschreibung – führt zu empathischeren LLM-Antworten?

---

## Übersicht der Änderungen gegenüber v2

Das FF2-Notebook basiert strukturell auf `BA_audio_analysis_v2.ipynb`, wurde aber grundlegend umgebaut: Anstatt echtes Audio zu laden und auszuwerten, definiert es drei Szenarien mit festen Prosodiewerten und testet systematisch drei Eingabeformate (Varianten A, B, C) gegeneinander.

---

## Zelle für Zelle: Was wurde geändert und warum?

---

### Zelle 1 – Titel und Übersicht (Markdown)

**Inhalt:** Beschreibt das Notebook als FF2-Experiment mit 3 Szenarien × 3 Varianten = 9 GPT-Aufrufen.

**Warum geändert:** v2 war eine allgemeine Pipeline. Das FF2-Notebook hat einen spezifischen Versuchsaufbau mit klarer unabhängiger Variable (das Format), der im Titel sichtbar sein muss.

---

### Zelle 2 – Forschungsbezug (Markdown)

**Inhalt:** Erklärt die drei Varianten A, B, C und warum der Vergleich fair ist.

**Warum geändert:** In v2 wurden alle drei JSON-Versionen (v1/v2/v3) verglichen, also unterschiedlicher *Inhalt*. In FF2 ist der Inhalt von B und C identisch – nur das Format unterscheidet sich. Diese Unterscheidung musste explizit dokumentiert werden.

---

### Zelle 3 – Setup / Imports (Code)

| | v2 | FF2 |
|---|---|---|
| **Neu** | – | `import pandas as pd` |
| **Entfernt** | `import numpy as np`, `import re`, `import librosa` | Kein Audio wird geladen |
| **Pfad** | `audio_path` + `output_dir` | Nur `OUTPUT_DIR` für Ausgabedateien |

**Warum:** Das FF2-Experiment arbeitet mit vordefinierten Werten statt echter Audioanalyse. Librosa und NumPy sind deshalb nicht mehr nötig. Pandas wird neu gebraucht für die Ergebnistabelle.

> **[SCREENSHOT EINFÜGEN: Ausgabe der Setup-Zelle – "Setup erfolgreich"]**

---

### Zelle 4 – Szenarien-Definition (Code)

**Neu in FF2 – in v2 nicht vorhanden.**

Das Dictionary `SCENARIOS` enthält drei Einträge (gestresst, traurig, neutral), jeder mit:

| Feld | Inhalt | Verwendung |
|---|---|---|
| `transcript` | Der gesprochene Satz | Alle Varianten |
| `prosody` | Prosodische Merkmale im v3-Format | Variante B (JSON) |
| `emotion` | Label + Score | Variante B (JSON) |
| `text_description` | Dieselbe Info als natürlicher Satz | Variante C (Text) |

**Warum so aufgebaut:** Indem alle Informationen einmal zentral definiert sind, ist sichergestellt, dass B und C in jedem Szenario exakt dieselben inhaltlichen Informationen enthalten. Nur das Format unterscheidet sich – das ist die einzige Variable des Experiments.

**Woher stammen die Prosodiewerte:** Die Werte für "gestresst" basieren auf dem realen Messwert aus v2 (gestresst.wav). Die Werte für "traurig" und "neutral" sind typische Richtwerte aus der Literatur zu emotionaler Prosodie (langsam + leise = Trauer; normal + neutral = keine Emotion).

> **[SCREENSHOT EINFÜGEN: Die SCENARIOS-Definition im Notebook – gut lesbar als Code-Block]**

---

### Zelle 5 – Prompt-Konstruktion (Code)

**Neu in FF2 – in v2 nicht als separate Funktion vorhanden.**

Drei Hilfsfunktionen bauen den User-Prompt je nach Variante:

```
build_variant_a(scenario) → nur transcript
build_variant_b(scenario) → JSON mit transcript + prosody + emotion
build_variant_c(scenario) → "transcript" (text_description)
```

Der **System-Prompt ist für alle 9 Aufrufe identisch** – das ist entscheidend für die Vergleichbarkeit.

**Warum Funktionen statt inline:** Durch die Auslagerung in Funktionen ist der Experiment-Loop in Zelle 6 kurz und übersichtlich. Jede Funktion kann unabhängig getestet werden (Vorschau am Beispiel "gestresst").

> **[SCREENSHOT EINFÜGEN: Ausgabe der Vorschau-Prints – Variante A, B und C für "gestresst" nebeneinander]**

---

### Zelle 6 – GPT-Experiment: 9 Aufrufe (Code)

**Gegenüber v2 verändert:**

| | v2 | FF2 |
|---|---|---|
| **Anzahl Aufrufe** | 3 (v1/v2/v3 für eine Audiodatei) | 9 (3 Szenarien × 3 Varianten) |
| **Loop** | Über JSON-Versionen | Über Szenarien × Varianten |
| **Ergebnisse** | Dict mit Antworten | Liste von Dicts (wird zu DataFrame) |
| **Temperatur** | Konfigurierbar | Fix auf 0.3 |

**Warum Temperatur fix auf 0.3:** Das Experiment misst Formatunterschiede, nicht Kreativitätsunterschiede. Eine feste Temperatur macht die Ergebnisse reproduzierbar und vergleichbar.

> **[SCREENSHOT EINFÜGEN: Ausgabe der 9 GPT-Antworten in der Notebook-Ausgabe]**

---

### Zelle 7 – Automatische Bewertung durch zweites GPT (Code)

**Neu in FF2 – in v2 nicht vorhanden.**

Ein zweites GPT-4o-mini bewertet jede der 9 Antworten auf drei Kriterien:

| Kriterium | Was wird gemessen | Skala |
|---|---|---|
| `emotionale_anerkennung` | Wird die Emotion der Person explizit anerkannt? | 1–5 |
| `situative_angemessenheit` | Passt die Antwort zur beschriebenen Situation? | 1–5 |
| `sprachliche_waerme` | Wie warm und fürsorglich klingt die Formulierung? | 1–5 |

**Warum ein zweites GPT für die Bewertung:** Die manuelle Bewertung von 9 × 3 Kriterien wäre subjektiv und zeitaufwändig. Ein LLM-Evaluator liefert strukturierte, reproduzierbare Scores. Temperatur 0.0 sorgt für maximale Konsistenz der Bewertungen.

**Warum `json.loads()` statt Freitext:** Das Evaluator-GPT wird angewiesen, ausschließlich ein JSON zurückzugeben. Das ermöglicht direktes Einlesen der Scores ohne Parsing-Logik.

> **[SCREENSHOT EINFÜGEN: Ausgabe der Bewertungsschleife – z.B. "[gestresst / A] → {'emotionale_anerkennung': 5, ...}"]**

---

### Zelle 8 – Ergebnistabelle mit pandas (Code)

**Neu in FF2 – in v2 nicht vorhanden.**

Alle 9 Ergebnisse werden in einem `pd.DataFrame` zusammengefasst und mit deutschen Spaltenbezeichnungen ausgegeben.

**Warum pandas:** DataFrame ermöglicht `groupby`, `mean`, Sortierung – alles wichtig für die Auswertung in der BA. Die CSV-Ausgabe ist direkt in Excel auswertbar.

> **[SCREENSHOT EINFÜGEN: Die vollständige DataFrame-Tabelle als Notebook-Ausgabe]**

---

### Zelle 9 – Aggregierte Auswertung (Code)

**Neu in FF2 – in v2 nicht vorhanden.**

```python
df.groupby("Variante")[...].mean().sort_values("Gesamt", ascending=False)
```

Zeigt die Durchschnittsbewertung pro Variante (A, B, C) über alle drei Szenarien.

**Warum hier:** Diese Aggregation ist der Kern der Antwort auf FF2 – sie zeigt direkt, welches Format im Schnitt zu empathischeren Antworten führt.

> **[SCREENSHOT EINFÜGEN: Ausgabe der aggregierten Tabelle mit Mittelwerten pro Variante]**

---

### Zelle 10 – Speichern als JSON und CSV (Code)

| Datei | Format | Zweck |
|---|---|---|
| `audio/ff2_results.csv` | CSV (UTF-8 mit BOM) | Auswertung in Excel / R / Python |
| `audio/ff2_results.json` | JSON mit Metadaten | Vollständige Reproduzierbarkeit |

**Warum UTF-8 mit BOM (`utf-8-sig`):** Standard-UTF-8 kann in Excel zu Zeichenfehlern bei Umlauten führen. BOM-Variante öffnet korrekt.

**Warum JSON zusätzlich:** Die CSV enthält keine Prompts. Im JSON sind System-Prompt, Temperatur und alle User-Inputs als Metadaten enthalten – wichtig für die wissenschaftliche Nachvollziehbarkeit.

> **[SCREENSHOT EINFÜGEN: Ausgabe "Gespeichert: ff2_results.csv / ff2_results.json"]**

---

## Ergebnisse des Experiments (Lauf vom 2026-05-13)

### Rohdaten

| Szenario | Variante | Em. Anerkennung | Sit. Angemessenheit | Spr. Wärme | Gesamt |
|---|---|---|---|---|---|
| gestresst | A | 5 | 5 | 4 | 14 |
| gestresst | B | 5 | 5 | 4 | 14 |
| gestresst | C | 5 | 5 | 4 | 14 |
| traurig | A | 5 | 5 | 5 | 15 |
| traurig | B | 5 | 5 | 5 | 15 |
| traurig | C | 5 | 5 | 5 | 15 |
| neutral | A | 4 | 5 | 4 | 13 |
| neutral | B | 4 | 5 | 4 | 13 |
| neutral | C | 3 | 4 | 4 | 11 |

### Aggregiert: Durchschnitt pro Variante

| Variante | Em. Anerkennung | Sit. Angemessenheit | Spr. Wärme | Gesamt ∅ |
|---|---|---|---|---|
| A (Baseline) | 4.67 | 5.00 | 4.33 | **14.00** |
| B (JSON) | 4.67 | 5.00 | 4.33 | **14.00** |
| C (Text) | 4.33 | 4.67 | 4.33 | **13.33** |

> **[SCREENSHOT EINFÜGEN: Die geöffnete ff2_results.csv in der IDE oder Excel-Ansicht]**

---

## Interpretation der Ergebnisse

**Auffällig:** A und B erzielen identische Durchschnittswerte (14.0). C schneidet leicht schlechter ab (13.33), vor allem im neutralen Szenario (Gesamt: 11 statt 13/14).

**Mögliche Erklärungen:**
- Das Modell (gpt-4o-mini) kann JSON direkt verarbeiten – das Format ist kein Hindernis
- Natürlichsprachliche Klammerbeschreibungen (C) können irritieren, wenn der Ton des Satzes neutral ist ("Das klingt nach einem produktiven Tag" – aber in Klammern steht "neutral und sachlich")
- Bei emotionalen Szenarien (gestresst, traurig) macht das Format keinen Unterschied – das Modell erkennt die Emotion aus dem Satz allein

**Limitation:** Die Bewertung durch ein zweites GPT ist selbst ein LLM – es fehlt ein menschlicher Goldstandard. Für die BA sollte zumindest eine Stichprobe manuell bewertet werden.

---

## Dateipfade (Zusammenfassung)

| Datei | Pfad |
|---|---|
| Notebook | `notebooks/BA_audio_analysis_FF2.ipynb` |
| CSV-Ergebnisse | `audio/ff2_results.csv` |
| JSON-Ergebnisse | `audio/ff2_results.json` |
| Diese Dokumentation | `docs/FF2_Notebook_Dokumentation.md` |
