# ECHO-Mental-Health-AI


# ◎ ECHO — Emotional Contextual Holistic Oracle

> *A conversational AI that doesn't just hear what you say — it understands what you mean.*

Built for **OpenAImer** (Track B: AI in Mental Health and Emotional Support), hosted at Jadavpur University.  
Sponsored by Mirror & TheAware.AI.

---

## What is ECHO?

Most mental health chatbots respond to the **message**. ECHO responds to the **person behind the message**.

When someone says *"I skipped class again. It's whatever."* — a standard chatbot hears a casual update. ECHO detects the minimizing language, the anhedonia signal, the implicit avoidance, and asks the one question that actually opens the right door.

ECHO does this through a **5-layer cognitive pipeline** that mirrors how a trained therapist actually thinks — not pattern matching, but building a living psychological model of the user across the entire conversation.

---

## Architecture

```
User Message
     ↓
┌─────────────────────────┐
│   1. SIGNAL DECODER     │  ← 10-dimensional emotional extraction via Gemini
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   2. NARRATIVE ENGINE   │  ← Builds living psychological profile per session
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   3. RETRIEVAL CORE     │  ← Hybrid semantic RAG over mental health corpus
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   4. RESPONSE SYNTH     │  ← Hyper-personalized prompt construction
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   5. SAFETY GATE        │  ← Crisis detection + India helpline escalation
└────────────┬────────────┘
             ↓
        Final Response
```

---

## The 5 Layers Explained

### Layer 1 — Signal Decoder (`signal_decoder.py`)
Every message is passed through Gemini with a structured extraction prompt that produces a 10-field emotional signal object:

| Field | What it captures |
|---|---|
| `primary_emotion` | Surface emotion shown |
| `underlying_need` | What they actually need (validation, relief, clarity…) |
| `hidden_distress` | What they are NOT saying |
| `linguistic_markers` | Minimizing words, self-blame, helplessness patterns |
| `distress_severity` | 0–10 integer score |
| `distress_type` | implicit / explicit / masked / neutral |
| `key_entities` | People, places, events mentioned |
| `narrative_theme` | Overarching life story theme (failure, isolation…) |
| `response_tone` | warm-curious / gentle-challenging / grounding… |
| `is_crisis` | Boolean — triggers safety escalation |

This runs at `temperature=0.1` for consistent, structured extraction.

---

### Layer 2 — Narrative Engine (`narrative_engine.py`)
**The core differentiator.** While other systems track conversation history, ECHO builds a **living psychological profile** that evolves with every turn:

```python
@dataclass
class UserNarrative:
    known_entities: dict        # "mom": "source of pressure"
    recurring_themes: list      # ["failure", "isolation"]
    cognitive_patterns: list    # ["self-blame", "minimizing"]
    emotional_vocabulary: list  # their own words: "hollow", "whatever"
    emotional_trajectory: list  # [{turn, severity, emotion}]
    preferred_engagement_style: str
    has_opened_up: bool
```

This profile is fed into every subsequent response — so by turn 4, ECHO knows this user has mentioned their parents twice, uses minimizing language under stress, and opened up more when gently challenged.

---

### Layer 3 — Retrieval Core (`retrieval_core.py`)
Hybrid RAG pipeline using ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`):

- **Knowledge base**: 20+ curated mental health documents covering validation techniques, CBT frameworks, implicit distress patterns, safety signals, and student-specific contexts
- **Enriched query**: Combines user message + decoded emotion + underlying need for retrieval
- **Grounding rule**: Retrieved knowledge is used *implicitly* — never quoted directly in responses

```python
enriched_query = f"{message} {primary_emotion} {underlying_need} {hidden_distress}"
```

---

### Layer 4 — Response Synthesizer (`response_synthesizer.py`)
Assembles the final prompt from all upstream context:

```
=== USER PSYCHOLOGICAL PROFILE ===
Known entities, recurring themes, cognitive patterns,
emotional vocabulary, trajectory arc, engagement style

=== CURRENT MESSAGE ANALYSIS ===
Decoded signals from Layer 1

=== THERAPEUTIC GROUNDING ===
Retrieved knowledge (implicit use only)

=== SEMANTIC MEMORY ===
Similar past moments from this conversation
```

Includes an **anti-generic check** — if the response contains phrases like *"I understand how you feel"* or *"That sounds really hard"*, it is automatically rejected and regenerated once.

---

### Layer 5 — Safety Gate (`safety_gate.py`)
Two-pass crisis detection:
1. **Pre-check**: Fast keyword scan before signal decoding
2. **Post-check**: Uses full decoded signals (`is_crisis`, severity ≥ 8)

When triggered, appends India-specific crisis resources:
- **iCall**: 9152987821 (Mon–Sat, 8am–10pm)
- **Vandrevala Foundation**: 1860-2662-345 (24/7, free, confidential)

---

## Frontend Features

### Real-time Narrative Sidebar
The sidebar updates live after every message, showing judges exactly what the system is tracking:
- 🟣 Recurring themes
- 🟢 People & events (entity tracking)
- 🟡 Cognitive patterns (self-blame, catastrophizing…)
- 🔴 Emotional vocabulary (the user's own words)

### Distress Timeline Chart
An animated SVG line chart plots `distress_severity` across every turn. Color shifts dynamically:
- Green (0–3) → Yellow (4–6) → Red (7–10)

This gives judges a visual representation of the emotional arc — one of the most powerful demo moments.

### Voice Input
Click 🎙 **Voice** → speak → message auto-sends. Uses the browser's Web Speech API with `en-IN` locale. No extra dependencies, no API calls.

### Typing Hesitation Analysis
Passively tracks keystrokes while the user types. After each send, the sidebar displays:
- Words per minute
- Pause count (gaps >2 seconds)
- Correction count (backspace/delete frequency)
- **Hesitation flag** — detected when ≥2 long pauses or max gap >4s

This detects distress *before* the message is even sent — directly targeting the Emotional Inference Accuracy rubric.

### Session Export (PDF)
Click ⬇ **Export** → generates a formatted report containing:
- Full conversation with per-message signal annotations
- Psychological profile (themes, patterns, entities, vocabulary)
- Distress timeline table
- Session metadata

Opens as a printable page → save as PDF via browser.

---

The detailed report on the mental health project can be accessed via the link below:
```
https://drive.google.com/file/d/17SnQSK4Xvt6etNScz73FCPJvFkksUqdg/view?usp=drivesdk
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 2.0 Flash (approved roster) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector DB | ChromaDB (in-memory + persistent) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (single file, zero build step) |
| Voice | Web Speech API (browser-native) |

---

## Project Structure

```
echo/
├── backend/
│   ├── main.py                  # FastAPI server — orchestrates full pipeline
│   ├── signal_decoder.py        # 10-dimensional emotional signal extraction
│   ├── narrative_engine.py      # Living psychological profile per session
│   ├── retrieval_core.py        # ChromaDB hybrid RAG + knowledge base
│   ├── response_synthesizer.py  # Final prompt builder + anti-generic check
│   ├── safety_gate.py           # Crisis detection + escalation
│   └── memory_store.py          # Session store + long-term semantic memory
├── frontend/
│   └── index.html               # Full UI — single file, no build needed
├── requirements.txt
└── .env                         # GEMINI_API_KEY goes here
```

---

## Setup & Running

### Prerequisites
- Python 3.10+
- A Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Clone & enter the project
```bash
git clone https://github.com/mandrita16/ECHO-Mental-Health-AI
cd ECHO-Mental-Health-AI
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the sentence-transformer model (~80MB). This only happens once.

### Step 4 — Add your API key
Create a `.env` file in the root `echo/` folder:
```
OPENROUTER_API_KEY=openrouter_key
CHROMA_TELEMETRY=false
```

### Step 5 — Start the server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
🚀 Initializing ECHO...
✅ Gemini model loaded.
✅ Embedding model loaded.
✅ Knowledge base ready.
🎯 ECHO is ready!
```

### Step 6 — Open the UI
Open `frontend/index.html` directly in your browser. No additional server needed.

---

## API Reference

### `POST /chat`
Main conversation endpoint.

**Request:**
```json
{
  "message": "I've been feeling really off lately",
  "session_id": "optional-existing-session-id"
}
```

**Response:**
```json
{
  "response": "That 'off' feeling...",
  "session_id": "uuid",
  "signals": {
    "primary_emotion": "dysthymia",
    "underlying_need": "validation",
    "distress_severity": 5,
    "is_crisis": false,
    ...
  },
  "turn_number": 1,
  "is_crisis": false
}
```

### `GET /session/{session_id}`
Returns the current narrative profile for a session — themes, entities, cognitive patterns, emotional vocabulary, trajectory.

### `POST /reset`
Clears a session and starts fresh.

### `GET /health`
Returns model initialization status.

---

## Judging Rubric Mapping

| Rubric Criterion | Weight | How ECHO addresses it |
|---|---|---|
| Emotional Inference Accuracy | 30% | Signal Decoder extracts 10 dimensions including hidden distress, linguistic markers, and masked language. Typing hesitation analysis detects distress before the message sends. |
| Memory & Personalization Continuity | 25% | Narrative Engine builds a living profile with entity tracking, emotional vocabulary mirroring, and trajectory-aware responses across all turns. |
| Grounding, Safety & Hallucination Control | 20% | ChromaDB RAG grounds responses in validated therapeutic knowledge. Safety Gate handles crisis escalation with India-specific resources. Anti-generic check prevents hallucinated platitudes. |
| Actionability & Response Craftsmanship | 15% | System prompt enforces zero generic fallbacks, response-to-person (not message), and one precise question per response. |
| UI Quality & Live Demo Smoothness | 10% | Real-time narrative sidebar, live distress chart, voice input, and PDF export — all visible and functional during judge interaction. |

---

## Response Quality Standard

ECHO responses are held to the competition's example quality bar:

> *User: "I skipped a few lectures this week. It's whatever."*  
> *ECHO: "'It's whatever' is doing a lot of work there — skipping things you used to show up for. What thought goes through your head right before you decide not to go?"*

The system prompt explicitly bans:
- "I understand how you feel"
- "That sounds really hard"  
- "You're not alone"
- "Have you tried..."
- "Things will get better"

If any of these appear, the response is automatically regenerated.

---

## Approved Model

This project uses **Gemini 2.0 Flash** — from the competition's approved LLM roster (Gemini 2.5 Flash / Pro).

---

## Competition

**OpenAImer** — Track B: AI in Mental Health and Emotional Support  
Organized by: SSID (Jadavpur University)  
Sponsored by: Mirror & TheAware.AI  
Round 2 (Onsite Demo): 11 April, 10:30 AM – 2:00 PM, Jadavpur University

---

## License

MIT License — free to use, modify, and build upon.
