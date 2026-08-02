# RAG Chatbot with Multimodal Video Retrieval

A retrieval-augmented chatbot that answers how-to questions by grounding
responses in retrieved WikiHow articles, then surfaces a ranked playlist of
relevant instructional video clips, found via CLIP-based text-to-video
retrieval, for each step of the answer. Built with a local open-weights LLM
(no external API calls), FAISS for vector search, and a Gradio UI.


![Demo](Animation.gif)

## What it does

Ask a how-to question ("how do I make pasta?") and the app:

1. **Retrieves** relevant WikiHow articles with a hybrid semantic search
   (title embeddings + title-and-summary embeddings, weighted, with a SQL
   keyword fallback) and injects them as context.
2. **Generates** a concise, numbered-step answer with Qwen2.5-1.5B-Instruct,
   constrained by a system prompt to stay short and grounded in the
   retrieved context.
3. **Retrieves matching video clips** for each generated step using CLIP
   text-to-image similarity search over pre-computed sliding-window clip
   embeddings from a HowTo100M video subset, and lets you click a step to
   seek straight to that moment in the source video.

The UI also exposes an inspection panel — the live system prompt, the exact
context retrieved for the current turn, and a running token count — so the
whole prompt pipeline is transparent rather than a black box.

## Architecture

```
                ┌─────────────────┐
   user query → │  ArticleRetriever │ → WikiHow context (SQLite + 2×FAISS)
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │  Prompt builder  │ → system + context + rolled history
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │ Qwen2.5-1.5B-Instruct │ → numbered-step answer
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
   per step   → │  VideoRetriever  │ → ranked clip playlist (CLIP + FAISS)
                └─────────────────┘
                         ▼
                   Gradio UI (chat + inspection + video panel)
```

- **`retrieval.py`** — `ArticleRetriever` (WikiHow semantic + keyword search
  over SQLite/FAISS) and `VideoRetriever` (CLIP text-to-clip search over
  FAISS).
- **`WikiHow_Preprocessing.py`** — builds the WikiHow SQLite DB and the two
  FAISS text indices from a cleaned WikiHow CSV.
- **`HowTo100M_Preprocessing.py`** — slices HowTo100M videos into
  fixed-length clips, embeds sampled frames with CLIP, and writes the clip
  FAISS index + metadata CSV.
- **`chatbot_backend.py`** — the per-turn pipeline: retrieval → prompt
  construction → generation → validation → step extraction → video lookup.
  Exposes `get_response(...)` and `get_video_playlist(...)`, called once
  per chat turn.
- **`app.py`** — the Gradio UI: chat panel, inspection panel, and the video
  playlist/seek panel.

## Notable engineering details

- **Hybrid retrieval with graceful fallback** — semantic search across two
  weighted FAISS indices, falling back to a SQL `LIKE` keyword search if no
  result clears the similarity threshold, so retrieval degrades rather than
  fails on out-of-distribution queries.
- **Bounded prompt, unbounded UI history** — the chat display always shows
  the full conversation, but only the last `MAX_HISTORY_MESSAGES` turns are
  sent to the model, keeping latency and token usage predictable in long
  conversations.
- **Client-side video seeking without native support** — Gradio's `Video`
  component has no "start at N seconds" argument, so seeking is done via a
  small JS snippet that reads the target timestamp out of hidden DOM
  elements and polls until the video element's actual loaded source matches
  the expected clip, avoiding race conditions between Python state updates
  and the browser swapping the video source.
- **Index/metadata consistency** — the video FAISS index and its metadata
  CSV are generated together in one preprocessing run so FAISS positions and
  metadata rows can't drift out of sync; the two files are always
  regenerated as a pair, never edited independently.
- **Path resolution independent of working directory** — every script
  resolves its own paths relative to `Path(__file__).resolve().parent`, so
  the app can be launched from any working directory as long as `outputs/`
  and `data/` sit next to the scripts.

## Setup

```bash
pip install -r requirements.txt
python WikiHow_Preprocessing.py       # builds outputs/wikihow.* (needs data/wikihow-cleaned/wikihow-cleaned.csv)
python HowTo100M_Preprocessing.py     # builds outputs/howto100m_clip.* (needs data/howto100m_bundle/)
python app.py
```

Run everything from the project root — `outputs/` and `data/` must sit
alongside the scripts. `app.py` and `retrieval.py` both resolve paths
relative to their own file location, not your shell's current directory.

**Data note:** the WikiHow CSV and HowTo100M video bundle are not included
in this repo (see `.gitignore`) since they're large third-party datasets.
Point `data/wikihow-cleaned/wikihow-cleaned.csv` and
`data/howto100m_bundle/` at your own copies before running the
preprocessing scripts, or swap in a different text/video corpus with the
same folder layout.

## Troubleshooting

If videos won't load or the wrong clip plays, rerun
`python HowTo100M_Preprocessing.py`. The clip FAISS index and its metadata
CSV must come from the same run — if they get out of sync, clips will still
load but seek/label the wrong moment. Rerunning regenerates both together.
The WikiHow-side script only needs rerunning if `wikihow-cleaned.csv`
itself changed.

## Known limitations

- **Video retrieval quality is bounded by corpus coverage.** If a question
  falls outside the topics in `data/howto100m_bundle` (e.g. asking about
  cooking when the bundle is mostly DIY/home-improvement tasks), CLIP still
  returns its closest available match, which can look unrelated. Check
  `outputs/howto100m_clip_metadata.csv`'s `task_name` column to see what the
  corpus actually covers before assuming a mismatched result is a bug.
- **Video search runs per generated step, not the literal question** — if
  generated steps drift off-topic (often because WikiHow retrieval also had
  weak coverage for that question), video results drift with them. The
  inspection panel's context box shows what was actually retrieved, which
  is the first place to check when results look off.
- Small local LLM (1.5B params) with sampling-based generation, so answer
  quality and formatting adherence vary between runs — this is a retrieval
  and pipeline-engineering project, not a benchmark of generation quality.

## Tech stack

Python · Gradio · FAISS · Sentence-Transformers (MiniLM) · CLIP
(`openai/clip-vit-base-patch32`) · Qwen2.5-1.5B-Instruct · SQLite · OpenCV
