# Arabic RAG Pipeline

A Retrieval-Augmented Generation pipeline for Arabic, built end-to-end and running
entirely on a locally hosted model — no external API calls.

Given a question in Arabic, the system retrieves the most relevant document from a
knowledge base and generates an answer constrained strictly to that text.

## How it works

1. **Knowledge base** — 7 Arabic children's stories held as plain text.
2. **Embedding** — each story is encoded with `paraphrase-multilingual-MiniLM-L12-v2`
   into a 384-dimensional vector.
3. **Retrieval** — the question is embedded the same way and compared against every
   document with cosine similarity; the highest-scoring document is selected.
4. **Generation** — the retrieved text and the question are sent to `aya:8b` running
   locally through Ollama, with a system prompt that forbids using outside knowledge
   and instructs the model to answer "لا أعرف" when the answer isn't in the text.

**Example**

> **Question:** من القصة اللي فيها حيوان بطيء فاز بالسباق؟ وش العبرة منها؟
> **Retrieved:** The Tortoise and the Hare (cosine similarity 0.673)
> **Answer:** الحيوان البطيء الذي فاز بالسباق هو السلحفاة، والعبرة من القصة هي أن
> المثابرة تغلب الغرور والثقة الزائدة.

## Model selection: the generator was the bottleneck, not retrieval

Retrieval returned the correct passage on every run, at an identical 0.673 similarity
score. Only the generator changed:

| Model | Result |
|---|---|
| `qwen2.5:0.5b` | Incoherent; invented events that appear nowhere in the source text |
| `qwen2.5:3b` | Fluent, but **inverted the outcome** — named the hare as the winner |
| `aya:8b` | Correct, concise, faithful to the retrieved passage |

Two things this made concrete:

- In a RAG system, *"wrong answer"* and *"wrong retrieval"* are separate failures with
  separate fixes. Nothing about the retrieval layer needed changing here.
- For Arabic, model choice matters more than parameter count alone. A 6x jump from
  0.5B to 3B still produced a factually inverted answer; Aya, which is explicitly
  trained across 23 languages including Arabic, handled the same prompt correctly.

Moving the constraints into a `system` message (rather than embedding them in the user
prompt) and lowering `temperature` to 0.2 also measurably improved adherence to the
retrieved text.

## Running it

Designed for Google Colab with a **T4 GPU** runtime — the notebook installs and starts
the Ollama server inside the session:

```bash
pip install ollama sentence-transformers scikit-learn
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull aya:8b
```

Then run the notebook top to bottom. Note that a Colab runtime restart wipes the
server, the installed binary, and the pulled model — rerun from the first cell.

`aya:8b` is a 5.4 GB 4-bit quantized build and loads fully into the T4's VRAM at 100%
GPU utilization, with no CPU offload.

## Known limitations

- No chunking strategy: each story is short enough to be a single chunk. Longer
  documents would need semantic chunking, and the embedding model truncates
  silently past 128 tokens.
- Top-1 retrieval only, with no reranking layer.
- Retrieval quality is not yet measured — a labeled question/passage set and
  Recall@K would be the next step.
- 7 documents is a demonstration corpus, not a scale test. Linear search over NumPy
  arrays is exact and fast here; a vector database with approximate search would be
  needed at scale.

## Stack

Python · sentence-transformers (Hugging Face) · scikit-learn · NumPy · Ollama · Aya-8B
