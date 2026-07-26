# Arabic RAG Pipeline

A minimal Retrieval-Augmented Generation pipeline for Arabic text, running fully
locally — no external API calls.

## How it works
1. Documents are embedded with a multilingual sentence-transformer
   (`paraphrase-multilingual-MiniLM-L12-v2`).
2. The user's question is embedded and matched against the document embeddings
   using cosine similarity.
3. The top-matching passage is passed to a local LLM (`qwen2.5:0.5b` via Ollama)
   with a prompt that constrains the answer strictly to the retrieved context — if
   the answer isn't there, the model says so instead of inventing one.

## Stack
Python · sentence-transformers (Hugging Face) · NumPy · Ollama (qwen2.5:0.5b)

## Run
```bash
pip install sentence-transformers
ollama pull qwen2.5:0.5b
```

## Notes
Built during the Tasar'u GPU Engineering bootcamp. Individual work.
Grounding was the point: the prompt explicitly forbids answering from outside the
retrieved passage — the difference between retrieval-augmented and retrieval-flavoured.
