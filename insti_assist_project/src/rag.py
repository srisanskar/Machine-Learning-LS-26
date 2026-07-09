"""
rag.py
Core RAG pipeline functions: retrieve top-k chunks for a query, build a
grounded prompt, call the Gemini API, and wire it all together.

This module is imported by app.py (Streamlit UI) but can also be run
directly for a console chatbot loop (run_chatbot()).
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4  # number of chunks to retrieve per query

GEMINI_MODEL_NAME = "gemini-2.5-flash"  # current stable model (gemini-1.5 was retired)

SYSTEM_PROMPT = """You are IITB Insti-Assist, a helpful assistant that answers questions about \
IIT Bombay's Hostel and Campus Life, using ONLY the context provided below.

Rules you must follow strictly:
1. Answer using ONLY the information in the provided context. Do not use outside knowledge.
2. If the context contains NO information relevant to the topic of the question at all \
(e.g. the question is about academics, faculty, or something entirely unrelated to hostel/campus \
life), respond exactly with: "I don't know based on the available documents." Do not guess.
3. If the question asks about a SPECIFIC hostel (e.g. "Hostel 5") but the context only contains \
rules/timings/policies for a DIFFERENT specific hostel (e.g. Hostel 10) on that same topic, DO NOT \
refuse. Instead, answer using the data you do have, and clearly flag that it is from the other \
hostel and may differ. For example: "I don't have specific data for Hostel 5, but according to \
Hostel 10's rules, guest entry is allowed until 10 PM — other hostels may set different timings/fines."
4. Be concise and direct. Quote specific rules, numbers, or timings from the context when relevant.
5. When citing hostel-specific numbers (fines, timings) that come from only one hostel's document, \
mention which hostel they belong to.
"""


class RAGPipeline:
    def __init__(self):
        self._load_index()
        self._load_embedder()
        self._load_llm()

    def _load_index(self):
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        chunks_path = os.path.join(INDEX_DIR, "chunks.json")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                "FAISS index not found. Run `python src/ingest.py` then "
                "`python src/build_index.py` first."
            )
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

    def _load_embedder(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def _load_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Create a .env file with GEMINI_API_KEY=your_key "
                "(get a free key at https://aistudio.google.com/app/apikey)"
            )
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single query string into a normalized vector."""
        vec = self.embedder.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, k: int = TOP_K):
        """Return the top-k most relevant chunks (with metadata + similarity score) for a query."""
        query_vec = self.embed_text(query)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source_file": self.metadata[idx]["source_file"],
                "source_url": self.metadata[idx]["source_url"],
                "score": float(score),
            })
        return results

    def build_prompt(self, query: str, retrieved_chunks):
        """Format retrieved chunks + query into a single LLM-ready prompt."""
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            context_blocks.append(
                f"[Source {i+1}: {chunk['source_file']}]\n{chunk['text']}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """Call the Gemini API and extract the text response."""
        response = self.llm.generate_content(prompt)
        try:
            return response.text.strip()
        except Exception:
            return "I don't know based on the available documents."

    def rag_answer(self, query: str, k: int = TOP_K):
        """
        Full pipeline: retrieve -> build_prompt -> generate_answer.
        Returns a dict with the answer and the sources used, so the UI
        can display "grounded in" citations.
        """
        retrieved = self.retrieve(query, k=k)

        if not retrieved:
            return {
                "answer": "I don't know based on the available documents.",
                "sources": [],
            }

        prompt = self.build_prompt(query, retrieved)
        answer = self.generate_answer(prompt)

        return {
            "answer": answer,
            "sources": retrieved,
        }


def run_chatbot():
    """Simple interactive console loop for quick testing without the Streamlit UI."""
    print("Loading IITB Insti-Assist (Hostel & Campus Life)...")
    pipeline = RAGPipeline()
    print("Ready! Ask a question (type 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = pipeline.rag_answer(query)
        print(f"\nAssistant: {result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            seen = set()
            for s in result["sources"]:
                if s["source_file"] not in seen:
                    print(f"  - {s['source_file']} (similarity: {s['score']:.2f})")
                    seen.add(s["source_file"])
        print()


if __name__ == "__main__":
    run_chatbot()
