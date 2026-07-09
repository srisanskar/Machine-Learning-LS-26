# IITB Insti-Assist

A Retrieval-Augmented Generation (RAG) based AI assistant that answers questions about IIT Bombay hostel and campus life using official institute documents.

The main goal of this project was to build an assistant that doesn't rely on the LLM's memory. Instead, it first retrieves relevant information from IIT Bombay documents and then uses that information to generate answers. If the required information isn't present in the documents, the assistant simply responds with **"I don't know"** instead of making something up.

---

## Project Scope

I chose the **Hostel & Campus Life** scope because it felt like the most practical use case for RAG. Information like hostel rules, mess timings, guest policies, campus facilities, etc. is spread across multiple documents and websites, making it a good retrieval problem.

The assistant currently answers questions related to:

- Hostel rules and fines
- Guest policies
- Mess rules and timings
- Sample mess menu
- Campus Code of Conduct
- Hostel Housing & Room Retention
- Library information
- Student Wellness Centre
- Gender Cell
- General campus-life information

---

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- sentence-transformers (`all-MiniLM-L6-v2`)
- FAISS
- pdfplumber
- python-dotenv

---

## Project Structure

```
data/
├── raw/
├── processed/
└── index/

src/
├── ingest.py
├── build_index.py
├── rag.py
├── app.py
└── utils.py

requirements.txt
README.md
IITB_Insti_Assist_Writeup.docx
```

---

## How it Works

The overall pipeline is pretty straightforward.

1. Read the IIT Bombay PDFs and text files.
2. Extract the text.
3. Split the text into chunks.
4. Generate embeddings for every chunk.
5. Store them in a FAISS vector index.
6. When a user asks a question, retrieve the most relevant chunks.
7. Send those chunks along with the question to Gemini.
8. Display the generated answer along with the source documents used.

---

## Documents Used

The knowledge base consists of **7 documents**.

Official PDFs:

- Campus Code of Conduct
- Hostel Housing & Room Retention
- Hostel 10 Rules
- Hostel 10 Mess Rules & Timings
- Hostel 16 Sample Weekly Mess Menu
- IIT Bombay Infrastructure Details

Additional source:

- HSS Fresher's Handbook (only the general campus-life sections)

Most of these are official IIT Bombay documents. For the HSS handbook, I only kept the campus-life related sections and removed department-specific content since it wasn't relevant for this assistant.

---

## Chunking Strategy

I split the documents into **800-character chunks** with a **150-character overlap**.

I picked these values because most hostel rules are fairly short and self-contained. Keeping the chunks around this size usually keeps one complete rule together while the overlap prevents information from getting cut off at chunk boundaries.

For embeddings, I used **all-MiniLM-L6-v2** since it's lightweight, runs locally, and performs well enough for a project of this size.

---

## Installation

Clone the repository and install the dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
GEMINI_API_KEY=your_api_key_here
```

Run the ingestion and indexing scripts.

```bash
python src/ingest.py
python src/build_index.py
```

Finally, launch the Streamlit app.

```bash
streamlit run src/app.py
```

---

## Example Questions

Some questions you can try are:

- What are the guest rules for Hostel 10?
- What are the mess timings?
- Where is the Student Wellness Centre?
- What facilities are available on campus?
- What happens if hostel rules are violated?
- What does the Campus Code of Conduct say about ragging?

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic search using FAISS
- Source document display
- Multi-turn chat memory
- Honest responses when information isn't available
- Simple Streamlit interface

---

## Limitations

There are still a few limitations.

- Only Hostel 10 has detailed hostel rules and mess information.
- Hostel 16 only contributes a sample weekly menu.
- The knowledge base is static and needs to be rebuilt whenever new documents are added.
- The embedding model is relatively small, so retrieval isn't perfect for very ambiguous questions.
- The assistant doesn't have live access to IIT Bombay websites.

---

## Future Improvements

If I had more time, I'd like to:

- Include documents for all IIT Bombay hostels instead of just a few.
- Add exact sentence-level citation highlighting.
- Show a confidence score for retrieved answers.
- Allow users to upload their own PDFs.
- Improve retrieval using a larger embedding model or a reranking step.

---

## Acknowledgements

This project was developed as the final project for the **WnCC Learners' Space NLP Program**. It was a great opportunity to understand how Retrieval-Augmented Generation systems work beyond simply calling an LLM API.