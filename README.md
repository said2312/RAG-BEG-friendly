\# RAG-BEG-friendly



A small local RAG system. Index PDFs, ask questions, get answers that cite the file and page. Runs on your computer with Ollama and ChromaDB, with no API key.



Status: in progress.



\## Run it



pip install -r requirements.txt

ollama pull llama3.2:3b

ollama pull nomic-embed-text

set RAG\_CHAT\_MODEL=llama3.2:3b

python -m src.cli ingest ./pdfs

python -m src.cli ask "Your question"



Web interface: pip install streamlit, then streamlit run app.py



\## Roadmap



\- \[ ] Evaluation set and hit-rate script

\- \[ ] Compare chunk sizes

\- \[ ] Hybrid search



.

