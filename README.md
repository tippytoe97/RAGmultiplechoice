NLP Question Answering with RAG Pipeline

This project is a class assignment where I built a Retrieval-Augmented Generation (RAG) pipeline to answer multiple-choice questions based on NLP research papers. It combines document retrieval (using FAISS) with language models (via HuggingFace Transformers) to generate answers grounded in actual context.

What This Project Does
1. Loads research paper content, splits it, and stores it in a FAISS vector store
2. Uses a retriever + language model to build a working RAG chain
3. Takes a question, finds the most relevant context, and generates an answer
4. Supports multiple-choice question answering

Project Files
- retriever.py         # Loads and processes documents for FAISS
- rag_chain.py         # Builds the RAG chain and defines prompts
- llm_app.py           # Flask app to run RAG with Ollama
- rag_app.py           # Flask app for querying the RAG system
- hashed_answers.json  # Generated answers for grading
- requirements.txt     # Python dependencies

Datasets Used
- Clickbait Dataset – Headlines with binary label
- Web of Science – Scientific articles labeled by domain
- CoNLL-2003 – Used for POS tagging and Named Entity Recognition

Models + Libraries
- HuggingFace Transformers (BERT and DistilBERT)
- FAISS for similarity search
- PyTorch
- Flask for deployment
- Ollama for local LLM support

How to Run It
1. Install dependencies:
pip install -r requirements.txt
2. Load and embed documents:
python retriever.py
3. Create the RAG chain:
python rag_chain.py

Bonus: BERT vs DistilBERT
I also tested out DistilBERT and compared it with BERT to see how well knowledge distillation works. DistilBERT gave decent results with faster inference.

Example Use Case
Ask a question like:
“Which paper introduced BERT?”

The pipeline retrieves a chunk of relevant text and uses the model to generate an answer. It can also match answers to multiple-choice options.


👨‍💻 About Me
I'm a data analytics master's student at Georgia Tech, learning about NLP, machine learning, and real-world applications of AI. This project helped me get more comfortable with combining different tools in the ML pipeline and deploying something beyond Jupyter notebooks.

📧 tippileung1121@gmail.com