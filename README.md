# DocuMind: Factual AI Document Chatbot with RAG

DocuMind is a premium, fully-functional Retrieval-Augmented Generation (RAG) document chatbot. It is implemented in **Python** using **Streamlit, LangChain, FAISS, and Gemini/OpenAI API**.

It allows you to upload documents (PDF, TXT, MD), split them into chunks, create high-dimensional embeddings, and query them locally using FAISS semantic search. Answers are factually grounded and include dynamic source reference boxes showing similarity metrics.

---

## Features

- **Document Processing**: Automatic parsing for `.pdf`, `.txt`, and `.md` files.
- **Smart Chunking**: Chunks text with overlap using LangChain's `RecursiveCharacterTextSplitter`.
- **Local FAISS Index**: Saves and queries vector embeddings locally using the FAISS library.
- **Multi-Model Provider Support**: Easily switch between **Google Gemini** (`gemini-2.5-flash`) and **OpenAI** (`gpt-4o-mini`) via sidebar settings.
- **Strict Q&A Grounding**: Leverages system guidelines ensuring the bot never speculates or answers outside the document context.
- **Interactive Citations**: Renders dynamic dropdown elements displaying the exact document sources and similarity percentages used to answer queries.

---

## Project Structure

```text
ai-document-chatbot/
├── app.py              # Main Streamlit Python Application
├── requirements.txt    # Python Package Dependencies
├── .env.example        # Environment variables configuration template
└── README.md           # Documentation & Startup Guide
```

---

## Setup & Running the Application

### 1. Prerequisites
- Python 3.9 to 3.11 installed.
- Access API key for either Google Gemini ([Google AI Studio](https://aistudio.google.com/)) or OpenAI.

### 2. Create a Virtual Environment (Recommended)
Navigate to the project directory and create a virtual environment:
```bash
cd /Users/yashvardhansingh/.gemini/antigravity/scratch/ai-document-chatbot
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Run the installation command:
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Open `.env` in a text editor and add your API credentials:
```env
GEMINI_API_KEY=your_actual_key_here
```
*(Alternatively, you can skip this step and enter your API keys directly into the sidebar text inputs when running the app).*

### 5. Launch the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

The web UI will automatically open in your default browser at **[http://localhost:8501](http://localhost:8501)**.
