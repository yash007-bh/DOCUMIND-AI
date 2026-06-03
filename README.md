# DocuMind: Factual AI Document Chatbot with RAG
DocuMind is a premium, fully-functional Retrieval-Augmented Generation (RAG) document chatbot. It allows you to upload documents (PDF, TXT, MD), index them locally using Google Gemini's embedding model, and chat with an AI assistant that answers questions *strictly* grounded in your uploaded documents.
DocuMind is a premium, fully-functional Retrieval-Augmented Generation (RAG) document chatbot. It is implemented in **Python** using **Streamlit, LangChain, FAISS, and Gemini/OpenAI API**.
It allows you to upload documents (PDF, TXT, MD), split them into chunks, create high-dimensional embeddings, and query them locally using FAISS semantic search. Answers are factually grounded and include dynamic source reference boxes showing similarity metrics.
---
## Features
- **Local Vector Database**: A custom, lightweight, in-memory vector store that saves to a local JSON file—no external database installation or setup required.
- **Multilingual / Multidocument Support**: Upload and chat across multiple PDF, TXT, and MD files simultaneously.
- **Factual Grounding**: Powered by `gemini-2.5-flash` with strict instructions to answer *only* from your uploaded files and avoid hallucinations.
- **Smart Source Citations**: Displays dynamic source cards indicating text excerpts and relevance match percentages. Click citations inside the bot replies to instantly highlight and scroll to the matching document segment.
- **Premium Glassmorphic UI**: Sleek dark mode design with drag-and-drop uploads, indexing progress meters, and dynamic message loading skeletons.
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
├── public/                 # Frontend Static Assets
│   ├── index.html          # Main HTML5 Layout
│   ├── style.css           # Custom Glassmorphic CSS Theme
│   └── app.js              # Client-Side Application Controller
├── db/                     # Vector Database Directory
│   └── vector-store.json   # (Auto-generated) Serialized Vector Store
├── server.js               # Express API Endpoint Server
├── rag-engine.js           # RAG Embedding, Chunker & Math Math Engine
├── package.json            # Node.js Project Manifest
├── .env.example            # Environment variables template
└── README.md               # Setup & Documentation
├── app.py              # Main Streamlit Python Application
├── requirements.txt    # Python Package Dependencies
├── .env.example        # Environment variables configuration template
└── README.md           # Documentation & Startup Guide
```
---
## Installation & Setup
## Setup & Running the Application
### Prerequisites
- Node.js (version 20 or higher recommended)
- A Google Gemini API Key. You can get one for free at [Google AI Studio](https://aistudio.google.com/).
### 1. Prerequisites
- Python 3.9 to 3.11 installed.
- Access API key for either Google Gemini ([Google AI Studio](https://aistudio.google.com/)) or OpenAI.
### 1. Install Dependencies
Navigate to the project folder and run the installation command:
### 2. Create a Virtual Environment (Recommended)
Navigate to the project directory and create a virtual environment:
```bash
npm install
cd /Users/yashvardhansingh/.gemini/antigravity/scratch/ai-document-chatbot
python3 -m venv venv
source venv/bin/activate
```
### 2. Configure Environment Variables
Copy the `.env.example` file to `.env`:
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
Open `.env` in your text editor and add your API key:
Open `.env` in a text editor and add your API credentials:
```env
GEMINI_API_KEY=AIzaSy...your_actual_api_key_here
GEMINI_API_KEY=your_actual_key_here
```
*(Alternatively, you can skip this step and enter your API keys directly into the sidebar text inputs when running the app).*
### 3. Start the Server
Run the startup script:
### 5. Launch the Application
Start the Streamlit application:
```bash
npm start
streamlit run app.py
```
For automatic reload during development, run:
```bash
npm run dev
```
The application will be accessible at: **[http://localhost:3000](http://localhost:3000)**
The web UI will automatically open in your default browser at **[http://localhost:8501](http://localhost:8501)**.
---
## How It Works Under The Hood
1. **Document Uploading**: When you upload a file, the backend parses it (using `pdf-parse` for PDFs or standard buffer conversion for plain text files).
2. **Text Chunking**: The text is split into segments of roughly ~1000 characters with a ~200 character overlap. The splitter attempts to end chunks cleanly at paragraph or sentence breaks.
3. **Embeddings Generation**: The backend uses the official `@google/genai` SDK to call the `text-embedding-004` model to generate high-dimensional vectors representing the semantic meaning of each chunk.
4. **Vector DB Storage**: The chunks and their embeddings are saved locally in `./db/vector-store.json`.
5. **Retrieval**: When you send a message, the server embeds your message and performs a cosine-similarity calculation against all document chunks in local memory. It returns the top 4 most relevant matches.
6. **Augmented Generation**: The server compiles a context payload containing the top matches and sends it along with the conversation history and your question to the `gemini-2.5-flash` model. It is instructed to reply *only* from the context and to provide numeric bracketed citations (e.g. `[1]`, `[2]`).
7. **Citations UI rendering**: The frontend captures the citations, makes the inline references clickable, and renders the corresponding matching cards at the bottom of the message bubble.