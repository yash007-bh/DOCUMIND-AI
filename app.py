import streamlit as st
import os
import io
from dotenv import load_dotenv
from pypdf import PdfReader
# LangChain core & splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# LangChain Integrations
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Load system environment variables
load_dotenv()
# Sync GEMINI_API_KEY to GOOGLE_API_KEY for LangChain Google GenAI compatibility
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# Set Page Configuration
st.set_page_config(
    page_title="DocuMind - AI RAG Document Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Premium UI CSS Injection
st.markdown("""
<style>
    /* Dark Theme Base Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f1524;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 1rem;
    }
    
    /* Titles and Header styles */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #f3f4f6 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Highlight references */
    .citation-ref {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #818cf8 !important;
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.85em;
        font-weight: 600;
    }
    
    /* Citation Cards styling */
    .citation-card {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #9ca3af;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .citation-source {
        color: #818cf8;
    }
    .citation-text {
        font-size: 13px;
        color: #d1d5db;
        font-style: italic;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)
# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
# --- Document Helper Functions ---
def parse_pdf(file_bytes) -> str:
    """Parses a PDF file from bytes in-memory and returns text."""
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
def process_uploaded_files(uploaded_files):
    """Processes, chunks and parses uploaded files into LangChain Document format."""
    documents = []
    
    for file in uploaded_files:
        file_bytes = file.read()
        file_name = file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        try:
            content = ""
            if file_extension == ".pdf":
                content = parse_pdf(file_bytes)
            elif file_extension in [".txt", ".md"]:
                content = file_bytes.decode("utf-8", errors="ignore")
                
            if content.strip():
                doc = Document(page_content=content, metadata={"source": file_name})
                documents.append(doc)
            else:
                st.sidebar.warning(f"File {file_name} contains no text.")
        except Exception as e:
            st.sidebar.error(f"Error parsing {file_name}: {str(e)}")
            
    return documents
def get_embeddings_model(provider, api_key):
    """Initializes and returns the selected LangChain Embeddings model."""
    if provider == "Google Gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
    elif provider == "OpenAI":
        return OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=api_key
        )
    return None
def get_llm(provider, api_key):
    """Initializes and returns the selected LangChain Chat Model."""
    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.2
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=api_key,
            temperature=0.2
        )
    return None
# --- UI Sidebar Layout ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
    st.title("DocuMind Control")
    st.markdown("Configure your model credentials and index documents below.")
    
    st.divider()
    
    # 1. Select Model Provider
    provider = st.selectbox(
        "API Provider",
        options=["Google Gemini", "OpenAI"],
        index=0
    )
    
    # Get API key from system environment
    default_key = ""
    if provider == "Google Gemini":
        default_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    elif provider == "OpenAI":
        default_key = os.getenv("OPENAI_API_KEY") or ""
        
    # 2. Enter API Key
    api_key = st.text_input(
        f"{provider} API Key",
        value=default_key,
        type="password",
        help="Obtain from Google AI Studio or OpenAI Platform dashboard."
    )
    
    # API key warning
    if not api_key:
        st.warning(f"Please provide your {provider} API Key to query or index.")
        
    st.divider()
    
    # 3. File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDFs, TXTs or Markdown documents."
    )
    
    # 4. Process indexing trigger button
    if st.button("Index Documents") and uploaded_files:
        if not api_key:
            st.error("API Key is required to process and embed documents!")
        else:
            with st.spinner("Processing documents and embedding text..."):
                # Parse files to LangChain Docs
                docs = process_uploaded_files(uploaded_files)
                
                if docs:
                    # Chunk files
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(docs)
                    
                    try:
                        # Initialize Embeddings
                        embeddings = get_embeddings_model(provider, api_key)
                        
                        # Generate FAISS database
                        vector_db = FAISS.from_documents(chunks, embeddings)
                        
                        # Save in session state
                        st.session_state.vector_db = vector_db
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        
                        st.success(f"Indexed {len(uploaded_files)} files into {len(chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Embedding failed: {str(e)}")
                else:
                    st.error("No valid text extracted from uploaded files.")
                    
    st.divider()
    
    # 5. Indexed Documents List
    st.subheader("Indexed Documents")
    if st.session_state.indexed_files:
        for file in st.session_state.indexed_files:
            st.caption(f"✓ {file}")
            
        if st.button("Clear Index"):
            st.session_state.vector_db = None
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents indexed yet.")
# --- Main Page Chat Interface ---
st.title("🧠 DocuMind: Factual RAG Chatbot")
st.caption("Answers are strictly grounded in your indexed local vector database.")
# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        
        # Render Citations for Assistant messages if they exist
        if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
            with st.expander(f"Sources used ({len(msg['citations'])})"):
                for idx, cite in enumerate(msg["citations"]):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span class="citation-source">[{idx+1}] {cite['source']}</span>
                            <span>Similarity Match: {cite['score']}%</span>
                        </div>
                        <div class="citation-text">"{cite['excerpt']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
# User Chat Input
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    # Render User Message
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate response
    with st.chat_message("assistant"):
        if not api_key:
            reply = f"⚠️ Please enter your {provider} API Key in the sidebar configuration to execute queries."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        elif st.session_state.vector_db is None:
            reply = "I cannot find any documents in my knowledge base. Please upload and index documents in the sidebar first!"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        else:
            with st.spinner("Searching document context and generating response..."):
                try:
                    # 1. Similarity Retrieval
                    db = st.session_state.vector_db
                    # Perform similarity search retrieving top 4 matches with relevance scores
                    # (FAISS returns distance. similarity_search_with_relevance_scores converts this to 0-1)
                    search_results = db.similarity_search_with_relevance_scores(user_query, k=4)
                    
                    citations = []
                    context_chunks = []
                    
                    for idx, (doc, score) in enumerate(search_results):
                        # Clean chunk text
                        clean_text = doc.page_content.replace('\n', ' ').strip()
                        source_name = doc.metadata.get("source", "Unknown Document")
                        
                        # Add to prompt context
                        context_chunks.append(f"[Source {idx+1}] Document: {source_name}\nContent:\n{doc.page_content}\n")
                        
                        # Save citation formatting
                        citations.append({
                            "source": source_name,
                            "excerpt": doc.page_content,
                            "score": int(score * 100) if score is not None else 0
                        })
                    
                    context_text = "\n---\n".join(context_chunks)
                    
                    # 2. Setup prompts and call LLM
                    system_instruction = (
                        "You are a helpful AI Document Assistant.\n"
                        "Your objective is to answer the user's question accurately using ONLY the provided document context.\n"
                        "Ensure you follow these rules strictly:\n"
                        "1. Base your answer solely on the provided CONTEXT. Do not use outside facts.\n"
                        "2. If the answer cannot be found or reasonably inferred from the CONTEXT, state clearly: "
                        "\"I cannot find the answer to this question in the uploaded documents.\"\n"
                        "3. Avoid speculation or inventing any details.\n"
                        "4. When stating a fact or answer from the documents, cite the source using brackets matching "
                        "the sources in the context, e.g., \"[1]\" or \"according to [Source 2]\".\n"
                        "5. Keep your tone professional, concise, and helpful."
                    )
                    
                    user_prompt = f"""
Here is the available context:
---
CONTEXT:
{context_text}
---
USER QUESTION:
{user_query}
"""
                    
                    # Initialize LLM
                    llm = get_llm(provider, api_key)
                    
                    # In LangChain, we pass system instruction and user prompt as list of messages
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content=system_instruction),
                        HumanMessage(content=user_prompt)
                    ]
                    
                    # Call LLM
                    response = llm.invoke(messages)
                    reply = response.content
                    
                    # Style in-text citations dynamically
                    import re
                    # Replaces [1] with HTML styled span
                    reply_styled = re.sub(r'\[([0-9]+)\]', r'<span class="citation-ref">[\1]</span>', reply)
                    
                    # Render Response
                    st.markdown(reply_styled, unsafe_allow_html=True)
                    
                    # Render Citations Expander
                    if citations:
                        with st.expander(f"Sources used ({len(citations)})"):
                            for idx, cite in enumerate(citations):
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span class="citation-source">[{idx+1}] {cite['source']}</span>
                                        <span>Similarity Match: {cite['score']}%</span>
                                    </div>
                                    <div class="citation-text">"{cite['excerpt']}"</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Save reply with citations to message list
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply_styled,
                        "citations": citations
                    })
                    
                except Exception as e:
                    error_reply = f"⚠️ An error occurred during RAG pipeline execution: {str(e)}"
                    st.error(error_reply)
                    st.session_state.messages.append({"role": "assistant", "content": error_reply})