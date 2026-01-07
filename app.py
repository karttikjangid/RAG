import streamlit as st
from youtube_ingestion import get_youtube_transcript
from pdf_ingestion import get_pdf_text
from chunking import get_chunks
from embedding import vector_embedding
from retrieval import search_best_chunks
from generation import generate_answer
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Cache the embedding model (loads only once)
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cache PDF text extraction
@st.cache_data(show_spinner="Extracting PDF text...")
def cached_pdf_extraction(file_name, file_bytes):
    """Cache PDF text extraction by file content hash"""
    temp_path = f"/tmp/{file_name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    text = get_pdf_text(temp_path)
    os.remove(temp_path)
    return text

# Cache YouTube transcript fetching
@st.cache_data(show_spinner="Fetching YouTube transcript...", ttl=3600)
def cached_youtube_transcript(url):
    """Cache YouTube transcript for 1 hour"""
    return get_youtube_transcript(url)

# Cache chunking operation
@st.cache_data(show_spinner="Chunking text...")
def cached_chunking(text, chunk_size=500, overlap=100):
    """Cache text chunking"""
    return get_chunks(text, chunk_size, overlap)

# Cache vector embeddings
@st.cache_data(show_spinner="Creating embeddings...")
def cached_embedding(_model, chunks):
    """Cache vector embeddings creation"""
    return _model.encode(chunks)

# Page Configuration
st.set_page_config(
    page_title="LecturMate - Your AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Light Green, Calm Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Global Font & Colors */
    * {
        font-family: 'Manrope', sans-serif !important;
    }
    
    html, body {
        background: linear-gradient(135deg, #f0f9f4 0%, #e8f5f0 100%) !important;
    }
    
    /* Main App Background - Light Green Gradient */
    .main, .appview-container, .main .block-container {
        background: linear-gradient(135deg, #f0f9f4 0%, #e8f5f0 100%) !important;
    }
    
    .main {
        padding: 0 !important;
    }
    
    /* Ensure background on all containers */
    section[data-testid="stMain"] {
        background: linear-gradient(135deg, #f0f9f4 0%, #e8f5f0 100%) !important;
    }
    
    /* Sidebar - Light Green Tint */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5fdf9 0%, #edf7f2 100%) !important;
        border-right: 1px solid #d1e7dd;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
        background: transparent !important;
    }
    
    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Header Section */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #2e7d32 !important;
    }
    
    /* Regular text */
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #2d3436 !important;
    }
    
    /* Chat Container */
    .stChatFloatingInputContainer {
        bottom: 1rem !important;
        background: transparent !important;
        border: none !important;
        padding: 0 15% !important;
    }
    
    /* Chat Input Box */
    .stChatInput {
        background-color: white !important;
        border: 1px solid #c8e6c9;
        border-radius: 1rem;
        box-shadow: 0 2px 12px rgba(76, 175, 80, 0.1);
    }
    
    .stChatInput input {
        color: #2d3436 !important;
        background-color: white !important;
    }
    
    .stChatInput input::placeholder {
        color: #81c784 !important;
    }
    
    /* Chat Messages Container */
    .stChatMessage {
        background-color: transparent !important;
        padding: 1rem 15% !important;
    }
    
    /* Message Content - CRITICAL: Ensure text is visible */
    [data-testid="stChatMessageContent"] {
        background-color: white !important;
        border-radius: 1rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        color: #2d3436 !important;
    }
    
    /* Force text color in all message elements */
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageContent"] div,
    [data-testid="stChatMessageContent"] strong,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] code {
        color: #2d3436 !important;
    }
    
    /* User Message - Light Green Background */
    .stChatMessage[data-testid*="user"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%) !important;
        border-left: 3px solid #66bb6a;
    }
    
    /* Assistant Message */
    .stChatMessage[data-testid*="assistant"] [data-testid="stChatMessageContent"] {
        background: white !important;
        border-left: 3px solid #81c784;
    }
    
    /* Avatar Styling */
    .stChatMessage [data-testid="chatAvatarIcon"] {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%) !important;
        color: white !important;
    }
    
    /* Primary Buttons - Green Theme */
    .stButton > button {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%) !important;
        color: white !important;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 2px 6px rgba(76, 175, 80, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5cb85f 0%, #43a047 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    
    /* Delete Buttons - Red/Orange Theme with proper text color */
    button[kind="secondary"],
    .stButton button[title*="Delete"],
    .stButton button[aria-label*="delete"] {
        background-color: #ffebee !important;
        color: #d32f2f !important;
        border: 1px solid #ffcdd2 !important;
    }
    
    button[kind="secondary"]:hover,
    .stButton button[title*="Delete"]:hover {
        background-color: #ffcdd2 !important;
        color: #c62828 !important;
        border-color: #ef5350 !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(232, 245, 233, 0.5) 100%) !important;
        border: 2px dashed #81c784;
        border-radius: 0.75rem;
        padding: 2rem 1.5rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #66bb6a;
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.7) 0%, rgba(200, 230, 201, 0.5) 100%) !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #2d3436 !important;
    }
    
    /* Tabs - Segmented Control */
    .stTabs {
        background-color: #e8f5e9 !important;
        border-radius: 0.5rem;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0.375rem;
        color: #558b2f !important;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.5rem 0.75rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #2e7d32 !important;
        box-shadow: 0 1px 3px rgba(76, 175, 80, 0.15);
    }
    
    /* Source Card */
    .source-card {
        background-color: white !important;
        border-radius: 0.75rem;
        padding: 0.75rem;
        margin: 0.75rem 0;
        border: 1px solid #e8f5e9;
        box-shadow: 0 2px 6px rgba(76, 175, 80, 0.08);
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: #81c784;
        transform: scale(1.015) translateY(-2px);
        box-shadow: 0 6px 16px rgba(76, 175, 80, 0.15);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 1px solid #c8e6c9;
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        color: #2d3436 !important;
        background-color: white !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #66bb6a;
        box-shadow: 0 0 0 2px rgba(102, 187, 106, 0.1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #81c784 !important;
    }
    
    /* Labels */
    label, .stTextInput label {
        color: #2e7d32 !important;
        font-weight: 500;
    }
    
    /* Scrollbar - Green Tint */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f8f4;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #81c784 0%, #66bb6a 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #66bb6a 0%, #4caf50 100%);
    }
    
    /* Success/Info/Error Messages */
    .stSuccess {
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
        border-radius: 0.5rem;
        border-left: 3px solid #66bb6a;
        font-size: 0.875rem;
    }
    
    .stSuccess p, .stSuccess div {
        color: #2e7d32 !important;
    }
    
    .stInfo {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
        border-radius: 0.5rem;
        border-left: 3px solid #42a5f5;
        font-size: 0.875rem;
    }
    
    .stInfo p, .stInfo div {
        color: #1565c0 !important;
    }
    
    .stWarning {
        background-color: #fff3e0 !important;
        color: #ef6c00 !important;
        border-radius: 0.5rem;
        border-left: 3px solid #ff9800;
        font-size: 0.875rem;
    }
    
    .stWarning p, .stWarning div {
        color: #ef6c00 !important;
    }
    
    .stError {
        background-color: #ffebee !important;
        color: #c62828 !important;
        border-radius: 0.5rem;
        border-left: 3px solid #ef5350;
        font-size: 0.875rem;
    }
    
    .stError p, .stError div {
        color: #c62828 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #66bb6a !important;
    }
    
    /* Columns */
    [data-testid="column"] {
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'source_details' not in st.session_state:
    st.session_state.source_details = []  # Store metadata for display
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    # App Header
    st.markdown("""
        <div style="padding: 1.25rem 1.5rem;">
            <h1 style="color: #131614; font-size: 1.125rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;">LecturMate</h1>
            <p style="color: #6e7c74; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.25rem 0 0 0;">Source Manager</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("""
        <div style="padding: 0 1rem 1.5rem 1rem;">
            <div style="background-color: white; border: 1px solid rgba(0,0,0,0.05); padding: 0.625rem 0.75rem; border-radius: 0.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 0.25rem;">
                <span style="font-size: 0.875rem; font-weight: 500; color: #131614;">📚 Library</span>
            </div>
        </div>
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0 1.5rem 1.5rem 1.5rem;" />
    """, unsafe_allow_html=True)
    
    # Study Material Section
    st.markdown("""
        <div style="padding: 0 1.5rem;">
            <h3 style="color: #131614; font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">Study Material</h3>
            <p style="color: #6e7c74; font-size: 0.875rem; margin: 0 0 1.5rem 0;">Upload sources to chat with.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs for PDF and YouTube
    tab1, tab2 = st.tabs(["📄 Upload File", "🎥 YouTube Link"])
    
    # PDF Tab
    with tab1:
        st.markdown('<div style="padding: 1rem 0;">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.sources:
                    # Process immediately with caching
                    file_bytes = uploaded_file.getbuffer()
                    pdf_text = cached_pdf_extraction(uploaded_file.name, bytes(file_bytes))
                    
                    st.session_state.full_text += "\n\n" + pdf_text
                    st.session_state.sources.append(uploaded_file.name)
                    
                    # Store metadata
                    st.session_state.source_details.append({
                        'name': uploaded_file.name,
                        'type': 'pdf',
                        'size': len(file_bytes),
                        'time': 'Just now'
                    })
                    
                    # Auto-process with caching
                    if not st.session_state.processing:
                        st.session_state.processing = True
                        
                        # Load model once (cached)
                        model = load_embedding_model()
                        
                        # Chunk text (cached with 500 char chunks, 100 char overlap)
                        st.session_state.chunks = cached_chunking(st.session_state.full_text, 500, 100)
                        
                        # Create embeddings (cached)
                        vectors = cached_embedding(model, st.session_state.chunks)
                        
                        st.session_state.vector_db = vectors
                        st.session_state.model = model
                        st.session_state.processing = False
                    
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # YouTube Tab
    with tab2:
        st.markdown('<div style="padding: 1rem 0;">', unsafe_allow_html=True)
        
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_input",
            label_visibility="collapsed"
        )
        
        if st.button("Add YouTube Video", key="add_youtube", use_container_width=True):
            if youtube_url and youtube_url not in st.session_state.sources:
                try:
                    # Use cached YouTube transcript
                    yt_text = cached_youtube_transcript(youtube_url)
                    st.session_state.full_text += "\n\n" + yt_text
                    st.session_state.sources.append(youtube_url)
                    
                    # Store metadata
                    st.session_state.source_details.append({
                        'name': youtube_url.split('v=')[-1][:20] + "...",
                        'type': 'youtube',
                        'size': 'Video',
                        'time': 'Just now'
                    })
                    
                    # Auto-process with caching
                    if not st.session_state.processing:
                        st.session_state.processing = True
                        
                        # Load model once (cached)
                        model = load_embedding_model()
                        
                        # Chunk text (cached with 500 char chunks, 100 char overlap)
                        st.session_state.chunks = cached_chunking(st.session_state.full_text, 500, 100)
                        
                        # Create embeddings (cached)
                        vectors = cached_embedding(model, st.session_state.chunks)
                        
                        st.session_state.vector_db = vectors
                        st.session_state.model = model
                        st.session_state.processing = False
                    
                    st.success("✅ Added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            elif youtube_url in st.session_state.sources:
                st.warning("Already added!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Active Sources
    st.markdown(f"""
        <div style="padding: 1.5rem 1.5rem 0.5rem 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="font-size: 0.875rem; font-weight: 600; color: #131614; margin: 0;">Active Sources</h4>
                <span style="font-size: 0.75rem; color: #6e7c74; background-color: #f5f5f4; padding: 0.125rem 0.5rem; border-radius: 9999px;">{len(st.session_state.sources)}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.source_details:
        for idx, source in enumerate(st.session_state.source_details):
            icon = "📄" if source['type'] == 'pdf' else "🎥"
            size_text = f"{source['size'] / (1024*1024):.1f} MB" if isinstance(source['size'], int) else source['size']
            
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                    <div class="source-card">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div style="flex: 1; min-width: 0;">
                            <h5 style="font-size: 0.875rem; font-weight: 600; color: #131614; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{source['name']}</h5>
                            <p style="font-size: 0.75rem; color: #6e7c74; margin: 0.25rem 0 0 0;">{size_text} • {source['time']}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{idx}", help="Delete source"):
                    # Remove source
                    st.session_state.sources.pop(idx)
                    st.session_state.source_details.pop(idx)
                    # Reprocess if sources remain
                    if st.session_state.sources:
                        st.session_state.processing = True
                        
                        # Load model once (cached)
                        model = load_embedding_model()
                        
                        # Chunk text (cached with 500 char chunks, 100 char overlap)
                        st.session_state.chunks = cached_chunking(st.session_state.full_text, 500, 100)
                        
                        # Create embeddings (cached)
                        vectors = cached_embedding(model, st.session_state.chunks)
                        
                        st.session_state.vector_db = vectors
                        st.session_state.model = model
                        st.session_state.processing = False
                    else:
                        st.session_state.full_text = ""
                        st.session_state.vector_db = None
                        st.session_state.model = None
                        st.session_state.chunks = None
                    st.rerun()
    else:
        st.markdown("""
            <div style="padding: 0 1.5rem;">
                <p style="font-size: 0.875rem; color: #6e7c74; text-align: center; padding: 2rem 0;">No sources added yet</p>
            </div>
        """, unsafe_allow_html=True)

# Main Chat Interface
st.markdown("""
    <div style="height: 4rem; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: space-between; padding: 0 2rem; background-color: rgba(250, 250, 249, 0.8); backdrop-filter: blur(12px);">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="width: 0.5rem; height: 0.5rem; border-radius: 9999px; background-color: #478561; animation: pulse 2s ease-in-out infinite;"></div>
            <h2 style="color: #131614; font-weight: 600; font-size: 1rem; margin: 0;">Discussion</h2>
        </div>
    </div>
""", unsafe_allow_html=True)

# Chat Messages Container
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Welcome message
        st.markdown('<div style="padding: 4rem 15%; text-align: center;"><div style="font-size: 0.75rem; font-weight: 500; color: rgba(110, 124, 116, 0.6); text-transform: uppercase; letter-spacing: 0.1em;">Today</div></div>', unsafe_allow_html=True)
        
        if st.session_state.sources:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"Hello! I've analyzed the {len(st.session_state.sources)} source{'s' if len(st.session_state.sources) > 1 else ''} you've uploaded. I'm ready to help you study. You can ask me to summarize key concepts, explain topics, or create a quiz based on your materials.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your sources..."):
    if st.session_state.vector_db is None:
        st.error("⚠️ Please add sources first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant chunks
                    top_chunks = search_best_chunks(
                        query=prompt,
                        model=st.session_state.model,
                        db_vectors=st.session_state.vector_db,
                        chunks=st.session_state.chunks,
                        k=3
                    )
                    
                    # Combine context
                    context = "\n\n".join([chunk['text'] for chunk in top_chunks])
                    
                    # Generate answer
                    answer = generate_answer(prompt, context)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Save message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

