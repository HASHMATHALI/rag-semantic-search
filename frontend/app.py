import streamlit as st
import requests

# Set page config for a premium aesthetic
st.set_page_config(page_title="Semantic RAG Search", page_icon="🔍", layout="wide")

# Custom CSS for a beautiful, premium design
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background-color: #0B0E14; /* Deep modern dark mode */
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Stunning Main Title */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 2rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #A0AAB2;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Result Card with Micro-Animations */
    .result-card {
        background: rgba(30, 33, 39, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        border-color: rgba(78, 205, 196, 0.4);
    }
    
    /* Badges */
    .score-badge {
        background: linear-gradient(90deg, #4ECDC4, #2AB7A8);
        color: #0B0E14;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    
    /* RAG Box gradient */
    .rag-box {
        background: linear-gradient(145deg, rgba(42, 45, 52, 0.8), rgba(30, 33, 39, 0.8));
        border-left: 5px solid #FF6B6B;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 40px;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Semantic Search ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Embedding-based RAG Powered by FAISS & FastAPI</p>', unsafe_allow_html=True)

# Application state
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# Sidebar parameters
with st.sidebar:
    st.header("⚙️ Search Engine Settings")
    top_k = st.slider("Documents to Retrieve", min_value=1, max_value=20, value=5)
    use_rag = st.checkbox("Generate Answer (RAG)", value=False, help="Requires GROQ_API_KEY running on backend")
    api_url = st.text_input("Backend URL", value="http://localhost:8002/search")
    
    st.markdown("---")
    st.markdown("**Status:** Vectors loaded in RAM")

# Main search bar
query = st.text_input("Enter your query:", placeholder="e.g. Technology advancements in autonomous vehicles...", label_visibility="collapsed")

if st.button("Search Database", type="primary", use_container_width=True) or query:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("🚀 Searching millions of vectors instantly..."):
            try:
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "use_rag": use_rag
                }
                response = requests.post(api_url, json=payload)
                if response.status_code == 200:
                    st.session_state.search_results = response.json()
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the Backend! Verify FastAPI is running on port 8000.")

# Display results
if st.session_state.search_results:
    data = st.session_state.search_results
    
    if data.get("rag_response"):
        st.markdown("### 🤖 Synthesized Answer (RAG)")
        st.markdown(f'<div class="rag-box">{data["rag_response"]}</div>', unsafe_allow_html=True)
    elif use_rag:
        st.markdown("### 🤖 Synthesized Answer (RAG)")
        st.markdown(f'<div class="rag-box" style="border-left-color: #888;">RAG was requested but no response generated. Have you set the OpenAI key on the backend?</div>', unsafe_allow_html=True)

    st.markdown(f"### 📂 Retrieved Contexts for: *{data['query']}*")
    for res in data["results"]:
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px; align-items: center;">
                <span style="color: #FF6B6B; font-weight: 600; font-size: 0.9rem;">DOCUMENT ID: {res['id']}</span>
                <span class="score-badge">L2 DISTANCE: {res['score']:.4f}</span>
            </div>
            <p style="color: #E2E8F0; font-size: 1.0rem; line-height: 1.5;">{res['text']}</p>
        </div>
        """, unsafe_allow_html=True)
