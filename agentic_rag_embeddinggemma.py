import streamlit as st
from pathlib import Path
from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.vectordb.lancedb import LanceDb, SearchType

BASE_DIR = Path(__file__).resolve().parent
LANCEDB_URI = BASE_DIR.parent / "tmp" / "lancedb"


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&family=Material+Symbols+Rounded:FILL@0;1&family=Material+Symbols+Outlined:FILL@0;1&display=swap');
            @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

            :root {
                --bg: #050b1e;
                --panel: #0b1736;
                --panel-2: #101f46;
                --border: #2f4472;
                --text: #dce6ff;
                --muted: #9bb0df;
                --accent: #8389ff;
                --accent-2: #5f66e6;
                --chip: #132857;
            }

            .stApp {
                background:
                    radial-gradient(1200px 600px at 80% -10%, #182f6e 0%, transparent 55%),
                    radial-gradient(900px 500px at -10% 20%, #0b2456 0%, transparent 45%),
                    var(--bg);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #06122d 0%, #050c20 100%);
                border-right: 1px solid rgba(111, 138, 197, 0.25);
            }

            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] a {
                color: var(--text);
                font-family: 'Plus Jakarta Sans', sans-serif;
            }

            [data-testid="stSidebar"] .material-symbols-rounded,
            [data-testid="stSidebar"] [class*="material-symbols"] {
                font-family: 'Material Symbols Rounded' !important;
            }

            [data-testid="stSidebarCollapseButton"],
            button[kind="headerNoPadding"],
            [data-testid="collapsedControl"] {
                opacity: 1 !important;
                visibility: visible !important;
            }

            [data-testid="stSidebarCollapseButton"] button,
            button[kind="headerNoPadding"],
            [data-testid="collapsedControl"] button {
                background: rgba(121, 142, 208, 0.14) !important;
                border: 1px solid rgba(130, 151, 217, 0.35) !important;
                border-radius: 10px !important;
                color: #dce6ff !important;
            }

            [data-testid="stSidebarCollapseButton"] button span,
            button[kind="headerNoPadding"] span,
            [data-testid="collapsedControl"] button span {
                color: #dce6ff !important;
                font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
                font-size: 22px !important;
                line-height: 1 !important;
                font-feature-settings: 'liga' 1;
                letter-spacing: normal !important;
            }

            [data-testid="stSidebarCollapseButton"] button p,
            button[kind="headerNoPadding"] p,
            [data-testid="collapsedControl"] button p,
            [data-testid="stSidebarCollapseButton"] button div,
            button[kind="headerNoPadding"] div,
            [data-testid="collapsedControl"] button div {
                font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            }

            .brand-title {
                font-family: 'Manrope', sans-serif;
                font-weight: 800;
                font-size: 1.2rem;
                letter-spacing: -0.02em;
                margin-bottom: 0;
            }

            .brand-subtitle {
                color: var(--muted);
                font-size: 0.82rem;
                margin-top: 0.1rem;
                margin-bottom: 1.2rem;
            }

            .nav-label {
                margin-top: 1.3rem;
                margin-bottom: 0.55rem;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.67rem;
                color: #7f97ce;
                font-weight: 700;
            }

            .nav-item {
                padding: 0.52rem 0.7rem;
                border-radius: 0.65rem;
                margin-bottom: 0.28rem;
                color: #9fb2df;
                border: 1px solid transparent;
                background: transparent;
                font-size: 0.92rem;
                font-weight: 500;
            }

            .nav-item.active {
                background: rgba(128, 142, 255, 0.16);
                border-color: rgba(149, 163, 255, 0.35);
                color: #dfe6ff;
            }

            .top-bar {
                background: rgba(7, 16, 39, 0.7);
                border: 1px solid rgba(79, 106, 164, 0.28);
                border-radius: 14px;
                padding: 0.72rem 1rem;
                margin-bottom: 1.35rem;
                backdrop-filter: blur(8px);
            }

            .top-title {
                font-family: 'Manrope', sans-serif;
                font-size: 1.15rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin: 0;
            }

            .status-pill {
                display: inline-block;
                margin-top: 0.25rem;
                border-radius: 999px;
                padding: 0.2rem 0.62rem;
                font-size: 0.67rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                background: rgba(112, 150, 255, 0.17);
                border: 1px solid rgba(141, 169, 255, 0.32);
                color: #d4deff;
            }

            .hero-card {
                border: 1px solid rgba(83, 114, 179, 0.28);
                border-radius: 18px;
                padding: 1.2rem 1.3rem 1rem;
                background: linear-gradient(180deg, rgba(12, 31, 71, 0.92), rgba(8, 21, 49, 0.93));
                margin-bottom: 0.85rem;
            }

            .hero-title {
                font-family: 'Manrope', sans-serif;
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.12;
                margin: 0;
                letter-spacing: -0.03em;
                font-weight: 800;
                color: #e6ebff;
            }

            .hero-subtitle {
                margin-top: 0.8rem;
                margin-bottom: 0;
                color: var(--muted);
                max-width: 770px;
                font-size: 1.03rem;
                line-height: 1.65;
            }

            .feature-card {
                border-radius: 14px;
                border: 1px solid rgba(76, 101, 154, 0.28);
                padding: 1.05rem 1.1rem;
                background: linear-gradient(160deg, rgba(12, 30, 67, 0.95), rgba(7, 20, 47, 0.95));
                min-height: 175px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            }

            .feature-title {
                margin: 0.35rem 0 0.25rem;
                font-family: 'Manrope', sans-serif;
                font-size: 1.14rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }

            .feature-body {
                margin: 0;
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.55;
            }

            .section-title {
                font-family: 'Manrope', sans-serif;
                font-size: 1.78rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin: 0.2rem 0 0.72rem;
            }

            .query-box {
                border-radius: 16px;
                border: 1px solid rgba(90, 118, 180, 0.3);
                background: linear-gradient(180deg, rgba(15, 35, 78, 0.95), rgba(12, 29, 65, 0.95));
                padding: 1rem 1rem 0.35rem;
                box-shadow: 0 18px 45px rgba(4, 10, 28, 0.4);
            }

            .source-item {
                border: 1px solid rgba(84, 109, 164, 0.28);
                border-radius: 11px;
                background: rgba(11, 28, 63, 0.84);
                padding: 0.7rem 0.85rem;
                margin-bottom: 0.45rem;
                font-size: 0.92rem;
            }

            .source-index {
                color: #9eb0db;
                font-weight: 700;
                margin-right: 0.35rem;
            }

            .source-empty {
                border: 1px dashed rgba(95, 121, 179, 0.42);
                border-radius: 11px;
                padding: 0.9rem;
                color: var(--muted);
                background: rgba(8, 21, 48, 0.5);
            }

            .stTextInput > div > div > input,
            .stTextArea textarea {
                background: rgba(8, 19, 46, 0.86);
                color: #e4eaff;
                border: 1px solid rgba(83, 111, 170, 0.35);
                border-radius: 11px;
                font-family: 'Plus Jakarta Sans', sans-serif;
            }

            .stTextInput > div > div > input:focus,
            .stTextArea textarea:focus {
                border-color: rgba(139, 160, 255, 0.7);
                box-shadow: 0 0 0 0.15rem rgba(126, 143, 255, 0.2);
            }

            div[data-testid="stButton"] > button {
                border: 0;
                border-radius: 10px;
                font-family: 'Manrope', sans-serif;
                font-weight: 700;
                padding: 0.6rem 1.05rem;
                transition: all 0.2s ease;
                color: #0f1d4f;
                background: linear-gradient(120deg, #a9b6ff, #7f8dff);
            }

            div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 7px 24px rgba(136, 154, 255, 0.36);
            }

            div[data-testid="stExpander"] {
                border: 1px solid rgba(83, 110, 165, 0.3) !important;
                border-radius: 11px !important;
                background: rgba(10, 24, 55, 0.78) !important;
            }

            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li {
                color: #d6e0ff;
            }

            @media (max-width: 900px) {
                .hero-card {
                    padding: 1rem;
                }
                .hero-subtitle {
                    font-size: 0.96rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_url_to_knowledge_base(url: str, kb: Knowledge):
    if not url:
        st.error("Please enter a URL")
        return

    if url in st.session_state.urls:
        st.warning("This URL has already been added.")
        return

    st.session_state.urls.append(url)
    with st.spinner("Adding source to knowledge base..."):
        kb.add_content(url=url)
        st.session_state.urls_loaded.add(url)
    st.success(f"Added source: {url}")
    st.rerun()


def render_sidebar(kb: Knowledge):
    with st.sidebar:
        st.markdown('<p class="brand-title">Curator AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="brand-subtitle">Local Intelligence</p>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)

        st.markdown('<p class="nav-label">Upload</p>', unsafe_allow_html=True)
        with st.form("add_source_form", clear_on_submit=True):
            url_input = st.text_input(
                "Source URL",
                placeholder="https://example.com/paper.pdf",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Upload Source")
            if submitted:
                add_url_to_knowledge_base(url_input.strip(), kb)

        st.markdown('<p class="nav-label">Navigation</p>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item active">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">Recent Queries</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item">System Status</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Documentation")
        st.caption("Help")


def render_top_bar():
    st.markdown(
        """
        <div class="top-bar">
            <p class="top-title">Local Curator</p>
            <span class="status-pill">Local Engine Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section():
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Agentic RAG with EmbeddingGemma (Local)</h1>
            <p class="hero-subtitle">
                Interact with your data through a private, high-performance retrieval engine designed for
                researchers. Complete privacy with local inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards():
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:1.7rem;">🛡️</div>
                <h3 class="feature-title">Local-first inference</h3>
                <p class="feature-body">All processing runs on your machine. No data leaves your hardware.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:1.7rem;">🔗</div>
                <h3 class="feature-title">URL-based PDF ingestion</h3>
                <p class="feature-body">Add documentation URLs and index them instantly as searchable context.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:1.7rem;">🗂️</div>
                <h3 class="feature-title">LanceDB vector search</h3>
                <p class="feature-body">Fast, local semantic retrieval using embedding vectors and similarity search.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_query_section(agent: Agent):
    st.markdown("<h2 class='section-title'>Ask Anything</h2>", unsafe_allow_html=True)
    st.markdown('<div class="query-box">', unsafe_allow_html=True)

    query = st.text_area(
        "Ask a complex question about your local documents",
        placeholder="Ask a complex question about your local documents...",
        label_visibility="collapsed",
        height=140,
        key="main_query",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            """
            <div style="display:flex; gap:8px; margin-top: 4px; margin-bottom: 8px;">
                <span style="background: var(--chip); border: 1px solid rgba(129,153,215,0.35); color:#b8c9f1; font-size:0.66rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; border-radius:999px; padding:0.24rem 0.58rem;">Knowledge Active</span>
                <span style="background: var(--chip); border: 1px solid rgba(129,153,215,0.35); color:#b8c9f1; font-size:0.66rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; border-radius:999px; padding:0.24rem 0.58rem;">Gemma + Llama3.2</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        run_clicked = st.button("Get Answer", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        if not query.strip():
            st.error("Please enter a question")
            return

        st.markdown("### Answer")
        with st.spinner("Searching knowledge and generating answer..."):
            try:
                response = ""
                resp_container = st.empty()
                gen = agent.run(query, stream=True)
                for resp_chunk in gen:
                    if resp_chunk.content is not None:
                        response += resp_chunk.content
                        resp_container.markdown(response)
            except Exception as exc:
                st.error(f"Error: {exc}")


def render_knowledge_sources():
    st.markdown("<h2 class='section-title'>Knowledge Sources</h2>", unsafe_allow_html=True)
    if st.session_state.urls:
        for idx, source_url in enumerate(st.session_state.urls, 1):
            st.markdown(
                f"<div class='source-item'><span class='source-index'>{idx:02d}</span>{source_url}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='source-empty'>No sources added yet. Use the Upload Source box in the sidebar to add a PDF URL.</div>",
            unsafe_allow_html=True,
        )


def render_how_it_works():
    st.markdown("<h2 class='section-title'>How This Works</h2>", unsafe_allow_html=True)
    with st.expander("Neural Retrieval Pipeline", expanded=True):
        st.markdown(
            """
            EmbeddingGemma transforms your documents into vector representations.
            For each question, the app performs semantic similarity search in LanceDB,
            retrieves the most relevant chunks, and forwards them to the language model.
            """
        )

    with st.expander("Agentic Reasoning Loop", expanded=False):
        st.markdown(
            """
            The agent uses retrieved context as grounding and then generates a response
            with Llama 3.2. Responses are streamed token-by-token for interactive UX.
            """
        )

st.set_page_config(
    page_title="Agentic RAG with Google's EmbeddingGemma",
    page_icon="🔥",
    layout="wide"
)

inject_styles()

@st.cache_resource
def load_knowledge_base():
    knowledge_base = Knowledge(
        vector_db=LanceDb(
            table_name="recipes",
            uri=str(LANCEDB_URI),
            search_type=SearchType.vector,
            embedder=OllamaEmbedder(id="embeddinggemma:latest", dimensions=768),
        ),
    )
    return knowledge_base

if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'urls_loaded' not in st.session_state:
    st.session_state.urls_loaded = set()

kb = load_knowledge_base()

for url in st.session_state.urls:
    if url not in st.session_state.urls_loaded:
        kb.add_content(url=url)
        st.session_state.urls_loaded.add(url)

agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    knowledge=kb,
    instructions=[
        "Search the knowledge base for relevant information and base your answers on it.",
        "Be clear, and generate well-structured answers.",
        "Use clear headings, bullet points, or numbered lists where appropriate.",
    ],
    search_knowledge=True,
    debug_mode=False,
    markdown=True,
)

render_sidebar(kb)
render_top_bar()
render_hero_section()
render_feature_cards()
st.markdown("<div style='height: 1.3rem;'></div>", unsafe_allow_html=True)
render_query_section(agent)
st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)

left_col, right_col = st.columns(2, gap="large")
with left_col:
    render_knowledge_sources()
with right_col:
    render_how_it_works()
