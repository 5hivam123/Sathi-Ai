"""
SaathiAI - Your Personal AI Learning Companion
Main Streamlit application entry point with custom Moon Walk typography & refined color palette.
"""

import os
import base64
from pathlib import Path
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SaathiAI – Your Personal AI Learning Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    ALLOWED_FILE_TYPES,
    ADMIN_PASSWORD,
    LLM_BACKEND,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_MODEL,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODEL,
)
from src.pdf_processor import process_document, DocumentProcessingError, sanitize_filename
from src.text_splitter import split_pages_into_chunks, split_text_into_chunks
from src.embeddings import get_embedding_model, embed_texts
from src.vector_store import VectorStore
from src.gemma import get_backend_display_name


# ── Custom Font Loader (Casko) ────────────────────────────────────────────────
@st.cache_data
def get_font_base64() -> str:
    font_path = Path(__file__).parent / "assets" / "fonts" / "casko.otf"
    if not font_path.exists():
        font_path = Path(__file__).parent / "assets" / "fonts" / "casko" / "CaskoLuxuryDemoRegular-Wp5lE.otf"
    if not font_path.exists():
        font_path = Path(__file__).parent / "assets" / "fonts" / "moon_walk.ttf"
    if font_path.exists():
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# ── CSS & Theme ───────────────────────────────────────────────────────────────
def load_css():
    font_b64 = get_font_base64()
    font_face_rule = ""
    if font_b64:
        font_face_rule = f"""
        @font-face {{
            font-family: 'Casko';
            src: url('data:font/opentype;charset=utf-8;base64,{font_b64}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}
        """

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

{font_face_rule}

/* ── Palette Tokens: 8B9A6E | F7F2EB | EAE2D6 | EEEEEE ── */
:root {{
    --primary: #8B9A6E;
    --primary-hover: #758359;
    --primary-light: #A4B585;
    --primary-dim: rgba(139, 154, 110, 0.15);
    
    --bg-main: #F7F2EB;
    --bg-card: #FFFFFF;
    --bg-card-alt: #EAE2D6;
    --bg-sidebar: #EAE2D6;
    --bg-neutral: #EEEEEE;
    
    --text-primary: #2C3524;
    --text-secondary: #58634E;
    --text-muted: #828E75;
    
    --border: #DCD4C7;
    --border-light: #EBE4D8;
    --border-accent: #8B9A6E;
    
    --success: #52796F;
    --warning: #DDA15E;
    --error: #BC4749;
    
    --gradient-hero: linear-gradient(135deg, #8B9A6E 0%, #5E6D45 100%);
    --gradient-btn: linear-gradient(135deg, #8B9A6E 0%, #738257 100%);
    --gradient-card: linear-gradient(135deg, #FFFFFF 0%, #F8F4ED 100%);
    --shadow-sm: 0 2px 10px rgba(44, 53, 36, 0.05);
    --shadow-md: 0 8px 24px rgba(139, 154, 110, 0.12);
}}

html, body, [class*="st-"] {{
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    color: var(--text-primary);
}}

/* Casko Font for Headings, Titles & Brand Marks */
h1, h2, h3, h4, .casko-font, .hero-title, .feature-title, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    font-family: 'Casko', 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.02em;
}}

/* Main App Container */
.stApp {{
    background-color: var(--bg-main) !important;
}}

/* ── Sidebar Whitespace Fix ── */
[data-testid="stSidebar"] {{
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebarContent"] {{
    padding-top: 1.2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

[data-testid="stSidebarUserContent"] {{
    padding-top: 0 !important;
}}

[data-testid="stSidebarHeader"] {{
    display: none !important;
}}

[data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
    margin: 0.8rem 0 !important;
}}

/* ── Interactive Buttons ── */
.stButton > button {{
    background: var(--gradient-btn) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(139, 154, 110, 0.25) !important;
    width: 100% !important;
}}

.stButton > button:hover {{
    background: var(--primary-hover) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 154, 110, 0.35) !important;
}}

.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* Secondary / Inactive Navigation Button */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
    background: var(--bg-main) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}

/* ── Inputs & TextAreas ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-dim) !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
}}

/* ── Material Icons & Expander Ligature Protection ── */
[data-testid="stIconMaterial"], 
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderIcon"],
[data-testid="stIcon"],
.material-symbols-rounded,
.material-icons,
[class*="material-"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
}}

/* ── Expanders ── */
.streamlit-expanderHeader,
details summary {{
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}}

details summary:hover {{
    border-color: var(--primary) !important;
    background: #FFFFFF !important;
}}

/* Ensure text inside expander header looks clean without overriding the icon font */
details summary [data-testid="stMarkdownContainer"] p,
details summary p {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    display: inline-block !important;
}}

details summary [data-testid="stExpanderToggleIcon"],
details summary svg {{
    color: var(--primary) !important;
    flex-shrink: 0 !important;
}}

/* ── File Uploader Clean Drop Zone Fix ── */
[data-testid="stFileUploader"] {{
    background: var(--bg-card) !important;
    border: 2px dashed var(--primary) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
}}

[data-testid="stFileUploader"] section {{
    background: transparent !important;
    padding: 0.5rem !important;
}}

[data-testid="stFileUploader"] button {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transform: none !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background: var(--primary-hover) !important;
    transform: none !important;
    box-shadow: none !important;
}}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}}

/* ── Divider ── */
hr {{
    border-color: var(--border) !important;
}}

/* ── Callout Alerts ── */
.stSuccess {{
    background: rgba(82, 121, 111, 0.12) !important;
    border: 1px solid var(--success) !important;
    border-radius: 12px !important;
    color: #244037 !important;
}}
.stError {{
    background: rgba(188, 71, 73, 0.12) !important;
    border: 1px solid var(--error) !important;
    border-radius: 12px !important;
    color: #611D1F !important;
}}
.stWarning {{
    background: rgba(221, 161, 94, 0.14) !important;
    border: 1px solid var(--warning) !important;
    border-radius: 12px !important;
    color: #5C3D17 !important;
}}
.stInfo {{
    background: rgba(139, 154, 110, 0.14) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}}

/* ── Custom Cards ── */
.saathi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    margin: 0.6rem 0;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}}
.saathi-card:hover {{
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}}

.hero-title {{
    font-family: 'Casko', sans-serif !important;
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.15;
    margin-bottom: 0.5rem;
}}

.hero-tagline {{
    font-family: 'Casko', 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.35rem;
    color: var(--primary-hover);
    font-weight: 600;
    margin-bottom: 1.2rem;
}}

.feature-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
    box-shadow: var(--shadow-sm);
}}
.feature-card:hover {{
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-4px);
    background: var(--gradient-card);
}}
.feature-icon {{
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
    display: block;
}}
.feature-title {{
    font-family: 'Casko', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}}
.feature-desc {{
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.65;
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}}
.status-ready {{
    background: rgba(139, 154, 110, 0.2);
    color: #44542E;
    border: 1px solid var(--primary);
}}
.status-not-ready {{
    background: rgba(221, 161, 94, 0.2);
    color: #8C5517;
    border: 1px solid var(--warning);
}}

.source-chip {{
    display: inline-block;
    background: rgba(139, 154, 110, 0.15);
    border: 1px solid var(--primary);
    border-radius: 16px;
    padding: 0.25rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0.2rem;
}}

.answer-box {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1rem;
    line-height: 1.85;
    white-space: pre-wrap;
    box-shadow: var(--shadow-sm);
    color: var(--text-primary);
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.6rem;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--text-secondary);
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border-color: var(--primary) !important;
}}

/* ── Scrollbars ── */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg-main); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "vector_store": VectorStore(),
        "embedding_model": None,
        "current_doc": None,
        "doc_processed": False,
        "page": "home",
        "chat_history": [],
        "quiz_state": None,
        "flashcard_index": 0,
        "flashcard_flipped": False,
        "flashcard_deck": None,
        "is_admin": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(show_spinner="Loading AI embedding model…")
def load_embedding_model_cached():
    return get_embedding_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / Branding
        st.markdown(f"""
<div style="text-align:center; padding: 0.2rem 0 0.8rem 0;">
  <div style="font-size:2.4rem; line-height:1;">🎓</div>
  <div class="casko-font" style="font-size:2.2rem; font-weight:800; color:#2C3524; letter-spacing:0.02em; margin-top:0.2rem;">
    {APP_NAME}
  </div>
  <div style="font-size:0.8rem; color:#58634E; margin-top:0.1rem; font-weight:500;">
    {APP_TAGLINE}
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        nav_items = [
            ("🏠", "Home", "home"),
            ("💬", "Ask My Notes", "chat"),
            ("📝", "Exam Mode", "exam"),
            ("🎯", "Quiz Me", "quiz"),
            ("🧠", "Flashcards", "flashcards"),
            ("⚡", "Quick Revision", "revision"),
        ]

        for icon, label, page_key in nav_items:
            active = st.session_state.page == page_key
            if st.button(
                f"{icon} {label}",
                key=f"nav_{page_key}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("<div style='margin: 0.4rem 0;'>", unsafe_allow_html=True)
        if st.button(
            "🚨 Exam Tomorrow",
            key="nav_exam_tomorrow",
            type="primary" if st.session_state.page == "exam_tomorrow" else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = "exam_tomorrow"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Document Status
        st.markdown("### 📚 Current Material")
        if st.session_state.doc_processed and st.session_state.current_doc:
            doc = st.session_state.current_doc
            st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #DCD4C7; border-left:4px solid #8B9A6E;
            border-radius:14px; padding:0.9rem; box-shadow:0 2px 8px rgba(44,53,36,0.05);">
  <div style="font-weight:700; color:#2C3524; font-size:0.9rem; word-break:break-all;">
    📄 {doc.filename}
  </div>
  <div style="margin-top:0.3rem;">
    <span class="status-badge status-ready">✓ Ready</span>
  </div>
  <div style="color:#58634E; font-size:0.8rem; margin-top:0.3rem;">
    {doc.page_count} pages · {st.session_state.vector_store.chunk_count} chunks
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#FFFFFF; border:1px solid #DCD4C7; border-left:4px solid #DDA15E;
            border-radius:14px; padding:0.9rem; box-shadow:0 2px 8px rgba(44,53,36,0.05);">
  <div style="color:#8C5517; font-size:0.85rem; font-weight:600;">⚠ No document uploaded</div>
  <div style="color:#828E75; font-size:0.78rem; margin-top:0.2rem;">
      Upload on Home page to begin
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Admin-Only API & Engine Settings ──
        if not st.session_state.get("is_admin", False):
            with st.expander("🔐 Admin API Settings", expanded=False):
                st.markdown(
                    "<div style='font-size:0.82rem; color:#58634E; line-height:1.45; margin-bottom:0.6rem;'>"
                    "🔒 Only administrators can configure AI models and link API keys."
                    "</div>",
                    unsafe_allow_html=True,
                )
                admin_pwd = st.text_input(
                    "Admin Password",
                    type="password",
                    placeholder="Enter password",
                    key="admin_pwd_login_input",
                )
                if st.button("🔓 Login as Admin", use_container_width=True, key="btn_login_admin"):
                    if admin_pwd and admin_pwd == ADMIN_PASSWORD:
                        st.session_state.is_admin = True
                        st.success("✓ Authenticated as Admin!")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
        else:
            with st.expander("⚙️ Admin: AI Engine & API Keys", expanded=True):
                st.markdown(
                    """<div style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.6rem;
                                 background:rgba(82,121,111,0.15); border:1px solid #52796F; border-radius:12px;
                                 color:#2C473E; font-size:0.78rem; font-weight:700; margin-bottom:0.8rem;">
                         🟢 Admin Mode Active
                       </div>""",
                    unsafe_allow_html=True,
                )
                backend_choice = st.selectbox(
                    "Inference Engine",
                    options=["jev", "gemini", "groq", "ollama", "openai_compat", "huggingface"],
                    format_func=lambda x: {
                        "jev": "🎯 Jev AI (Custom Integrated)",
                        "gemini": "✨ Google Gemini API (Recommended)",
                        "groq": "⚡ Groq Cloud (Super Fast)",
                        "ollama": "🦙 Ollama (Local Self-Hosted)",
                        "openai_compat": "🤖 OpenAI / Custom API",
                        "huggingface": "🤗 Local HuggingFace",
                    }.get(x, x),
                    index=["jev", "gemini", "groq", "ollama", "openai_compat", "huggingface"].index(
                        os.getenv("LLM_BACKEND", LLM_BACKEND) if os.getenv("LLM_BACKEND", LLM_BACKEND) in ["jev", "gemini", "groq", "ollama", "openai_compat", "huggingface"] else "jev"
                    ),
                    key="sidebar_backend_select",
                )
                os.environ["LLM_BACKEND"] = backend_choice

                if backend_choice == "jev":
                    jev_key = st.text_input(
                        "Jev AI Key",
                        value=os.getenv("JEV_API_KEY", "sk_jev-ai_a_orq1I0PYvgBe0TrYKNCOjxSEhByk2A"),
                        type="password",
                        key="sidebar_jev_key",
                    )
                    if jev_key:
                        os.environ["JEV_API_KEY"] = jev_key
                    jev_url = st.text_input(
                        "Jev Base URL",
                        value=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1"),
                        key="sidebar_jev_url",
                    )
                    if jev_url:
                        os.environ["JEV_BASE_URL"] = jev_url
                    jev_model = st.selectbox(
                        "Model",
                        ["jev-latest", "llama-3.3-70b", "gpt-4o-mini", "claude-3-5-sonnet"],
                        key="sidebar_jev_model",
                    )
                    os.environ["JEV_MODEL"] = jev_model

                elif backend_choice == "gemini":
                    gemini_key = st.text_input(
                        "Gemini API Key",
                        value=os.getenv("GEMINI_API_KEY", ""),
                        type="password",
                        help="Get free key at aistudio.google.com",
                        key="sidebar_gemini_key",
                    )
                    if gemini_key:
                        os.environ["GEMINI_API_KEY"] = gemini_key
                    gemini_model = st.selectbox(
                        "Model",
                        ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b"],
                        key="sidebar_gemini_model",
                    )
                    os.environ["GEMINI_MODEL"] = gemini_model

                elif backend_choice == "groq":
                    groq_key = st.text_input(
                        "Groq API Key",
                        value=os.getenv("GROQ_API_KEY", ""),
                        type="password",
                        help="Get free key at console.groq.com",
                        key="sidebar_groq_key",
                    )
                    if groq_key:
                        os.environ["GROQ_API_KEY"] = groq_key
                    groq_model_choice = st.selectbox(
                        "Groq Model",
                        [
                            "llama-3.3-70b-versatile",
                            "llama-3.1-8b-instant",
                            "deepseek-r1-distill-llama-70b",
                            "gemma2-9b-it",
                            "qwen/qwen3.6-27b",
                            "openai/gpt-oss-120b",
                            "✏️ Enter Custom Model ID",
                        ],
                        key="sidebar_groq_model_choice",
                    )
                    if groq_model_choice == "✏️ Enter Custom Model ID":
                        groq_model = st.text_input(
                            "Custom Groq Model ID",
                            value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                            key="sidebar_groq_custom_model",
                        )
                    else:
                        groq_model = groq_model_choice
                    os.environ["GROQ_MODEL"] = groq_model

                elif backend_choice == "ollama":
                    ollama_model = st.selectbox(
                        "Ollama Model",
                        [
                            "llama3.2:latest",
                            "llama3.1:8b",
                            "gemma2:2b",
                            "gemma2:9b",
                            "phi4:latest",
                            "qwen2.5:7b",
                            "mistral:latest",
                        ],
                        key="sidebar_ollama_model",
                    )
                    os.environ["OLLAMA_MODEL"] = ollama_model

                elif backend_choice == "openai_compat":
                    compat_key = st.text_input(
                        "API Key",
                        value=os.getenv("OPENAI_COMPAT_API_KEY", ""),
                        type="password",
                        key="sidebar_compat_key",
                    )
                    compat_url = st.text_input(
                        "Base URL",
                        value=os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"),
                        key="sidebar_compat_url",
                    )
                    compat_model = st.selectbox(
                        "Model",
                        ["gpt-4o-mini", "gpt-4o", "deepseek-chat", "llama-3.3-70b-instruct"],
                        key="sidebar_compat_model",
                    )
                    if compat_key:
                        os.environ["OPENAI_COMPAT_API_KEY"] = compat_key
                    if compat_url:
                        os.environ["OPENAI_COMPAT_BASE_URL"] = compat_url
                    if compat_model:
                        os.environ["OPENAI_COMPAT_MODEL"] = compat_model

                st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
                if st.button("🔒 Lock & Exit Admin", use_container_width=True, key="btn_lock_admin"):
                    st.session_state.is_admin = False
                    st.rerun()

        st.markdown("---")
        st.markdown(f"""
<div style="font-size:0.75rem; color:#828E75; text-align:center;">
  v{APP_VERSION} · Open-Source AI<br>
  Engine: {get_backend_display_name()[:30]}
</div>
""", unsafe_allow_html=True)


# ── Upload Section ────────────────────────────────────────────────────────────
def render_upload_section():
    """Renders the document upload panel and handles processing."""
    st.markdown("### 📤 Upload Study Material")

    uploaded = st.file_uploader(
        "Upload your PDF or TXT study material",
        type=ALLOWED_FILE_TYPES,
        key="file_uploader",
        help=f"Supported: {', '.join(f.upper() for f in ALLOWED_FILE_TYPES)} · Max 50 MB",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        filename = sanitize_filename(uploaded.name)

        # Check if this is a new file
        is_new = (
            st.session_state.current_doc is None
            or st.session_state.current_doc.filename != filename
            or len(file_bytes) != getattr(st.session_state, "_last_file_size", -1)
        )

        if is_new:
            with st.status("Processing your study material…", expanded=True) as status:
                try:
                    # Step 1: Extract text
                    st.write("📄 Extracting text from document…")
                    doc = process_document(filename, file_bytes)
                    st.write(f"✓ {doc.page_count} pages processed")

                    # Step 2: Chunk text
                    st.write("✂️ Splitting into smart chunks…")
                    if doc.page_texts:
                        chunks = split_pages_into_chunks(
                            doc.page_texts, source_filename=doc.filename
                        )
                    else:
                        chunks = split_text_into_chunks(
                            doc.text, source_filename=doc.filename
                        )
                    st.write(f"✓ {len(chunks)} knowledge chunks created")

                    # Step 3: Load embedding model
                    st.write("🤖 Loading AI embedding model…")
                    model = load_embedding_model_cached()
                    st.session_state.embedding_model = model
                    st.write("✓ Embedding model ready")

                    # Step 4: Generate embeddings
                    st.write("🧠 Generating embeddings…")
                    texts = [c.text for c in chunks]
                    embeddings = embed_texts(texts, model=model)
                    st.write(f"✓ {len(embeddings)} embeddings generated")

                    # Step 5: Build FAISS index
                    st.write("🗄️ Building knowledge base…")
                    vs = VectorStore()
                    vs.build(chunks, embeddings)
                    st.write("✓ Knowledge base ready")

                    # Update session state
                    st.session_state.vector_store = vs
                    st.session_state.current_doc = doc
                    st.session_state.doc_processed = True
                    st.session_state._last_file_size = len(file_bytes)
                    st.session_state.chat_history = []
                    st.session_state.quiz_state = None
                    st.session_state.flashcard_deck = None

                    status.update(label="✅ SaathiAI is ready!", state="complete")

                except DocumentProcessingError as exc:
                    status.update(label="❌ Processing failed", state="error")
                    st.error(str(exc))
                    return

                except Exception as exc:
                    status.update(label="❌ Unexpected error", state="error")
                    st.error(f"An unexpected error occurred: {exc}")
                    return

            st.success(
                f"🎉 **{filename}** is ready! Head to any study mode using the sidebar."
            )
        else:
            st.info("✓ Document already processed and ready.")


# ── Home Page ─────────────────────────────────────────────────────────────────
def page_home():
    load_css()

    # Hero section
    col_hero, col_upload = st.columns([1.2, 0.8], gap="large")

    with col_hero:
        st.markdown(f"""
<div class="hero-title">🎓 {APP_NAME}</div>
<div class="hero-tagline">{APP_TAGLINE}</div>
<p style="color:#58634E; font-size:1.05rem; line-height:1.8; max-width:520px; font-weight:500;">
  Upload your study material and use open-source AI to understand difficult concepts,
  generate exam answers, test yourself, and prepare — even if your exam is tomorrow.
</p>
""", unsafe_allow_html=True)

        # Quick nav buttons
        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💬 Ask My Notes"):
                st.session_state.page = "chat"
                st.rerun()
        with c2:
            if st.button("📝 Exam Mode"):
                st.session_state.page = "exam"
                st.rerun()
        with c3:
            if st.button("🚨 Exam Tomorrow"):
                st.session_state.page = "exam_tomorrow"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_upload:
        render_upload_section()

    st.markdown("---")

    # Feature cards
    st.markdown("### ✨ What SaathiAI Can Do For You")
    features = [
        ("💬", "Ask My Notes", "Ask questions directly from your uploaded material. SaathiAI retrieves relevant sections and explains them clearly."),
        ("📝", "Exam Mode", "Generate structured exam answers for 2, 5, or 10 marks grounded strictly in your actual notes."),
        ("🎯", "Quiz Me", "Test your understanding with AI-generated MCQ quizzes with instant scoring and detailed explanations."),
        ("🧠", "Flashcards", "Create front/back flashcards for active recall and rapid memorization of key concepts."),
        ("⚡", "Quick Revision", "Get a structured revision cheat sheet for any topic: definitions, formulas, and practice questions."),
        ("🚨", "Exam Tomorrow", "Emergency revision package: high-yield topics, must-know formulas, and top predicted questions."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
<div class="feature-card">
  <span class="feature-icon">{icon}</span>
  <div class="feature-title">{title}</div>
  <div class="feature-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)
            if i < 3:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Open AI message
    st.markdown("""
<div style="background:#FFFFFF; border:1px solid #DCD4C7; border-left:4px solid #8B9A6E;
            border-radius:16px; padding:1.5rem; margin-top:1rem; box-shadow:0 2px 10px rgba(44,53,36,0.05);">
  <h4 class="casko-font" style="margin:0 0 0.5rem 0; color:#2C3524;">🔓 Grounded Open-Source Architecture</h4>
  <p style="color:#58634E; margin:0; line-height:1.75; font-size:0.95rem;">
    SaathiAI is built around Retrieval-Augmented Generation (RAG) and open-weight AI architectures.
    Your study material is vectorized locally into a FAISS index, ensuring accurate answers with exact page citations and zero hallucination.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Chat Page ─────────────────────────────────────────────────────────────────
def page_chat():
    load_css()
    st.markdown("## 💬 Ask My Notes")
    st.markdown(
        "<p style='color:#58634E;'>Ask any question from your uploaded study material.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first. Go to the **Home** page.")
        return

    from src.rag import rag_answer

    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message)
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(message["answer"])
                if message.get("sources"):
                    with st.expander("📚 Sources from your notes"):
                        for chunk in message["sources"]:
                            page_info = f" · Page {chunk.page_number}" if chunk.page_number else ""
                            st.markdown(f"""
<div class="source-chip">📄 {chunk.source_filename}{page_info}</div>
""", unsafe_allow_html=True)
                            st.markdown(f"> {chunk.text[:300]}{'…' if len(chunk.text) > 300 else ''}")

    # Input
    question = st.chat_input("Ask a question from your notes…")
    if question:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        st.session_state.chat_history.append(("user", question))

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Searching your notes and generating answer…"):
                result = rag_answer(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    embedding_model=st.session_state.embedding_model,
                )

            st.markdown(result.answer)

            if result.source_chunks:
                with st.expander("📚 Sources from your notes"):
                    for chunk in result.source_chunks:
                        page_info = f" · Page {chunk.page_number}" if chunk.page_number else ""
                        st.markdown(f"""
<div class="source-chip">📄 {chunk.source_filename}{page_info}</div>
""", unsafe_allow_html=True)
                        st.markdown(f"> {chunk.text[:300]}{'…' if len(chunk.text) > 300 else ''}")

        st.session_state.chat_history.append((
            "assistant",
            {"answer": result.answer, "sources": result.source_chunks},
        ))

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ── Exam Mode Page ────────────────────────────────────────────────────────────
def page_exam():
    load_css()
    st.markdown("## 📝 Exam Mode")
    st.markdown(
        "<p style='color:#58634E;'>Generate structured exam answers formatted for university marking schemes.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first.")
        return

    from src.exam_mode import generate_exam_answer

    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. OSI Model, Database Normalization, Supervised Learning…",
            key="exam_topic",
        )
    with col2:
        marks = st.selectbox("Marks Scheme", ["2", "5", "10"], index=1, key="exam_marks")

    generate_btn = st.button("📝 Generate Exam Answer", use_container_width=True)

    if generate_btn:
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Generating {marks}-mark answer for: {topic}…"):
                result = generate_exam_answer(
                    topic=topic,
                    marks=marks,
                    vector_store=st.session_state.vector_store,
                    embedding_model=st.session_state.embedding_model,
                )

            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown("### 📄 Generated Answer")
                st.markdown(f"**Topic:** `{topic}` | **Marks:** `{marks}`")
                st.markdown("---")
                st.markdown(f"""
<div class="answer-box">{result['answer']}</div>
""", unsafe_allow_html=True)

                if result["sources"]:
                    with st.expander("📚 Sources from your notes"):
                        for chunk in result["sources"]:
                            page_info = f" · Page {chunk.page_number}" if chunk.page_number else ""
                            st.markdown(f"**{chunk.source_filename}{page_info}**")
                            st.markdown(f"> {chunk.text[:250]}…")


# ── Quiz Page ─────────────────────────────────────────────────────────────────
def page_quiz():
    load_css()
    st.markdown("## 🎯 Quiz Me")
    st.markdown(
        "<p style='color:#58634E;'>Test your knowledge with AI-generated multiple-choice questions.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first.")
        return

    from src.quiz_generator import generate_quiz

    if st.session_state.quiz_state is None or not st.session_state.quiz_state.is_valid:
        col1, col2, col3 = st.columns(3)
        with col1:
            num_q = st.selectbox("Number of Questions", [5, 10, 15], index=0)
        with col2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        with col3:
            topic = st.text_input("Topic (optional)", placeholder="Leave blank for all topics")

        if st.button("🎯 Generate Quiz", use_container_width=True):
            with st.spinner("Generating quiz questions from notes…"):
                quiz = generate_quiz(
                    vector_store=st.session_state.vector_store,
                    num_questions=num_q,
                    difficulty=difficulty,
                    topic=topic,
                    embedding_model=st.session_state.embedding_model,
                )

            if not quiz.is_valid:
                st.error(f"❌ {quiz.error}")
            else:
                st.session_state.quiz_state = quiz
                st.session_state.quiz_submitted = False
                st.rerun()

    else:
        quiz = st.session_state.quiz_state
        submitted = st.session_state.get("quiz_submitted", False)

        if not submitted:
            st.markdown(f"### 📋 Quiz – {quiz.total} Questions")
            st.progress(0)

            with st.form("quiz_form"):
                answers = {}
                for i, q in enumerate(quiz.questions):
                    st.markdown(f"**Q{i+1}. {q.question}**")
                    opts = [f"{k}. {v}" for k, v in q.options.items()]
                    chosen = st.radio(
                        f"Answer for Q{i+1}",
                        opts,
                        key=f"q_{i}",
                        label_visibility="collapsed",
                    )
                    answers[i] = chosen[0] if chosen else ""
                    st.markdown("---")

                submitted_btn = st.form_submit_button(
                    "✅ Submit Answers", use_container_width=True
                )

            if submitted_btn:
                for i, q in enumerate(quiz.questions):
                    q.user_answer = answers.get(i, "")
                st.session_state.quiz_submitted = True
                st.rerun()

        else:
            correct, total = quiz.score()
            pct = round(correct / total * 100) if total > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{correct}/{total}")
            col2.metric("Percentage", f"{pct}%")
            col3.metric("Incorrect", f"{total - correct}")

            if pct >= 80:
                st.success("🎉 Excellent work! You've mastered this material.")
            elif pct >= 60:
                st.info("👍 Good effort! Review the questions below.")
            else:
                st.warning("📖 Keep studying! Review the explanations below.")

            st.markdown("---")
            st.markdown("### 📋 Detailed Review")

            for i, q in enumerate(quiz.questions):
                is_correct = q.is_correct
                icon = "✅" if is_correct else "❌"
                with st.expander(f"{icon} Q{i+1}. {q.question}", expanded=not is_correct):
                    for key, val in q.options.items():
                        if key == q.correct_answer:
                            st.markdown(f"**✓ {key}. {val}** ← Correct")
                        elif key == q.user_answer and not is_correct:
                            st.markdown(f"~~{key}. {val}~~ ← Your answer")
                        else:
                            st.markdown(f"{key}. {val}")
                    if q.explanation:
                        st.info(f"💡 {q.explanation}")

            if st.button("🔄 Try Another Quiz", use_container_width=True):
                st.session_state.quiz_state = None
                st.session_state.quiz_submitted = False
                st.rerun()


# ── Flashcards Page ───────────────────────────────────────────────────────────
def page_flashcards():
    load_css()
    st.markdown("## 🧠 Flashcards")
    st.markdown(
        "<p style='color:#58634E;'>Study key concepts with active recall flashcards.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first.")
        return

    from src.flashcards import generate_flashcards

    if st.session_state.flashcard_deck is None or not st.session_state.flashcard_deck.is_valid:
        col1, col2 = st.columns(2)
        with col1:
            num_cards = st.selectbox("Number of Flashcards", [5, 10, 15, 20], index=1)
        with col2:
            topic = st.text_input("Topic (optional)", placeholder="Leave blank for general flashcards")

        if st.button("🧠 Generate Flashcards", use_container_width=True):
            with st.spinner("Creating flashcards…"):
                deck = generate_flashcards(
                    vector_store=st.session_state.vector_store,
                    num_cards=num_cards,
                    topic=topic,
                    embedding_model=st.session_state.embedding_model,
                )

            if not deck.is_valid:
                st.error(f"❌ {deck.error}")
            else:
                st.session_state.flashcard_deck = deck
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_flipped = False
                st.rerun()
    else:
        deck = st.session_state.flashcard_deck
        idx = st.session_state.flashcard_index
        flipped = st.session_state.flashcard_flipped
        card = deck.cards[idx]

        st.markdown(f"**Card {idx + 1} of {deck.count}**")
        st.progress((idx + 1) / deck.count)

        if not flipped:
            st.markdown(f"""
<div style="background:#FFFFFF; border:2px solid #DCD4C7; border-radius:24px;
            padding:3.5rem 2rem; text-align:center; min-height:220px;
            display:flex; flex-direction:column; justify-content:center; align-items:center;
            box-shadow:0 8px 24px rgba(139,154,110,0.12);">
  <div class="casko-font" style="color:#828E75; font-size:0.9rem; margin-bottom:0.8rem; text-transform:uppercase;
              letter-spacing:0.08em; font-weight:700;">FRONT / QUESTION</div>
  <div style="font-size:1.45rem; font-weight:700; color:#2C3524; line-height:1.5;">
    {card.front}
  </div>
  <div style="color:#828E75; font-size:0.88rem; margin-top:1.8rem; font-weight:500;">
    Click "Flip Card" to reveal definition
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:#F8F4ED; border:2px solid #8B9A6E; border-radius:24px;
            padding:3.5rem 2rem; text-align:center; min-height:220px;
            display:flex; flex-direction:column; justify-content:center; align-items:center;
            box-shadow:0 8px 24px rgba(139,154,110,0.18);">
  <div class="casko-font" style="color:#8B9A6E; font-size:0.9rem; margin-bottom:0.8rem; text-transform:uppercase;
              letter-spacing:0.08em; font-weight:700;">BACK / EXPLANATION</div>
  <div style="font-size:1.2rem; color:#2C3524; line-height:1.85; max-width:640px; font-weight:500;">
    {card.back}
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='margin:1.2rem 0;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", disabled=idx == 0, use_container_width=True):
                st.session_state.flashcard_index = max(0, idx - 1)
                st.session_state.flashcard_flipped = False
                st.rerun()
        with col2:
            if st.button("🔄 Flip Card", use_container_width=True):
                st.session_state.flashcard_flipped = not flipped
                st.rerun()
        with col3:
            if st.button("➡️ Next", disabled=idx == deck.count - 1, use_container_width=True):
                st.session_state.flashcard_index = min(deck.count - 1, idx + 1)
                st.session_state.flashcard_flipped = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔄 Generate New Flashcards", use_container_width=False):
            st.session_state.flashcard_deck = None
            st.session_state.flashcard_index = 0
            st.session_state.flashcard_flipped = False
            st.rerun()


# ── Quick Revision Page ───────────────────────────────────────────────────────
def page_revision():
    load_css()
    st.markdown("## ⚡ Quick Revision")
    st.markdown(
        "<p style='color:#58634E;'>Get a rapid revision cheat sheet for any topic from your notes.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first.")
        return

    from src.revision import generate_quick_revision

    topic = st.text_input(
        "Topic to Revise",
        placeholder="e.g. TCP Handshake, Operating System Scheduling, Neural Networks…",
        key="revision_topic",
    )

    if st.button("⚡ Generate Revision Guide", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a topic to revise.")
        else:
            with st.spinner(f"Generating revision guide for: {topic}…"):
                result = generate_quick_revision(
                    topic=topic,
                    vector_store=st.session_state.vector_store,
                    embedding_model=st.session_state.embedding_model,
                )

            if not result.is_valid:
                st.error(result.error)
            else:
                st.markdown(f"# ⚡ Quick Revision: {topic}")
                st.markdown("---")
                st.markdown(result.content)


# ── Exam Tomorrow Page ────────────────────────────────────────────────────────
def page_exam_tomorrow():
    load_css()
    st.markdown("""
<div style="background:#FFFFFF; border:2px solid #8B9A6E; border-left:6px solid #8B9A6E;
            border-radius:20px; padding:1.8rem; margin-bottom:1.5rem; box-shadow:0 4px 16px rgba(139,154,110,0.12);">
  <h2 class="casko-font" style="margin:0; color:#2C3524;">🚨 Exam Tomorrow Mode</h2>
  <p style="color:#58634E; margin:0.5rem 0 0 0; font-size:1.02rem;">
    Emergency high-yield revision package synthesized directly from your study material.
  </p>
  <p style="color:#828E75; font-size:0.85rem; margin:0.4rem 0 0 0; font-style:italic;">
    ⚠️ AI-generated prioritization based on syllabus frequency.
  </p>
</div>
""", unsafe_allow_html=True)

    if not st.session_state.doc_processed:
        st.warning("📤 Please upload a study document first.")
        return

    from src.revision import generate_exam_tomorrow_plan

    if st.button(
        "🚨 Generate Emergency Revision Package",
        type="primary",
        use_container_width=True,
        key="exam_tomorrow_generate",
    ):
        with st.spinner(
            "Analyzing your notes and synthesizing your emergency exam package…"
        ):
            result = generate_exam_tomorrow_plan(
                vector_store=st.session_state.vector_store,
                embedding_model=st.session_state.embedding_model,
            )

        if not result.is_valid:
            st.error(f"❌ {result.error}")
            return

        if result.important_topics:
            st.markdown("## 🔥 Must-Study Topics")
            for t in result.important_topics:
                with st.container():
                    icon = ["🥇", "🥈", "🥉"] + ["🔥"] * 10
                    st.markdown(f"""
<div class="saathi-card">
  <span style="font-size:1.3rem;">{icon[t.rank - 1]}</span>
  <strong class="casko-font" style="font-size:1.2rem; margin-left:0.5rem; color:#2C3524;">{t.topic}</strong>
  {f'<br><span style="color:#58634E; font-size:0.92rem; margin-left:2.2rem;">{t.reason}</span>' if t.reason else ''}
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("# 🚨 Exam Tomorrow Revision Package")
        st.markdown(result.revision_content)


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    load_css()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        page_home()
    elif page == "chat":
        page_chat()
    elif page == "exam":
        page_exam()
    elif page == "quiz":
        page_quiz()
    elif page == "flashcards":
        page_flashcards()
    elif page == "revision":
        page_revision()
    elif page == "exam_tomorrow":
        page_exam_tomorrow()
    else:
        page_home()


if __name__ == "__main__":
    main()
