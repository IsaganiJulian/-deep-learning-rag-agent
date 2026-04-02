from __future__ import annotations

import html
import os
import re
import uuid
from pathlib import Path

import streamlit as st

# Load Streamlit secrets into environment variables before any other imports
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ[_key] = _value
except Exception:
    pass

from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager, get_shared_store

# Clear cached settings so it re-reads from environment on first call
get_settings.cache_clear()

# ------------------ Cached Resources ------------------ #


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the single shared in-memory vector store for this app instance."""
    return get_shared_store()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ------------------ Session State ------------------ #


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "chat_thread_id": str(uuid.uuid4()),
        "selected_document": None,
        "uploaded_names": [],
        "show_reply_toast": False,
        "prompt_queue": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------ UI helpers (presentation only) ------------------ #


def _simple_markdown_to_html(text: str) -> str:
    """Minimal inline formatting for chat bubbles (no extra deps)."""
    escaped = html.escape(text)
    escaped = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        escaped,
    )
    return escaped.replace("\n", "<br/>")


def _source_tags_for_display(source: str) -> list[tuple[str, str]]:
    """Build muted tag labels for expandable source rows."""
    s = str(source)
    tags: list[tuple[str, str]] = []
    lower = s.lower()
    if ".pdf" in lower or "pdf" in lower:
        tags.append(("type", "PDF"))
    elif ".md" in lower or "markdown" in lower:
        tags.append(("type", "MD"))
    else:
        tags.append(("type", "PDF"))
    page_m = re.search(r"page\s*[:#]?\s*(\d+)", s, re.I)
    if page_m:
        tags.append(("page", f"Page {page_m.group(1)}"))
    else:
        tags.append(("page", "Page —"))
    topic_m = re.search(r"topic\s*[:]\s*([^|\]]+)", s, re.I)
    if topic_m:
        tags.append(("topic", f"Topic: {topic_m.group(1).strip()}"))
    else:
        demo_m = re.search(r"SOURCE:\s*([^|]+)", s, re.I)
        label = demo_m.group(1).strip() if demo_m else "General"
        tags.append(("topic", f"Topic: {label}"))
    return tags


# ------------------ Global styling ------------------ #


def inject_global_css() -> None:
    _fonts = (
        "https://fonts.googleapis.com/css2?"
        "family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&"
        "family=Source+Sans+3:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap"
    )
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="FONT_HREF" rel="stylesheet" />
<style>
    :root {
        --rag-font-display: "Bricolage Grotesque", ui-sans-serif, system-ui, sans-serif;
        --rag-font-body: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
        --rag-accent-a: #f97316;
        --rag-accent-b: #14b8a6;
        --rag-accent-warm: #fbbf24;
        --rag-mid: #0d9488;
        --rag-surface: #fffefb;
        --rag-surface-elevated: #f7f4ee;
        --rag-muted: #57534e;
        --rag-border: #e7e5e4;
        --rag-user-bg: linear-gradient(135deg, #ea580c 0%, #f97316 45%, #fbbf24 100%);
        --rag-ai-bg: linear-gradient(165deg, #f4f4f5 0%, #e7e5e4 100%);
        --rag-card-shadow:
            0 1px 0 rgba(255, 255, 255, 0.65) inset,
            0 2px 4px rgba(28, 25, 23, 0.04),
            0 12px 36px rgba(28, 25, 23, 0.07);
        --rag-shadow-hover: 0 18px 48px rgba(234, 88, 12, 0.14);
    }
    /*
     * Streamlit scrolls an inner div, not .stApp — gradients on .stApp look fixed while
     * content moves. Keep .stApp flat; use .rag-hero-band for the dark header.
     */
    .stApp {
        font-family: var(--rag-font-body);
        background-color: #ece8e1;
        background-image:
            radial-gradient(ellipse 120% 80% at 100% 0%, rgba(20, 184, 166, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 80% 60% at 0% 100%, rgba(249, 115, 22, 0.06) 0%, transparent 45%),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 23px,
                rgba(120, 113, 108, 0.04) 23px,
                rgba(120, 113, 108, 0.04) 24px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 23px,
                rgba(120, 113, 108, 0.04) 23px,
                rgba(120, 113, 108, 0.04) 24px
            );
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    section.main {
        background-color: transparent;
        background-image: none;
    }
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }
    /* Workbench rows: extra space between columns; nested rows stay compact */
    section.main .stHorizontalBlock {
        gap: 1.75rem !important;
        align-items: stretch !important;
    }
    section.main .stHorizontalBlock .stHorizontalBlock {
        gap: 0.65rem !important;
    }
    /* Hero: scrolls with document (not painted on viewport-fixed .stApp) */
    .rag-hero-band {
        position: relative;
        overflow: hidden;
        background-color: #0c0a09;
        background-image:
            radial-gradient(ellipse 70% 55% at 15% 20%, rgba(249, 115, 22, 0.35) 0%, transparent 55%),
            radial-gradient(ellipse 60% 50% at 90% 10%, rgba(20, 184, 166, 0.28) 0%, transparent 50%),
            radial-gradient(ellipse 50% 40% at 70% 90%, rgba(251, 191, 36, 0.12) 0%, transparent 45%),
            linear-gradient(165deg, #0c0a09 0%, #1c1917 48%, #292524 100%);
        border-radius: 0 0 22px 22px;
        padding: 0.15rem 1.35rem 1.35rem;
        margin: -0.5rem -1rem 1.35rem;
        box-shadow:
            0 4px 0 rgba(20, 184, 166, 0.35),
            0 24px 56px rgba(12, 10, 9, 0.35);
    }
    .rag-hero-band::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
        opacity: 0.45;
        pointer-events: none;
    }
    .rag-hero-inner {
        position: relative;
        z-index: 1;
    }
    .rag-hero-blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(48px);
        opacity: 0.55;
        pointer-events: none;
        z-index: 0;
    }
    .rag-hero-blob-a {
        width: 180px;
        height: 180px;
        background: #f97316;
        top: -40px;
        right: 8%;
        animation: rag-blob-drift 14s ease-in-out infinite;
    }
    .rag-hero-blob-b {
        width: 140px;
        height: 140px;
        background: #14b8a6;
        bottom: -30px;
        left: 5%;
        animation: rag-blob-drift 18s ease-in-out infinite reverse;
    }
    @keyframes rag-blob-drift {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(12px, -8px) scale(1.05); }
    }
    @media (max-width: 768px) {
        .rag-hero-band {
            margin-left: 0;
            margin-right: 0;
            border-radius: 0;
        }
    }
    /* Gradient accent bar */
    .rag-header-accent {
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, #f97316, #fbbf24, #14b8a6, #0d9488);
        margin: 0 0 1.35rem 0;
        box-shadow: 0 0 28px rgba(249, 115, 22, 0.5);
    }
    .rag-hero-badge {
        display: inline-block;
        font-family: var(--rag-font-display);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #fde68a;
        background: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.4);
        padding: 0.38rem 0.85rem;
        border-radius: 6px;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .rag-hero-title {
        font-family: var(--rag-font-display);
        font-weight: 800;
        font-size: 2.4rem;
        letter-spacing: -0.03em;
        background: linear-gradient(118deg, #fffbeb 0%, #fed7aa 28%, #5eead4 72%, #ccfbf1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.35rem;
        line-height: 1.1;
        filter: drop-shadow(0 2px 24px rgba(249, 115, 22, 0.25));
    }
    .rag-hero-caption {
        font-family: var(--rag-font-body);
        font-size: 1.06rem;
        color: #d6d3d1;
        font-weight: 500;
        margin-bottom: 0;
        letter-spacing: 0.01em;
        max-width: 40rem;
        line-height: 1.55;
    }
    /* Main-area bordered panels (cards) */
    section.main [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        padding: 1.35rem 1.4rem !important;
        background: var(--rag-surface) !important;
        border: 1px solid rgba(120, 113, 108, 0.14) !important;
        box-shadow: var(--rag-card-shadow) !important;
        transition: box-shadow 0.28s ease, border-color 0.28s ease, transform 0.25s ease !important;
    }
    section.main [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.7) inset,
            0 20px 50px rgba(28, 25, 23, 0.09) !important;
        border-color: rgba(20, 184, 166, 0.28) !important;
    }
    /* Primary column: document (secondary weight) */
    section.main .stHorizontalBlock [data-testid="column"]:first-child
        [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(165deg, #fffefb 0%, #f5f0e8 100%) !important;
    }
    /* Chat column: stronger presence */
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        [data-testid="stVerticalBlockBorderWrapper"] {
        border-left: 4px solid #f97316 !important;
        box-shadow:
            -6px 0 24px rgba(20, 184, 166, 0.18),
            var(--rag-card-shadow) !important;
        background: linear-gradient(180deg, #fffefb 0%, #faf7f2 100%) !important;
    }
    .rag-card-title {
        font-family: var(--rag-font-display);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #78716c;
        margin: 0 0 0.85rem 0;
    }
    .rag-card-title.rag-card-title-emphasis {
        color: #44403c;
        font-size: 0.76rem;
    }
    .rag-card-sub {
        font-family: var(--rag-font-body);
        margin: -0.5rem 0 0.9rem 0;
        font-size: 0.9rem;
        color: #a8a29e;
        font-weight: 500;
    }
    .rag-section-divider {
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(249, 115, 22, 0.35) 22%,
            rgba(20, 184, 166, 0.35) 78%,
            transparent
        );
        margin: 1.35rem 0;
    }
    .rag-section-divider.rag-divider-tight {
        margin: 0.65rem 0 1rem 0;
    }
    .rag-section-divider.rag-divider-header {
        margin: 1.1rem 0 1.35rem 0;
    }
    /* Document column card: align vertical footprint with chat */
    section.main .stHorizontalBlock [data-testid="column"]:first-child
        [data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 560px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    /* Badges */
    .rag-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.22rem 0.58rem;
        border-radius: 6px;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .rag-badge-topic {
        background: rgba(249, 115, 22, 0.12);
        color: #c2410c;
        border: 1px solid rgba(249, 115, 22, 0.22);
    }
    .rag-badge-diff {
        background: rgba(20, 184, 166, 0.12);
        color: #0f766e;
        border: 1px solid rgba(20, 184, 166, 0.22);
    }
    .rag-badge-muted {
        background: #f5f5f4;
        color: #57534e;
        border: 1px solid #e7e5e4;
    }
    /* Empty states */
    .rag-empty {
        text-align: center;
        padding: 2rem 1.25rem 1.5rem;
        color: #78716c;
    }
    .rag-empty-icon-wrap {
        width: 4.35rem;
        height: 4.35rem;
        margin: 0 auto 1rem;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        line-height: 1;
        background: linear-gradient(145deg, #ffedd5 0%, #ccfbf1 100%);
        box-shadow:
            0 6px 20px rgba(234, 88, 12, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(20, 184, 166, 0.2);
        transform: rotate(-3deg);
    }
    .rag-empty-icon {
        font-size: 2.5rem;
        line-height: 1;
        margin-bottom: 0.6rem;
        filter: grayscale(0.15);
    }
    .rag-empty-title {
        font-family: var(--rag-font-display);
        font-weight: 700;
        color: #292524;
        margin-bottom: 0.45rem;
        font-size: 1.14rem;
        letter-spacing: -0.02em;
    }
    .rag-empty-hint {
        font-family: var(--rag-font-body);
        font-size: 0.91rem;
        line-height: 1.55;
        max-width: 24rem;
        margin: 0 auto;
        color: #78716c;
    }
    .rag-prompt-chips-label {
        font-family: var(--rag-font-body);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a8a29e;
        margin: 1.25rem 0 0.65rem;
    }
    /* Chat shell + scroll */
    .rag-chat-shell {
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    .rag-chat-scroll {
        scroll-behavior: smooth;
        overflow-y: auto !important;
    }
    @keyframes rag-msg-fade-in {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    /* Chat bubbles */
    .rag-chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 1.25rem;
        align-items: flex-end;
        gap: 0.65rem;
        animation: rag-msg-fade-in 0.42s ease-out forwards;
    }
    .rag-chat-row-user {
        justify-content: flex-end;
    }
    .rag-chat-row-assistant {
        justify-content: flex-start;
    }
    .rag-bubble {
        max-width: min(88%, 34rem);
        border-radius: 18px 18px 6px 18px;
        padding: 0.85rem 1.1rem;
        font-size: 0.95rem;
        line-height: 1.55;
        position: relative;
        box-shadow: 0 4px 16px rgba(28, 25, 23, 0.08);
    }
    .rag-chat-row-assistant .rag-bubble {
        border-radius: 18px 18px 18px 6px;
    }
    .rag-bubble-user {
        background: var(--rag-user-bg);
        color: #fffbeb;
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-left: auto;
        box-shadow:
            0 4px 20px rgba(234, 88, 12, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    .rag-bubble-user a {
        color: #fef3c7;
    }
    .rag-bubble-assistant {
        background: var(--rag-ai-bg);
        color: #292524;
        border: 1px solid rgba(120, 113, 108, 0.2);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .rag-bubble-inner {
        margin: 0;
    }
    .rag-bubble-inner p {
        margin: 0 0 0.4rem 0;
    }
    .rag-bubble-inner p:last-child {
        margin-bottom: 0;
    }
    .rag-avatar {
        font-size: 1.4rem;
        line-height: 1;
        flex-shrink: 0;
        user-select: none;
    }
    /* Source expanders */
    .rag-source-wrap {
        max-width: min(92%, 36rem);
        margin-top: 0.35rem;
        margin-bottom: 0.5rem;
    }
    .rag-source-wrap [data-testid="stExpander"] {
        border: 1px solid rgba(20, 184, 166, 0.25) !important;
        border-radius: 10px !important;
        background: linear-gradient(180deg, #fafaf9 0%, #f5f5f4 100%) !important;
    }
    .rag-source-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.5rem;
    }
    .rag-source-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid rgba(249, 115, 22, 0.2);
    }
    .rag-source-snippet {
        font-size: 0.82rem;
        color: #44403c;
        line-height: 1.45;
    }
    /* Sidebar cards */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(ellipse 100% 80% at 0% 0%, rgba(249, 115, 22, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse 70% 50% at 100% 100%, rgba(20, 184, 166, 0.15) 0%, transparent 50%),
            linear-gradient(198deg, #0c0a09 0%, #1c1917 45%, #292524 100%);
        border-right: 1px solid rgba(251, 191, 36, 0.15);
        box-shadow: 6px 0 32px rgba(12, 10, 9, 0.45);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] [data-testid="stCaption"] {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        padding: 1.15rem 1.2rem !important;
        background: rgba(41, 37, 36, 0.55) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(214, 211, 209, 0.12) !important;
        box-shadow:
            0 4px 24px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: box-shadow 0.28s ease, border-color 0.28s ease !important;
    }
    section[data-testid="stSidebar"]
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(20, 184, 166, 0.35) !important;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(249, 115, 22, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }
    .rag-sidebar-title {
        font-family: var(--rag-font-display);
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.25rem 0;
        color: #f8fafc;
    }
    .rag-sidebar-sub {
        font-family: var(--rag-font-body);
        font-size: 0.84rem;
        color: #94a3b8;
        margin: 0 0 1rem 0;
        line-height: 1.45;
    }
    .rag-sidebar-eyebrow {
        font-family: var(--rag-font-display);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #5eead4;
        margin: 0 0 0.5rem 0;
    }
    /* Callout: draws the eye to Browse files + drop zone */
    .rag-sidebar-callout {
        font-family: var(--rag-font-body);
        font-size: 0.82rem;
        line-height: 1.45;
        color: #e7e5e4;
        background: linear-gradient(
            125deg,
            rgba(249, 115, 22, 0.18) 0%,
            rgba(20, 184, 166, 0.12) 100%
        );
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 10px;
        padding: 0.65rem 0.75rem 0.7rem;
        margin: 0 0 0.85rem 0;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .rag-sidebar-callout-icon {
        display: inline-block;
        margin-right: 0.35rem;
    }
    .rag-sidebar-callout strong {
        color: #fef3c7;
        font-weight: 700;
    }
    .rag-file-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 0.45rem;
        font-size: 0.84rem;
        color: #e2e8f0;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .rag-file-row:hover {
        border-color: rgba(20, 184, 166, 0.45);
        box-shadow: 0 2px 14px rgba(20, 184, 166, 0.12);
    }
    .rag-ready-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-family: var(--rag-font-body);
        font-size: 0.75rem;
        font-weight: 600;
        color: #a7f3d0;
        background: rgba(20, 184, 166, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.35);
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        margin-top: 0.55rem;
    }
    /* Chat column: flex card so input sits at bottom */
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        [data-testid="stVerticalBlockBorderWrapper"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 560px !important;
    }
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        [data-testid="stVerticalBlockBorderWrapper"] > div {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    /* Chat input — short hint above field (no duplicate send control) */
    .rag-chat-send-hint {
        font-family: var(--rag-font-body);
        font-size: 0.8rem;
        color: #78716c;
        margin: 0.35rem 0 0.5rem 0;
        line-height: 1.45;
    }
    .rag-chat-send-hint strong {
        color: #44403c;
    }
    /* Chat input — sticky at bottom of card + glow on focus */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #d6d3d1 !important;
        box-shadow:
            0 2px 0 rgba(255, 255, 255, 0.85) inset,
            0 6px 24px rgba(28, 25, 23, 0.07) !important;
        background: #ffffff !important;
        margin-top: auto !important;
        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #14b8a6 !important;
        box-shadow:
            0 0 0 4px rgba(20, 184, 166, 0.2),
            0 0 0 1px rgba(249, 115, 22, 0.25),
            0 10px 32px rgba(249, 115, 22, 0.12) !important;
    }
    /* Send button — larger, high-contrast “primary” control */
    [data-testid="stChatInput"] button {
        border-radius: 999px !important;
        min-width: 3.15rem !important;
        min-height: 3.15rem !important;
        background: linear-gradient(
            145deg,
            #ea580c 0%,
            #f97316 38%,
            #14b8a6 100%
        ) !important;
        color: #fff !important;
        border: 2px solid rgba(254, 243, 199, 0.85) !important;
        box-shadow:
            0 0 0 3px rgba(249, 115, 22, 0.35),
            0 8px 22px rgba(234, 88, 12, 0.45) !important;
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            filter 0.18s ease !important;
    }
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.07);
        filter: brightness(1.05);
        box-shadow:
            0 0 0 4px rgba(20, 184, 166, 0.35),
            0 12px 28px rgba(234, 88, 12, 0.5) !important;
    }
    [data-testid="stChatInput"] button:active {
        transform: scale(0.97);
    }
    [data-testid="stChatInput"] button svg {
        display: none !important;
    }
    [data-testid="stChatInput"] button::after {
        content: "➤";
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1;
    }
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.35rem !important;
    }
    /* Primary buttons — darker gradient on hover */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            var(--rag-accent-a),
            var(--rag-accent-b)
        ) !important;
        border: none !important;
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            filter 0.18s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(0.95);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.35) !important;
    }
    .stButton > button {
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease,
            filter 0.18s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.18) !important;
    }
    section.main [data-testid="stVerticalBlock"] > div[style*="height"] {
        scroll-behavior: smooth;
    }
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        [data-testid="stExpander"] {
        margin-left: 2.35rem;
        max-width: min(92%, 36rem);
    }
    /* App footer */
    .rag-app-footer {
        font-family: var(--rag-font-body);
        text-align: center;
        font-size: 0.78rem;
        color: #78716c;
        letter-spacing: 0.08em;
        margin-top: 2.25rem;
        padding-top: 1.35rem;
        border-top: 1px dashed rgba(120, 113, 108, 0.28);
    }
    .rag-app-footer strong {
        color: #44403c;
        font-weight: 600;
    }
    /* Streamlit widgets in dark sidebar */
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] textarea {
        background: rgba(15, 23, 42, 0.55) !important;
        color: #f1f5f9 !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border-radius: 12px !important;
        padding: 0.15rem 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: rgba(28, 25, 23, 0.55) !important;
        border: 2px dashed rgba(94, 234, 212, 0.45) !important;
        border-radius: 12px !important;
        padding: 1rem 0.85rem 1.05rem !important;
        box-shadow:
            0 0 0 1px rgba(249, 115, 22, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] span {
        color: #d6d3d1 !important;
    }
    /* Highlight “Browse files” — primary control in the drop zone */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
        font-family: var(--rag-font-body) !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.1rem !important;
        color: #0c0a09 !important;
        background: linear-gradient(180deg, #fef3c7 0%, #fbbf24 45%, #f59e0b 100%) !important;
        border: 2px solid rgba(254, 243, 199, 0.95) !important;
        box-shadow:
            0 0 0 2px rgba(249, 115, 22, 0.45),
            0 6px 20px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.15s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
        transform: translateY(-1px) scale(1.02);
        filter: brightness(1.05);
        box-shadow:
            0 0 0 3px rgba(20, 184, 166, 0.45),
            0 10px 26px rgba(249, 115, 22, 0.35) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:active,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:active {
        transform: translateY(0) scale(0.99);
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .rag-empty-title {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .rag-empty-hint {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] .rag-empty-icon-wrap {
        background: linear-gradient(
            145deg,
            rgba(249, 115, 22, 0.22) 0%,
            rgba(20, 184, 166, 0.18) 100%
        );
        border-color: rgba(251, 191, 36, 0.25);
    }
    section[data-testid="stSidebar"] .stSuccess,
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
    }
    section[data-testid="stSidebar"] .stSuccess p,
    section[data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #a7f3d0 !important;
    }
    /* Quick start pills (buttons in chat column when empty) */
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        .stButton > button {
        font-family: var(--rag-font-body) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        border: 1px solid rgba(20, 184, 166, 0.35) !important;
        background: linear-gradient(180deg, #fffefb, #f5f0e8) !important;
        color: #c2410c !important;
        padding: 0.45rem 0.9rem !important;
        transition:
            transform 0.15s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease !important;
    }
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        .stButton > button:hover {
        border-color: rgba(249, 115, 22, 0.45) !important;
        box-shadow: 0 4px 16px rgba(249, 115, 22, 0.12) !important;
        transform: translateY(-1px);
    }
    /* Clear chat (tertiary) — compact, distinct from quick-start pills */
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        .stButton > button[kind="tertiary"] {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        border-radius: 8px !important;
        padding: 0.4rem 0.55rem !important;
        color: #78716c !important;
        background: rgba(255, 255, 255, 0.65) !important;
        border: 1px dashed rgba(120, 113, 108, 0.45) !important;
    }
    section.main .stHorizontalBlock [data-testid="column"]:nth-child(2)
        .stButton > button[kind="tertiary"]:hover {
        color: #b45309 !important;
        border-color: rgba(249, 115, 22, 0.55) !important;
        background: #fffefb !important;
    }
    /* Success toast — match warm lab palette */
    section.main .stSuccess {
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.12), rgba(251, 191, 36, 0.1)) !important;
        border: 1px solid rgba(20, 184, 166, 0.35) !important;
        border-radius: 10px !important;
    }
    section.main .stSuccess p {
        color: #0f766e !important;
        font-weight: 500 !important;
    }
</style>
""".replace("FONT_HREF", _fonts),
        unsafe_allow_html=True,
    )


# ------------------ Sidebar ------------------ #


def render_ingestion_panel():
    st.sidebar.markdown(
        '<p class="rag-sidebar-title">Corpus ingestion</p>'
        '<p class="rag-sidebar-sub">Build your interview knowledge base — upload once, '
        "retrieve everywhere.</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<p class="rag-sidebar-eyebrow">Study materials</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="rag-sidebar-callout"><span class="rag-sidebar-callout-icon" '
        'aria-hidden="true">📂</span>'
        "Add files: tap <strong>Browse files</strong> in the box below, or drag "
        "PDFs / Markdown into the drop zone.</div>",
        unsafe_allow_html=True,
    )
    with st.sidebar.container(border=True):
        uploaded_files = st.sidebar.file_uploader(
            "PDF or Markdown files",
            type=["pdf", "md"],
            accept_multiple_files=True,
            help="Up to 200MB per session · PDF and Markdown supported",
        )

        st.sidebar.caption("Drag in multiple files · PDF / Markdown")

        if uploaded_files:
            names = [f.name for f in uploaded_files]
            st.session_state.uploaded_names = names

            st.sidebar.markdown("**Files selected**")
            for name in names:
                ext = Path(name).suffix.lower()
                icon = "📕" if ext == ".pdf" else "📝"
                safe = html.escape(name)
                row = (
                    f'<div class="rag-file-row">{icon}'
                    '<span style="flex:1;overflow:hidden;'
                    'text-overflow:ellipsis;white-space:nowrap;">'
                    f"{safe}</span></div>"
                )
                st.sidebar.markdown(row, unsafe_allow_html=True)

            st.sidebar.markdown(
                '<div class="rag-ready-pill"><span>●</span> Ready to ingest</div>',
                unsafe_allow_html=True,
            )
            st.sidebar.success(f"{len(uploaded_files)} file(s) staged for ingestion")

            if st.sidebar.button(
                "Ingest files",
                key="rag_ingest_files",
                type="primary",
                use_container_width=True,
            ):
                with st.sidebar:
                    with st.spinner("Chunking and indexing files..."):
                        try:
                            settings = get_settings()
                            corpus_dir = Path(settings.corpus_dir)
                            corpus_dir.mkdir(parents=True, exist_ok=True)

                            saved_paths: list[Path] = []
                            for uploaded in uploaded_files:
                                target = corpus_dir / uploaded.name
                                target.write_bytes(uploaded.getvalue())
                                saved_paths.append(target)

                            chunker = get_chunker()
                            vector_store = get_vector_store()
                            chunks = chunker.chunk_files(saved_paths)
                            if not chunks:
                                st.error("No chunks were produced from the uploaded files.")
                            else:
                                result = vector_store.ingest(chunks)
                                st.success(
                                    f"Ingested: {result.ingested}, "
                                    f"skipped duplicates: {result.skipped}, "
                                    f"errors: {len(result.errors)}"
                                )
                                if result.errors:
                                    for err in result.errors[:3]:
                                        st.caption(f"- {err}")

                                docs = vector_store.list_documents()
                                st.session_state.uploaded_names = [d["source"] for d in docs]
                        except Exception as e:
                            st.error(f"Ingestion failed: {e}")
        else:
            existing = get_vector_store().list_documents()
            st.session_state.uploaded_names = [d["source"] for d in existing]
            st.sidebar.markdown(
                """
<div class="rag-empty" style="padding:1.1rem 0.35rem 0.85rem;">
  <div class="rag-empty-icon-wrap" aria-hidden="true">📚</div>
  <div class="rag-empty-title">No documents yet</div>
  <div class="rag-empty-hint">Drop PDFs or Markdown here to ground answers in your own
  notes.</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.sidebar.markdown(
        '<div class="rag-section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Documents power retrieval for grounded answers.")


# ------------------ Document Viewer ------------------ #


def render_document_viewer():
    st.markdown(
        '<p class="rag-card-title">Document viewer</p>'
        '<p class="rag-card-sub">Preview and context for what the model can cite</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rag-section-divider rag-divider-tight"></div>',
        unsafe_allow_html=True,
    )

    names = st.session_state.get("uploaded_names") or []

    if not names:
        with st.container(border=True):
            st.markdown(
                """
<div class="rag-empty">
  <div class="rag-empty-icon-wrap" aria-hidden="true">📄</div>
  <div class="rag-empty-title">Your knowledge base is empty</div>
  <div class="rag-empty-hint">Use the sidebar to upload PDFs or Markdown. Titles and
  snippets appear here after ingestion.</div>
</div>
""",
                unsafe_allow_html=True,
            )
        return

    selected = st.selectbox(
        "Document",
        options=names,
        key="selected_document",
        label_visibility="collapsed",
    ) or names[0]

    with st.container(border=True):
        st.markdown(
            f'<p style="margin:0 0 0.85rem 0;font-weight:700;color:#292524;'
            f'font-size:1.02rem;letter-spacing:-0.02em;">{html.escape(selected)}</p>',
            unsafe_allow_html=True,
        )

        chunks = get_vector_store().get_document_chunks(selected)

        topic = chunks[0].metadata.topic if chunks else "Deep learning"
        difficulty = chunks[0].metadata.difficulty if chunks else "Mixed"
        st.markdown(
            f'<span class="rag-badge rag-badge-topic">Topic · {html.escape(topic)}</span>'
            f'<span class="rag-badge rag-badge-diff">Difficulty · {html.escape(difficulty)}</span>'
            f'<span class="rag-badge rag-badge-muted">Chunks · {len(chunks)}</span>',
            unsafe_allow_html=True,
        )

        doc_scroll = st.container(height=280)
        with doc_scroll:
            st.markdown(
                f'<p style="font-size:0.9rem;color:#94a3b8;">'
                f'{len(chunks)} chunk(s) indexed and ready for retrieval.</p>',
                unsafe_allow_html=True,
            )


# ------------------ Chat Interface ------------------ #


def _render_chat_bubbles():
    for message in st.session_state.chat_history:
        role = message["role"]
        content_html = _simple_markdown_to_html(message["content"])
        if role == "user":
            st.markdown(
                '<div class="rag-chat-row rag-chat-row-user">'
                '<div class="rag-bubble rag-bubble-user">'
                '<div class="rag-bubble-inner">'
                f"{content_html}"
                "</div></div>"
                '<span class="rag-avatar" aria-hidden="true">🧑‍💻</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="rag-chat-row rag-chat-row-assistant">'
                '<span class="rag-avatar" aria-hidden="true">🤖</span>'
                '<div class="rag-bubble rag-bubble-assistant">'
                '<div class="rag-bubble-inner">'
                f"{content_html}"
                "</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if message.get("sources"):
                for idx, source in enumerate(message["sources"], start=1):
                    tags = _source_tags_for_display(source)
                    tag_html = "".join(
                        f'<span class="rag-source-tag">[{html.escape(t[1])}]</span>'
                        for t in tags
                    )
                    title = f"📎 Source {idx} · referenced passage"
                    with st.expander(title, expanded=False):
                        st.markdown(
                            f'<div class="rag-source-tags">{tag_html}</div>',
                            unsafe_allow_html=True,
                        )
                        snippet = html.escape(str(source))
                        st.markdown(
                            f'<p class="rag-source-snippet">{snippet}</p>',
                            unsafe_allow_html=True,
                        )


def render_chat_interface(graph):
    if st.session_state.pop("show_reply_toast", False):
        st.success("Response ready — expand **Sources** on the assistant message.")

    with st.container(border=True):
        _chat_head_l, _chat_head_r = st.columns([5, 1])
        with _chat_head_l:
            st.markdown(
                '<p class="rag-card-title rag-card-title-emphasis">Interview prep chat</p>'
                '<div class="rag-section-divider rag-divider-tight"></div>'
                '<p class="rag-card-sub">Grounded answers with traceable sources</p>',
                unsafe_allow_html=True,
            )
        with _chat_head_r:
            if st.session_state.chat_history:
                if st.button(
                    "Clear",
                    key="rag_clear_chat",
                    help="Clear this conversation",
                    type="tertiary",
                    use_container_width=True,
                ):
                    st.session_state.chat_history = []
                    st.session_state.chat_thread_id = str(uuid.uuid4())
                    st.rerun()

        chat_container = st.container(height=440)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    """
<div class="rag-empty" style="padding:2.25rem 1rem 1.5rem;">
  <div class="rag-empty-icon-wrap" aria-hidden="true">✨</div>
  <div class="rag-empty-title">Start your AI interview practice</div>
  <div class="rag-empty-hint">Ask about deep learning interviews — or tap quick start
  below.</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                _render_chat_bubbles()

        if not st.session_state.chat_history:
            st.markdown(
                '<p class="rag-prompt-chips-label">Quick start</p>',
                unsafe_allow_html=True,
            )
            q1, q2, q3 = st.columns(3)
            quick_prompts = (
                "Explain CNNs in simple terms",
                "What is attention in Transformers?",
                "How does backpropagation work?",
            )
            cols = (q1, q2, q3)
            for col, label, idx in zip(cols, quick_prompts, range(3)):
                with col:
                    if st.button(
                        label,
                        key=f"quick_prompt_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.prompt_queue = label
                        st.rerun()

        st.markdown(
            '<p class="rag-chat-send-hint">Type below, then press <strong>Enter</strong> '
            "to send (or use the send button on the right).</p>",
            unsafe_allow_html=True,
        )
        query = st.chat_input(
            "Ask about CNNs, Transformers, or interview questions...",
        )
        if not query and st.session_state.get("prompt_queue"):
            query = st.session_state.pop("prompt_queue")

        if query:
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            with st.spinner("🤖 Thinking..."):
                try:
                    config = {
                        "configurable": {
                            "thread_id": st.session_state.chat_thread_id,
                        }
                    }
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=query)]},
                        config=config,
                    )
                    final = result.get("final_response")
                    if final is None:
                        response_text = (
                            "No response was produced. Check the agent pipeline and logs."
                        )
                        sources: list[str] = []
                    else:
                        response_text = final.answer
                        sources = list(final.sources)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                        }
                    )

                    st.session_state.show_reply_toast = True
                    st.rerun()

                except Exception as e:
                    st.session_state.chat_history.pop()
                    st.error(f"Error: {e}")


# ------------------ Main ------------------ #


def main():
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_global_css()

    st.markdown(
        f"""
<div class="rag-hero-band">
  <div class="rag-hero-blob rag-hero-blob-a" aria-hidden="true"></div>
  <div class="rag-hero-blob rag-hero-blob-b" aria-hidden="true"></div>
  <div class="rag-hero-inner">
    <div class="rag-header-accent"></div>
    <p class="rag-hero-badge">Interview prep · RAG</p>
    <p class="rag-hero-title">🧠 {html.escape(settings.app_title)}</p>
    <p class="rag-hero-caption">Traceable citations and crisp answers — focused practice,
    not generic chat.</p>
    <div class="rag-section-divider rag-divider-header"></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    initialise_session_state()

    vs = get_vector_store()
    get_chunker()

    # Seed uploaded_names from ChromaDB on first load so existing corpus is visible
    if not st.session_state.uploaded_names:
        existing_docs = vs.list_documents()
        if existing_docs:
            st.session_state.uploaded_names = [d["source"] for d in existing_docs]
    graph = get_graph()

    render_ingestion_panel()

    st.markdown(
        '<div class="rag-section-divider"></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([5, 7], gap="large")

    with col1:
        render_document_viewer()

    with col2:
        render_chat_interface(graph)

    st.markdown(
        '<p class="rag-app-footer">Built with <strong>LangChain</strong> · '
        "<strong>Streamlit</strong></p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
