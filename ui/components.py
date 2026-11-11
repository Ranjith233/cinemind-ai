"""
CineMind AI - Premium Cinematic UI Styling & Components
Injects custom CSS for deep charcoal/navy glassmorphic theme, elegant typography,
and movie studio aesthetics.
"""

import streamlit as st


def apply_cinematic_styles() -> None:
    """Injects high-end cinema styling into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Root color palette */
        :root {
            --bg-deep: #090c13;
            --bg-card: rgba(18, 24, 38, 0.75);
            --bg-card-hover: rgba(26, 36, 56, 0.85);
            --border-glass: rgba(212, 175, 55, 0.15);
            --border-glass-bright: rgba(212, 175, 55, 0.4);
            --gold-accent: #d4af37;
            --gold-glow: rgba(212, 175, 55, 0.25);
            --amber-accent: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        /* App Background */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #151b2a 0%, #090c13 75%, #05070a 100%) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }

        /* Header typography */
        h1, h2, h3 {
            font-family: 'Cinzel', serif !important;
            letter-spacing: 0.05em;
            color: #ffffff;
        }

        /* Cinema Brand Header Banner */
        .cinemind-header {
            background: linear-gradient(135deg, rgba(20, 26, 42, 0.85) 0%, rgba(12, 16, 26, 0.95) 100%);
            border: 1px solid var(--border-glass);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .cinemind-logo-title {
            font-family: 'Cinzel', serif;
            font-size: 28px;
            font-weight: 900;
            letter-spacing: 0.12em;
            background: linear-gradient(90deg, #ffffff 0%, var(--gold-accent) 60%, var(--amber-accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .cinemind-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: var(--text-secondary);
            letter-spacing: 0.04em;
            margin-top: 4px;
        }

        /* Glassmorphic Cards */
        .cinema-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(10px);
            transition: all 0.25s ease;
            margin-bottom: 16px;
        }

        .cinema-card:hover {
            border-color: var(--border-glass-bright);
            box-shadow: 0 6px 24px var(--gold-glow);
            transform: translateY(-2px);
        }

        /* Citation Badge */
        .citation-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(212, 175, 55, 0.12);
            border: 1px solid rgba(212, 175, 55, 0.3);
            color: #fde68a;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
            margin-right: 6px;
            margin-bottom: 6px;
            font-family: 'JetBrains Mono', monospace;
        }

        .chapter-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            font-size: 11px;
            font-weight: 500;
            padding: 2px 8px;
            border-radius: 6px;
            margin-right: 6px;
        }

        .confidence-high {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #6ee7b7;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
        }

        .confidence-medium {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fcd34d;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
        }

        .confidence-low {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
        }

        /* Chat container overrides */
        .stChatMessage {
            background: rgba(15, 20, 32, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #090c14 !important;
            border-right: 1px solid rgba(212, 175, 55, 0.1) !important;
        }

        /* Button styling */
        .stButton button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.2s ease !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            padding: 10px 20px;
            color: var(--text-secondary);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(212, 175, 55, 0.1) !important;
            color: #fde68a !important;
            border-bottom: 2px solid var(--gold-accent) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header(title: str = "CINEMIND AI", subtitle: str = "Advanced Cinema Intelligence & Film Research RAG") -> None:
    """Renders the top branding bar."""
    st.markdown(
        f"""
        <div class="cinemind-header">
            <div>
                <h1 class="cinemind-logo-title">
                    <span>🎬</span> {title}
                </h1>
                <div class="cinemind-subtitle">{subtitle}</div>
            </div>
            <div style="text-align: right;">
                <span class="citation-badge">v2.5 ARCHITECTURE</span>
                <span class="chapter-badge">HYBRID RRF + CROSS-ENCODER</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
