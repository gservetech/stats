import streamlit as st

def apply_custom_styles():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main { background: #1a1d21; }
        .stApp { background: linear-gradient(180deg, #1a1d21 0%, #0d0f11 100%); }
        .header {
            background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
            padding: 1.5rem 2rem;
            margin: -1rem -1rem 1.5rem -1rem;
            border-radius: 0;
        }
        .header h1 { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
        .header p { color: rgba(255,255,255,0.85); margin-top: 0.3rem; font-size: 0.9rem; }

        .status-ok {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 0.4rem 0.8rem;
            background: rgba(0, 215, 117, 0.15);
            border: 1px solid #00d775;
            border-radius: 15px;
            color: #00d775;
            font-weight: 600;
            font-size: 0.8rem;
        }
        .status-error {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 0.4rem 0.8rem;
            background: rgba(255, 71, 87, 0.15);
            border: 1px solid #ff4757;
            border-radius: 15px;
            color: #ff4757;
            font-weight: 600;
            font-size: 0.8rem;
        }

        .stButton > button {
            background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e2328 0%, #15181c 100%);
            border-right: 1px solid #3d4450;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

    /* ---- Responsive typography (mobile + desktop) ---- */
    html, body, [class*="css"]  {
      font-size: 16px;
    }

    /* Make headers/labels easier to read */
    h1, h2, h3, h4 { letter-spacing: 0.2px; }

    /* Desktop: bump overall size */
    @media (min-width: 992px) {
      html, body, [class*="css"] { font-size: 18px; }
      .header h1 { font-size: 2.2rem !important; }
      .header p  { font-size: 1.05rem !important; }
    }

    /* Mobile: keep compact, prevent overflow */
    @media (max-width: 600px) {
      .header { padding: 1rem 1rem !important; }
      .header h1 { font-size: 1.5rem !important; }
      .header p { font-size: 0.9rem !important; }
      .stButton > button { padding: 0.6rem 0.8rem !important; }
    }

    /* Bigger metric cards readability */
    div[data-testid="stMetric"] > div {
      padding: 10px 12px;
    }
    div[data-testid="stMetricLabel"] p {
      font-size: 0.95rem !important;
    }
    div[data-testid="stMetricValue"] {
      font-size: 1.6rem !important;
    }

    /* Dataframes: increase font on desktop, allow horizontal scroll on mobile */
    div[data-testid="stDataFrame"] { border-radius: 10px; }
    @media (min-width: 992px) {
      div[data-testid="stDataFrame"] * { font-size: 0.95rem !important; }
    }
    @media (max-width: 600px) {
      div[data-testid="stDataFrame"] { overflow-x: auto; }
    }


        /* 🚀 FORCE FULL WIDTH ON DESKTOP (override Streamlit default max-width) */
        [data-testid="stAppViewContainer"] .main .block-container,
        section.main > div.block-container,
        .block-container {
            max-width: 100% !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }

        /* Make charts \u0026 dataframes stretch to container */
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stDataFrame"] > div {
            width: 100% !important;
        }

        /* Desktop font sizing (fix \"too small\" look) */
        @media (min-width: 1200px) {
            html, body, [class*=\"css\"] { font-size: 18px !important; }
        }

        /* Mobile: tighter padding + slightly smaller font */
        @media (max-width: 768px) {
            [data-testid="stAppViewContainer"] .main .block-container,
            section.main > div.block-container,
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            html, body, [class*=\"css\"] { font-size: 15px !important; }
        }
    </style>
    """,
        unsafe_allow_html=True
    )
