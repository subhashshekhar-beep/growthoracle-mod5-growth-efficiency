# Streamlit app — GrowthOracle (Module 5 only)
# Growth Efficiency — Resources → Outcomes (Standalone Project)
# ------------------------------------------------------------
# How to run:
#   pip install streamlit pandas numpy plotly pyyaml
#   streamlit run app.py

import os, sys, json, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import to_html
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    to_html = None
    _HAS_PLOTLY = False

# ---- Page ----
st.set_page_config(
    page_title="GrowthOracle — Module 5: Growth Efficiency",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀",
)
st.title("GrowthOracle — Module 5: Growth Efficiency — Resources → Outcomes")
st.caption("Find under‑invested winners and over‑invested laggards with quadrant analysis & action lists")

# ---- Logger ----
@st.cache_resource
def get_logger(level=logging.INFO):
    logger = logging.getLogger("growthoracle_mod5")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("\n%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# ---- Defaults / Config ----
_DEFAULT_CONFIG = {
    "performance": {"sample_row_limit": 350_000, "seed": 42},
    "defaults": {"date_lookback_days": 60}
}

@st.cache_resource
def load_config():
    cfg = _DEFAULT_CONFIG.copy()
    if yaml is not None:
        for candidate in ["config.yaml", "growthoracle.yaml", "settings.yaml"]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        user_cfg = yaml.safe_load(f) or {}
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
                    logger.info(f"Loaded configuration from {candidate}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {candidate}: {e}")
    return cfg

CONFIG = load_config()

# ---- Validation Core (minimal) ----
@dataclass
class ValidationMessage:
    category: str  # "Critical" | "Warning" | "Info"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    def __init__(self):
        self.messages: List[ValidationMessage] = []

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages:
            return pd.DataFrame(columns=["category", "code", "message", "context"])
        return pd.DataFrame([{
            "category": m.category,
            "code": m.code,
            "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])

# ---- Helpers ----
def download_df_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.warning(f"No data to download for {label}")
        return
    try:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def export_plot_html(fig, name: str):
    if to_html is None or fig is None:
        st.info("Plotly HTML export not available.")
        return
    try:
        html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label=f"Export {name} (HTML)",
            data=html_str.encode("utf-8"),
            file_name=f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Failed to export plot: {e}")

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

# ---- CSV Readers ----
def read_csv_safely(upload, name: str, vc: ValidationCollector) -> Optional[pd.DataFrame]:
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided"); return None
    try_encodings = [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc) if enc else pd.read_csv(upload)
            if df.empty or df.shape[1] == 0:
                vc.add("Critical", "EMPTY_CSV", f"{name} appears empty"); return None
            return df
        except Exception as e:
            last_err = e
            continue
    vc.add("Critical", "CSV_ENCODING", f"Failed to read {name}", last_error=str(last_err))
    return None

# ---- Mapping ----
def _guess_colmap(prod_df, ga4_df, gsc_df):
    if prod_df is None or gsc_df is None:
        return {}, {}, {}
    prod_map = {
        "msid": "Msid" if "Msid" in prod_df.columns else next((c for c in prod_df.columns if c.lower()=="msid"), None),
        "title": "Title" if "Title" in prod_df.columns else next((c for c in prod_df.columns if "title" in c.lower()), None),
        "path": "Path" if "Path" in prod_df.columns else next((c for c in prod_df.columns if "path" in c.lower()), None),
        "publish": "Publish Time" if "Publish Time" in prod_df.columns else next((c for c in prod_df.columns if "publish" in c.lower()), None),
    }
    ga4_map = {}
    if ga4_df is not None and not ga4_df.empty:
        ga4_map = {
            "msid": "customEvent:msid" if "customEvent:msid" in ga4_df.columns else next((c for c in ga4_df.columns if "msid" in c.lower()), None),
            "date": "date" if "date" in ga4_df.columns else next((c for c in ga4_df.columns if c.lower()=="date"), None),
            "pageviews": "screenPageViews" if "screenPageViews" in ga4_df.columns else next((c for c in ga4_df.columns if "pageview" in c.lower()), None),
            "users": "totalUsers" if "totalUsers" in ga4_df.columns else next((c for c in ga4_df.columns if "users" in c.lower()), None),
            "engagement": "userEngagementDuration" if "userEngagementDuration" in ga4_df.columns else next((c for c in ga4_df.columns if "engagement" in c.lower()), None),
            "bounce": "bounceRate" if "bounceRate" in ga4_df.columns else next((c for c in ga4_df.columns if "bounce" in c.lower()), None),
        }
    gsc_map = {
        "date": "Date" if "Date" in gsc_df.columns else next((c for c in gsc_df.columns if c.lower()=="date"), None),
        "page": "Page" if "Page" in gsc_df.columns else next((c for c in gsc_df.columns if "page" in c.lower()), None),
        "query": "Query" if "Query" in gsc_df.columns else next((c for c in gsc_df.columns if "query" in c.lower()), None),
        "clicks": "Clicks" if "Clicks" in gsc_df.columns else next((c for c in gsc_df.columns if "clicks" in c.lower()), None),
        "impr": "Impressions" if "Impressions" in gsc_df.columns else next((c for c in gsc_df.columns if "impr" in c.lower()), None),
        "ctr": "CTR" if "CTR" in gsc_df.columns else next((c for c in gsc_df.columns if "ctr" in c.lower()), None),
        "pos": "Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
    }
    return prod_map, ga4_map, gsc_map

# ---- Standardization & Merge ----
def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    p = prod_df.copy() if prod_df is not None else None
    if p is not None and mappings["prod"].get("publish") and mappings["prod"]["publish"] in p.columns:
        try:
            p["Publish Time"] = pd.to_datetime(p[mappings["prod"]["publish"]], errors="coerce", utc=True)
        except Exception:
            pass

    g4 = ga4_df.copy() if ga4_df is not None else None
    if g4 is not None and mappings["ga4"].get("date") and mappings["ga4"]["date"] in g4.columns:
        g4["date"] = pd.to_datetime(g4[mappings["ga4"]["date"]], errors="coerce").dt.date

    gs = gsc_df.copy() if gsc_df is not None else None
    if gs is not None and mappings["gsc"].get("date") and mappings["gsc"]["date"] in gs.columns:
        gs["date"] = pd.to_datetime(gs[mappings["gsc"]["date"]], errors="coerce").dt.date

    return p, g4, gs


def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # Rename to standard names
    std_names = {
        "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "Publish Time"},
        "ga4": {"msid": "msid", "date": "date", "pageviews": "screenPageViews", "users": "totalUsers", "engagement": "userEngagementDuration", "bounce": "bounceRate"},
        "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks", "impr": "Impressions", "ctr": "CTR", "pos": "Position"}
    }
    try:
        if prod_df is not None: prod_df.rename(columns={prod_map.get(k): v for k, v in std_names["prod"].items() if prod_map.get(k)}, inplace=True)
        if ga4_df is not None: ga4_df.rename(columns={ga4_map.get(k): v for k, v in std_names["ga4"].items() if ga4_map.get(k)}, inplace=True)
        if gsc_df is not None: gsc_df.rename(columns={gsc_map.get(k): v for k, v in std_names["gsc"].items() if gsc_map.get(k)}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Column renaming failed: {e}")
        return None, vc

    # Dates
    prod_df, ga4_df, gsc_df = standardize_dates_early(prod_df, ga4_df, gsc_df, {"prod": std_names["prod"], "ga4": std_names["ga4"], "gsc": std_names["gsc"]}, vc)

    # MSID cleanup
    for df, name in [(prod_df, "Production"), (ga4_df, "GA4")]:
        if df is not None and "msid" in df.columns:
            df["msid"] = pd.to_numeric(df["msid"], errors="coerce")
            df.dropna(subset=["msid"], inplace=True)
            if not df.empty: df["msid"] = df["msid"].astype("int64")

    if gsc_df is not None and "page_url" in gsc_df.columns:
        gsc_df["msid"] = gsc_df["page_url"].astype(str).str.extract(r'(\d+)\.cms').iloc[:, 0]
        gsc_df["msid"] = pd.to_numeric(gsc_df["msid"], errors="coerce")
        gsc_df.dropna(subset=["msid"], inplace=True)
        if not gsc_df.empty: gsc_df["msid"] = gsc_df["msid"].astype("int64")

        # Numeric conversions & clamps
        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 100))]:
            if col in gsc_df.columns:
                gsc_df[col] = pd.to_numeric(gsc_df[col], errors="coerce")
                gsc_df[col] = gsc_df[col].clip(lower=clamp[0]) if clamp[1] is None else gsc_df[col].clip(lower=clamp[0], upper=clamp[1])

        # CTR cleanup: accept %, decimals, or compute
        if "CTR" in gsc_df.columns:
            if gsc_df["CTR"].dtype == "object":
                tmp = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False).str.replace(",", "").str.strip()
                gsc_df["CTR"] = pd.to_numeric(tmp, errors="coerce") / 100.0
            gsc_df["CTR"] = gsc_df["CTR"].clip(lower=0, upper=1)
        elif {"Clicks","Impressions"}.issubset(gsc_df.columns):
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)

    # Merge GSC × Prod (GA4 optional)
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_PREP_FAIL", "Missing GSC or Production data"); return None, vc

    prod_cols = [c for c in ["msid","Title","Path","Publish Time"] if c in prod_df.columns]
    master = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how="left")

    # Enrich with categories
    if "Path" in master.columns:
        cats = master["Path"].astype(str).strip('/').split('/') if isinstance(master["Path"], str) else None
    if "Path" in master.columns:
        cats = master["Path"].astype(str).str.strip('/').str.split('/', n=2, expand=True)
        master["L1_Category"] = cats[0].fillna("Uncategorized")
        master["L2_Category"] = cats[1].fillna("General")
    else:
        master["L1_Category"] = "Uncategorized"
        master["L2_Category"] = "General"

    # Attach GA4 daily metrics if present
    if ga4_df is not None and not ga4_df.empty and "date" in ga4_df.columns:
        numeric_cols = [c for c in ["screenPageViews","totalUsers","userEngagementDuration","bounceRate"] if c in ga4_df.columns]
        if numeric_cols:
            ga4_daily = ga4_df.groupby(["msid","date"], as_index=False)[numeric_cols].sum(min_count=1)
            master = pd.merge(master, ga4_daily, on=["msid","date"], how="left")

    master["_lineage"] = "GSC→PROD→(GA4)"
    return master, vc

# ---- Efficiency math ----
def compute_category_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inputs (articles) and outputs and compute efficiency per article."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    # Ensure numeric
    for col in ["totalUsers","screenPageViews","Clicks","Impressions","userEngagementDuration","bounceRate","Position"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    g = d.groupby(["L1_Category","L2_Category"]).agg(
        total_articles = pd.NamedAgg(column="msid", aggfunc=lambda s: pd.Series(s).nunique()),
        total_users    = pd.NamedAgg(column="totalUsers", aggfunc="sum") if "totalUsers" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_pvs      = pd.NamedAgg(column="screenPageViews", aggfunc="sum") if "screenPageViews" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_clicks   = pd.NamedAgg(column="Clicks", aggfunc="sum") if "Clicks" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_impr     = pd.NamedAgg(column="Impressions", aggfunc="sum") if "Impressions" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_eng_s      = pd.NamedAgg(column="userEngagementDuration", aggfunc="mean") if "userEngagementDuration" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_bounce     = pd.NamedAgg(column="bounceRate", aggfunc="mean") if "bounceRate" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_position   = pd.NamedAgg(column="Position", aggfunc="mean") if "Position" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan)
    ).reset_index()

    # Per-article efficiency (guard against div/0)
    g["users_per_article"] = g["total_users"]  / g["total_articles"].replace(0, np.nan)
    g["pvs_per_article"]   = g["total_pvs"]    / g["total_articles"].replace(0, np.nan)
    g["clicks_per_article"] = g["total_clicks"] / g["total_articles"].replace(0, np.nan)
    g["impr_per_article"]   = g["total_impr"]   / g["total_articles"].replace(0, np.nan)

    # Tidy NA
    g = g.replace([np.inf, -np.inf], np.nan)
    return g


def plot_efficiency_quadrant(cat_df: pd.DataFrame, outcome: str, y_mode: str = "Total") -> None:
    """
    outcome ∈ {"total_users","total_pvs","total_clicks","total_impr"}
    y_mode: "Total" or "Per Article"
    """
    if not _HAS_PLOTLY:
        st.info("Plotly required for the quadrant chart."); return
    if cat_df is None or cat_df.empty or "total_articles" not in cat_df.columns:
        st.info("No category efficiency data to plot."); return
    if outcome not in cat_df.columns and f"{outcome.split('_',1)[1]}_per_article" not in cat_df.columns:
        st.info("Selected outcome not available."); return

    df = cat_df.copy()
    x = "total_articles"
    if y_mode == "Total":
        y = outcome
        y_label = outcome.replace("_"," ").title()
    else:
        per_map = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }
        y = per_map.get(outcome)
        y_label = y.replace("_"," ").title() if y else "Per Article"

    df = df.dropna(subset=[x, y])
    if df.empty:
        st.info("Nothing to display after cleaning."); return

    # Medians for quadrant split
    x_med = df[x].median()
    y_med = df[y].median()

    fig = px.scatter(
        df, x=x, y=y, color="L1_Category", size="total_clicks" if "total_clicks" in df.columns else None,
        hover_data=["L1_Category","L2_Category","total_articles","total_users","total_pvs","users_per_article","pvs_per_article"],
        title=f"Resources → Outcomes Quadrant ({'Total' if y_mode=='Total' else 'Efficiency'})",
        labels={x: "Total Articles", y: y_label}
    )

    # Add median lines
    fig.add_hline(y=y_med, line_dash="dash", opacity=0.4)
    fig.add_vline(x=x_med, line_dash="dash", opacity=0.4)

    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    export_plot_html(fig, f"quadrant_{outcome}_{'total' if y_mode=='Total' else 'per_article'}")


def opportunity_lists(cat_df: pd.DataFrame, outcome: str, y_mode: str = "Total") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (under_invested, over_invested) tables with potential gain/excess vs median production."""
    if cat_df is None or cat_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = cat_df.copy()
    x = "total_articles"
    if y_mode == "Total":
        y = outcome
        y_per_article_col = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }[outcome]
    else:
        per_map = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }
        y = per_map[outcome]
        y_per_article_col = y

    df = df.dropna(subset=[x, y, y_per_article_col])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x_med = df[x].median()
    y_med = df[y].median()

    # Tags
    df["tag"] = np.where((df[x] < x_med) & (df[y] >= y_med), "Under-invested",
                   np.where((df[x] >= x_med) & (df[y] < y_med), "Over-invested", "Other"))

    # Potential: move each category to median production at current efficiency
    df["delta_articles_to_median"] = (x_med - df[x]).clip(lower=0)  # only for under-invested
    df["potential_gain"] = (df["delta_articles_to_median"] * df[y_per_article_col]).round(0)

    under = (df[df["tag"] == "Under-invested"]
             .sort_values(["potential_gain", y_per_article_col], ascending=False)
             [["L1_Category","L2_Category","total_articles",y,y_per_article_col,"delta_articles_to_median","potential_gain"]]
             .rename(columns={y: ("Outcome (Y)" if y_mode=="Total" else y.replace('_',' ').title()),
                              y_per_article_col: "Outcome per Article"}))

    over = (df[df["tag"] == "Over-invested"]
            .assign(excess_articles_vs_median=(df[x] - x_med).clip(lower=0))
            .sort_values(["excess_articles_vs_median", x], ascending=False)
            [["L1_Category","L2_Category","total_articles","excess_articles_vs_median", y_per_article_col]]
            .rename(columns={y_per_article_col: "Outcome per Article"}))

    return under, over

# ---- Sidebar: Filters & Settings ----
with st.sidebar:
    st.subheader("Settings")
    outcome_choice = st.selectbox(
        "Outcome metric",
        ["total_users","total_pvs","total_clicks","total_impr"],
        format_func=lambda x: {
            "total_users":"Users",
            "total_pvs":"Pageviews",
            "total_clicks":"GSC Clicks",
            "total_impr":"GSC Impressions"
        }[x]
    )
    y_mode = st.radio("Y-axis", ["Total","Per Article"], index=0, horizontal=True)

    st.markdown("---")
    st.subheader("Analysis Period")
    end = date.today()
    start = end - timedelta(days=CONFIG["defaults"]["date_lookback_days"])
    start_date = st.date_input("Start Date", value=start)
    end_date = st.date_input("End Date", value=end)
    if start_date > end_date:
        st.warning("Start date is after end date. Swapping.")
        start_date, end_date = end_date, start_date

    st.markdown("---")
    st.subheader("Logging")
    log_level = st.selectbox("Log Level", ["DEBUG","INFO","WARNING","ERROR"], index=1)
    get_logger(getattr(logging, log_level))

# ---- Stepper ----
st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", [
    "1) Get CSV Templates",
    "2) Upload & Map Columns",
    "3) Validate & Process",
    "4) Analyze (Module 5)"
], horizontal=True)

# Templates

def _make_template_production():
    return pd.DataFrame({
        "Msid": [101, 102, 103],
        "Title": ["Budget 2025 highlights explained", "IPL 2025 schedule & squads", "Monsoon updates: city-by-city guide"],
        "Path": ["/business/budget-2025/highlights", "/sports/cricket/ipl-2025/schedule", "/news/monsoon/guide"],
        "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00", "2025-09-01 07:00:00"]
    })

def _make_template_ga4():
    return pd.DataFrame({
        "customEvent:msid": [101, 101, 102, 102, 103],
        "date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "screenPageViews": [5000, 6000, 15000, 12000, 7000],
        "totalUsers": [4000, 4500, 10000, 8000, 5200],
        "userEngagementDuration": [52.3, 48.2, 41.0, 44.7, 63.1],
        "bounceRate": [0.42, 0.45, 0.51, 0.49, 0.38]
    })

def _make_template_gsc():
    return pd.DataFrame({
        "Date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "Page": [
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/news/monsoon/guide/103.cms"
        ],
        "Query": ["budget 2025", "budget highlights", "ipl 2025 schedule", "ipl squads", "monsoon city guide"],
        "Clicks": [200, 240, 1200, 1100, 300],
        "Impressions": [5000, 5500, 40000, 38000, 7000],
        "CTR": [0.04, 0.0436, 0.03, 0.0289, 0.04286],
        "Position": [8.2, 8.0, 12.3, 11.7, 9.1]
    })

if step == "1) Get CSV Templates":
    st.info("Download sample CSV templates to understand required structure.")
    colA, colB, colC = st.columns(3)
    with colA:
        df = _make_template_production(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_production.csv", "Download Production Template")
    with colB:
        df = _make_template_ga4(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_ga4.csv", "Download GA4 Template")
    with colC:
        df = _make_template_gsc(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_gsc.csv", "Download GSC Template")
    st.stop()

# Step 2: uploads + mapping
st.subheader("Upload Your Data Files")
col1, col2, col3 = st.columns(3)
with col1:
    prod_file = st.file_uploader("Production Data (CSV)", type=["csv"], key="prod_csv")
    if prod_file: st.success(f"✓ Production: {prod_file.name}")
with col2:
    ga4_file = st.file_uploader("GA4 Data (CSV) — optional", type=["csv"], key="ga4_csv")
    if ga4_file: st.success(f"✓ GA4: {ga4_file.name}")
with col3:
    gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")
    if gsc_file: st.success(f"✓ GSC: {gsc_file.name}")

if not all([prod_file, gsc_file]):
    st.warning("Please upload Production & GSC files to proceed"); st.stop()

vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read) if ga4_file else None
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)

if any(df is None or df.empty for df in [prod_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Column mapping UI
st.subheader("Column Mapping")
prod_guess, ga4_guess, gsc_guess = _guess_colmap(prod_df_raw, ga4_df_raw if ga4_df_raw is not None else pd.DataFrame(), gsc_df_raw)

with st.expander("Production Mapping", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    prod_map = {}
    prod_map["msid"] = c1.selectbox("MSID", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("msid")) if prod_guess.get("msid") in prod_df_raw.columns else 0)
    prod_map["title"] = c2.selectbox("Title", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("title")) if prod_guess.get("title") in prod_df_raw.columns else 0)
    prod_map["path"] = c3.selectbox("Path", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("path")) if prod_guess.get("path") in prod_df_raw.columns else 0)
    prod_map["publish"] = c4.selectbox("Publish Time", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("publish")) if prod_guess.get("publish") in prod_df_raw.columns else 0)

with st.expander("GA4 Mapping (optional)", expanded=False):
    if ga4_df_raw is not None:
        c1, c2, c3, c4 = st.columns(4)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox("MSID (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("msid")) if ga4_guess.get("msid") in ga4_df_raw.columns else 0)
        ga4_map["date"] = c2.selectbox("Date (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("date")) if ga4_guess.get("date") in ga4_df_raw.columns else 0)
        ga4_map["pageviews"] = c3.selectbox("Pageviews", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("pageviews")) if ga4_guess.get("pageviews") in ga4_df_raw.columns else 0)
        ga4_map["users"] = c4.selectbox("Users", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("users")) if ga4_guess.get("users") in ga4_df_raw.columns else 0)
    else:
        ga4_map = {}
        st.info("GA4 optional — adds users/pageviews to the aggregates when available.")

with st.expander("GSC Mapping", expanded=True):
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    gsc_map = {}
    gsc_map["date"] = c1.selectbox("Date (GSC)", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("date")) if gsc_guess.get("date") in gsc_df_raw.columns else 0)
    gsc_map["page"] = c2.selectbox("Page URL", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("page")) if gsc_guess.get("page") in gsc_df_raw.columns else 0)
    gsc_map["query"] = c3.selectbox("Query", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("query")) if gsc_guess.get("query") in gsc_df_raw.columns else 0)
    gsc_map["clicks"] = c4.selectbox("Clicks", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("clicks")) if gsc_guess.get("clicks") in gsc_df_raw.columns else 0)
    gsc_map["impr"] = c5.selectbox("Impressions", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("impr")) if gsc_guess.get("impr") in gsc_df_raw.columns else 0)
    gsc_map["ctr"] = c6.selectbox("CTR", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("ctr")) if gsc_guess.get("ctr") in gsc_df_raw.columns else 0)
    gsc_map["pos"] = c7.selectbox("Position", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("pos")) if gsc_guess.get("pos") in gsc_df_raw.columns else 0)

# Process & merge
with st.spinner("Processing & merging datasets..."):
    master_df, vc_after = process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)

if master_df is None or master_df.empty:
    st.error("Data processing failed critically. Please check mappings and file contents.")
    st.dataframe(vc_after.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Date filter (if present)
if "date" in master_df.columns:
    m = master_df.copy()
    try:
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.date
        mask = (m["date"] >= start_date) & (m["date"] <= end_date)
        filtered_df = m[mask].copy()
        st.info(f"Date filter applied: {len(filtered_df):,} rows from {start_date} to {end_date}")
    except Exception:
        filtered_df = master_df
else:
    filtered_df = master_df

st.success(f"✅ Master dataset ready: {filtered_df.shape[0]:,} rows × {filtered_df.shape[1]} columns")

if step != "4) Analyze (Module 5)":
    st.info("Move to **Step 4** to run the Growth Efficiency analysis.")
    st.stop()

# -----------------------------
# ANALYSIS — Module 5 outputs
# -----------------------------
st.header("🚀 Module 5: Growth Efficiency — Resources → Outcomes")

cat_eff = compute_category_efficiency(filtered_df)
if cat_eff is None or cat_eff.empty:
    st.info("No category efficiency data available.")
    st.stop()

# Quadrant chart
plot_efficiency_quadrant(cat_eff, outcome_choice, y_mode=y_mode)

# Leaderboard
with st.expander("Efficiency Table (downloadable)", expanded=False):
    show_cols = ["L1_Category","L2_Category","total_articles",
                 "total_users","total_pvs","total_clicks","total_impr",
                 "users_per_article","pvs_per_article","clicks_per_article","impr_per_article",
                 "avg_eng_s","avg_bounce","avg_position"]
    show_cols = [c for c in show_cols if c in cat_eff.columns]
    st.dataframe(cat_eff[show_cols].sort_values("pvs_per_article" if "pvs_per_article" in cat_eff.columns else "users_per_article", ascending=False),
                 use_container_width=True, hide_index=True)
    download_df_button(cat_eff[show_cols], f"growth_efficiency_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                       "Download Growth Efficiency (CSV)")

# Action lists
under, over = opportunity_lists(cat_eff, outcome_choice, y_mode=y_mode)
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 🚀 Under‑invested Winners (scale production)")
    if not under.empty:
        st.dataframe(under.head(15), use_container_width=True, hide_index=True)
        download_df_button(under, f"under_invested_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           "Download Under‑invested (CSV)")
    else:
        st.info("None detected at current thresholds.")
with c2:
    st.markdown("### 🧰 Over‑invested Laggards (fix or reduce)")
    if not over.empty:
        st.dataframe(over.head(15), use_container_width=True, hide_index=True)
        download_df_button(over, f"over_invested_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           "Download Over‑invested (CSV)")
    else:
        st.info("None detected at current thresholds.")

st.markdown("---")
st.caption("GrowthOracle — Module 5 (Standalone)")
