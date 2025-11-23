import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="NYC LL97 Fines Explorer",
    layout="wide"
)

# Path to your CLEANED file
CSV_PATH = "LL97_cleaned_no_outliers.csv"

type_col = "Primary Property Type - Self Selected"
gfa_col = "Property GFA - Self-Reported (ft²)"
year_built_col = "Year Built"
fine_2024_col = "Fine_2024_$"
fine_2030_col = "Fine_2030_$"
borough_col = "Borough"

# -----------------------------
# ZIP → BOROUGH MAPPING
# -----------------------------
zip_to_borough = {
    # Manhattan
    **{str(z): "Manhattan" for z in range(10001, 10293)},

    # Bronx
    **{str(z): "Bronx" for z in list(range(10451, 10476)) + [10499]},

    # Brooklyn
    **{str(z): "Brooklyn" for z in range(11201, 11257)},

    # Queens
    **{str(z): "Queens" for z in list(range(11004, 11110)) + list(range(11351, 11698))},

    # Staten Island
    **{str(z): "Staten Island" for z in range(10301, 10315)},
}


def find_borough(zipcode):
    zipcode = str(zipcode).strip()
    zipcode = zipcode[:5]  # first 5 digits
    return zip_to_borough.get(zipcode, "Unknown")


# -----------------------------
# DATA LOADING & PREP
# -----------------------------
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, low_memory=False)


@st.cache_data
def prepare_viz_data(df: pd.DataFrame) -> pd.DataFrame:
    df_viz = df.copy()

    # Fine flags
    df_viz["Fine2024_flag"] = df_viz[fine_2024_col] > 0
    df_viz["Fine2030_flag"] = df_viz[fine_2030_col] > 0

    # ---------- GFA buckets: 0–100k, 100–200k, ..., 1400–1500k, 1500k+ ----------
    bin_size = 100_000
    max_edge = 1_500_000
    gfa_bins = list(range(0, max_edge + bin_size, bin_size)) + [np.inf]

    gfa_labels = []
    for i in range(len(gfa_bins) - 1):
        start = gfa_bins[i]
        end = gfa_bins[i + 1]
        if np.isinf(end):
            label = "1500k+"
        else:
            start_k = int(start // 1000)
            end_k = int(end // 1000)
            if start_k == 0:
                label = f"0–{end_k}k"
            else:
                label = f"{start_k}k–{end_k}k"
        gfa_labels.append(label)

    df_viz["GFA_100k_bucket"] = pd.cut(
        df_viz[gfa_col],
        bins=gfa_bins,
        labels=gfa_labels,
        include_lowest=True
    )

    # ---------- Building age & 10-year buckets (as of 2024) ----------
    reference_year = 2024
    df_viz["Building_Age"] = reference_year - df_viz[year_built_col]
    df_viz.loc[df_viz["Building_Age"] < 0, "Building_Age"] = np.nan

    max_age = df_viz["Building_Age"].max()
    if np.isfinite(max_age) and max_age > 0:
        upper_age = int(np.ceil(max_age / 10) * 10)
    else:
        upper_age = 150

    age_bins = np.arange(0, upper_age + 10, 10)
    age_labels = [f"{age_bins[i]}–{age_bins[i+1]} yrs"
                  for i in range(len(age_bins) - 1)]

    df_viz["Age_10yr_bucket"] = pd.cut(
        df_viz["Building_Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )

    return df_viz


@st.cache_data
def build_summary(df_viz: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated summary by type, GFA bucket, and age bucket.
    """
    group_cols = [type_col, "GFA_100k_bucket", "Age_10yr_bucket"]

    summary = (
        df_viz
        .groupby(group_cols, dropna=False)
        .agg(
            n_buildings=("Property ID", "count"),
            n_paying_2024=("Fine2024_flag", "sum"),
            n_paying_2030=("Fine2030_flag", "sum"),
            fines_2024_total=(fine_2024_col, "sum"),
            fines_2030_total=(fine_2030_col, "sum"),
        )
        .reset_index()
    )

    return summary


def add_pct_and_avg(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """
    Add % paying and average fines per paying building.
    Drop categories with no fines at all.
    Round everything to whole numbers (no decimals).
    """
    df = df.copy()

    # Percentages
    df["pct_2024"] = np.where(
        df["n_buildings"] > 0,
        df["n_paying_2024"] / df["n_buildings"] * 100,
        np.nan,
    )
    df["pct_2030"] = np.where(
        df["n_buildings"] > 0,
        df["n_paying_2030"] / df["n_buildings"] * 100,
        np.nan,
    )

    # Average fine per paying building
    df["avg_fine_2024"] = np.where(
        df["n_paying_2024"] > 0,
        df["fines_2024_total"] / df["n_paying_2024"],
        np.nan,
    )
    df["avg_fine_2030"] = np.where(
        df["n_paying_2030"] > 0,
        df["fines_2030_total"] / df["n_paying_2030"],
        np.nan,
    )

    # Only categories where any fines are paid
    df = df[(df["n_paying_2024"] + df["n_paying_2030"]) > 0]

    # Round everything that makes sense to whole numbers
    int_cols = [
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "pct_2024", "pct_2030",
        "avg_fine_2024", "avg_fine_2030",
    ]
    for c in int_cols:
        df[c] = df[c].fillna(0).round(0).astype(int)

    df[group_label] = df[group_label].astype(str)
    return df


# -----------------------------
# LOAD DATA
# -----------------------------
st.title("NYC LL97 Fines Explorer")

try:
    df_raw = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Create Borough column from Postal Code
if "Postal Code" in df_raw.columns:
    df_raw[borough_col] = df_raw["Postal Code"].apply(find_borough)
else:
    st.warning("Postal Code column not found; Borough set to 'Unknown' for all rows.")
    df_raw[borough_col] = "Unknown"

df_viz = prepare_viz_data(df_raw)
summary = build_summary(df_viz)

# -----------------------------
# SUMMARY TABLES
# -----------------------------
summary_type = (
    summary
    .groupby(type_col, dropna=False)
    .agg(
        n_buildings=("n_buildings", "sum"),
        n_paying_2024=("n_paying_2024", "sum"),
        n_paying_2030=("n_paying_2030", "sum"),
        fines_2024_total=("fines_2024_total", "sum"),
        fines_2030_total=("fines_2030_total", "sum"),
    )
    .reset_index()
)
summary_type = add_pct_and_avg(summary_type, type_col)

summary_gfa = (
    summary
    .groupby("GFA_100k_bucket", dropna=False)
    .agg(
        n_buildings=("n_buildings", "sum"),
        n_paying_2024=("n_paying_2024", "sum"),
        n_paying_2030=("n_paying_2030", "sum"),
        fines_2024_total=("fines_2024_total", "sum"),
        fines_2030_total=("fines_2030_total", "sum"),
    )
    .reset_index()
)
summary_gfa = summary_gfa.dropna(subset=["GFA_100k_bucket"])
summary_gfa = add_pct_and_avg(summary_gfa, "GFA_100k_bucket")

summary_age = (
    summary
    .groupby("Age_10yr_bucket", dropna=False)
    .agg(
        n_buildings=("n_buildings", "sum"),
        n_paying_2024=("n_paying_2024", "sum"),
        n_paying_2030=("n_paying_2030", "sum"),
        fines_2024_total=("fines_2024_total", "sum"),
        fines_2030_total=("fines_2030_total", "sum"),
    )
    .reset_index()
)
summary_age = summary_age.dropna(subset=["Age_10yr_bucket"])
summary_age = add_pct_and_avg(summary_age, "Age_10yr_bucket")

# Summary by Borough
summary_borough = (
    df_viz
    .groupby(borough_col, dropna=False)
    .agg(
        n_buildings=("Property ID", "count"),
        n_paying_2024=("Fine2024_flag", "sum"),
        n_paying_2030=("Fine2030_flag", "sum"),
        fines_2024_total=(fine_2024_col, "sum"),
        fines_2030_total=(fine_2030_col, "sum"),
    )
    .reset_index()
)

# Drop Unknown from borough chart (optional)
summary_borough = summary_borough[summary_borough[borough_col] != "Unknown"]
summary_borough = add_pct_and_avg(summary_borough, borough_col)

# -----------------------------
# KPI CARDS (entire dataset)
# -----------------------------
total_buildings = len(df_viz)
n_paying_2024 = int(df_viz["Fine2024_flag"].sum())
n_paying_2030 = int(df_viz["Fine2030_flag"].sum())

pct_paying_2024 = (n_paying_2024 / total_buildings * 100) if total_buildings > 0 else 0
pct_paying_2030 = (n_paying_2030 / total_buildings * 100) if total_buildings > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total buildings", f"{total_buildings:,}")
col2.metric("Buildings paying in 2024", f"{n_paying_2024:,}")
col3.metric("% of buildings paying in 2024", f"{pct_paying_2024:.0f}%")
col4.metric("% of buildings paying in 2030", f"{pct_paying_2030:.0f}%")

st.markdown("---")

year_label_map = {"pct_2024": "2024", "pct_2030": "2030"}


def _choose_avg(row):
    if row["Fine Year"] == "2024":
        return row["avg_fine_2024"]
    else:
        return row["avg_fine_2030"]


# -----------------------------
# BAR: % PAYING BY TYPE
# -----------------------------
st.subheader("Share of buildings paying fines by type (only types with fines)")

long_type = summary_type.melt(
    id_vars=[type_col, "n_buildings", "n_paying_2024", "n_paying_2030",
             "fines_2024_total", "fines_2030_total",
             "avg_fine_2024", "avg_fine_2030"],
    value_vars=["pct_2024", "pct_2030"],
    var_name="Fine Year",
    value_name="Percent Paying",
)

long_type["Fine Year"] = long_type["Fine Year"].map(year_label_map)
long_type["Avg Fine (per paying bldg)"] = long_type.apply(_choose_avg, axis=1)

fig_type = px.bar(
    long_type,
    x=type_col,
    y="Percent Paying",
    color="Fine Year",
    barmode="group",
    custom_data=[
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "Avg Fine (per paying bldg)",
    ],
)

fig_type.update_traces(
    texttemplate="%{y:.0f}%",
    textposition="outside",
)

fig_type.update_traces(
    hovertemplate=(
        "%{x}<br>"
        "Year: %{marker.color}<br>"
        "Percent paying: %{y:.0f}%<br>"
        "Buildings: %{customdata[0]:,}<br>"
        "Paying 2024: %{customdata[1]:,}<br>"
        "Paying 2030: %{customdata[2]:,}<br>"
        "Total fines 2024: $%{customdata[3]:,.0f}<br>"
        "Total fines 2030: $%{customdata[4]:,.0f}<br>"
        "Avg fine (per paying bldg): $%{customdata[5]:,.0f}"
        "<extra></extra>"
    )
)

fig_type.update_layout(
    xaxis_title="Building type",
    yaxis_title="% of buildings paying fines",
    yaxis_range=[0, 100],
    xaxis_tickangle=-45,
    margin=dict(l=10, r=10, t=40, b=140),
)

st.plotly_chart(fig_type, use_container_width=True)

# -----------------------------
# BAR: % PAYING BY GFA BUCKET
# -----------------------------
st.subheader("Share of buildings paying fines by GFA bucket (only buckets with fines)")

long_gfa = summary_gfa.melt(
    id_vars=["GFA_100k_bucket", "n_buildings", "n_paying_2024", "n_paying_2030",
             "fines_2024_total", "fines_2030_total",
             "avg_fine_2024", "avg_fine_2030"],
    value_vars=["pct_2024", "pct_2030"],
    var_name="Fine Year",
    value_name="Percent Paying",
)

long_gfa["Fine Year"] = long_gfa["Fine Year"].map(year_label_map)
long_gfa["Avg Fine (per paying bldg)"] = long_gfa.apply(_choose_avg, axis=1)

fig_gfa = px.bar(
    long_gfa,
    x="GFA_100k_bucket",
    y="Percent Paying",
    color="Fine Year",
    barmode="group",
    custom_data=[
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "Avg Fine (per paying bldg)",
    ],
)

fig_gfa.update_traces(
    texttemplate="%{y:.0f}%",
    textposition="outside",
)

fig_gfa.update_traces(
    hovertemplate=(
        "GFA bucket: %{x}<br>"
        "Year: %{marker.color}<br>"
        "Percent paying: %{y:.0f}%<br>"
        "Buildings: %{customdata[0]:,}<br>"
        "Paying 2024: %{customdata[1]:,}<br>"
        "Paying 2030: %{customdata[2]:,}<br>"
        "Total fines 2024: $%{customdata[3]:,.0f}<br>"
        "Total fines 2030: $%{customdata[4]:,.0f}<br>"
        "Avg fine (per paying bldg): $%{customdata[5]:,.0f}"
        "<extra></extra>"
    )
)

fig_gfa.update_layout(
    xaxis_title="GFA bucket (ft²)",
    yaxis_title="% of buildings paying fines",
    yaxis_range=[0, 100],
    xaxis_tickangle=-45,
    margin=dict(l=10, r=10, t=40, b=140),
)

st.plotly_chart(fig_gfa, use_container_width=True)

# -----------------------------
# BAR: % PAYING BY AGE BUCKET
# -----------------------------
st.subheader("Share of buildings paying fines by building age bucket (only buckets with fines)")

long_age = summary_age.melt(
    id_vars=["Age_10yr_bucket", "n_buildings", "n_paying_2024", "n_paying_2030",
             "fines_2024_total", "fines_2030_total",
             "avg_fine_2024", "avg_fine_2030"],
    value_vars=["pct_2024", "pct_2030"],
    var_name="Fine Year",
    value_name="Percent Paying",
)

long_age["Fine Year"] = long_age["Fine Year"].map(year_label_map)
long_age["Avg Fine (per paying bldg)"] = long_age.apply(_choose_avg, axis=1)

fig_age = px.bar(
    long_age,
    x="Age_10yr_bucket",
    y="Percent Paying",
    color="Fine Year",
    barmode="group",
    custom_data=[
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "Avg Fine (per paying bldg)",
    ],
)

fig_age.update_traces(
    texttemplate="%{y:.0f}%",
    textposition="outside",
)

fig_age.update_traces(
    hovertemplate=(
        "Age bucket: %{x}<br>"
        "Year: %{marker.color}<br>"
        "Percent paying: %{y:.0f}%<br>"
        "Buildings: %{customdata[0]:,}<br>"
        "Paying 2024: %{customdata[1]:,}<br>"
        "Paying 2030: %{customdata[2]:,}<br>"
        "Total fines 2024: $%{customdata[3]:,.0f}<br>"
        "Total fines 2030: $%{customdata[4]:,.0f}<br>"
        "Avg fine (per paying bldg): $%{customdata[5]:,.0f}"
        "<extra></extra>"
    )
)

fig_age.update_layout(
    xaxis_title="Building age (years)",
    yaxis_title="% of buildings paying fines",
    yaxis_range=[0, 100],
    xaxis_tickangle=-45,
    margin=dict(l=10, r=10, t=180, b=180),
)

st.plotly_chart(fig_age, use_container_width=True)

# -----------------------------
# BAR: % PAYING BY BOROUGH
# -----------------------------
st.subheader("Share of buildings paying fines by borough")

borough_order = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
summary_borough[borough_col] = pd.Categorical(
    summary_borough[borough_col],
    categories=borough_order,
    ordered=True
)

long_borough = summary_borough.melt(
    id_vars=[borough_col, "n_buildings", "n_paying_2024", "n_paying_2030",
             "fines_2024_total", "fines_2030_total",
             "avg_fine_2024", "avg_fine_2030"],
    value_vars=["pct_2024", "pct_2030"],
    var_name="Fine Year",
    value_name="Percent Paying",
)

long_borough["Fine Year"] = long_borough["Fine Year"].map(year_label_map)
long_borough["Avg Fine (per paying bldg)"] = long_borough.apply(_choose_avg, axis=1)

fig_borough = px.bar(
    long_borough,
    x=borough_col,
    y="Percent Paying",
    color="Fine Year",
    barmode="group",
    custom_data=[
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "Avg Fine (per paying bldg)",
    ],
)

fig_borough.update_traces(
    texttemplate="%{y:.0f}%",
    textposition="outside",
)

fig_borough.update_traces(
    hovertemplate=(
        "Borough: %{x}<br>"
        "Year: %{marker.color}<br>"
        "Percent paying: %{y:.0f}%<br>"
        "Buildings: %{customdata[0]:,}<br>"
        "Paying 2024: %{customdata[1]:,}<br>"
        "Paying 2030: %{customdata[2]:,}<br>"
        "Total fines 2024: $%{customdata[3]:,.0f}<br>"
        "Total fines 2030: $%{customdata[4]:,.0f}<br>"
        "Avg fine (per paying bldg): $%{customdata[5]:,.0f}"
        "<extra></extra>"
    )
)

fig_borough.update_layout(
    xaxis_title="Borough",
    yaxis_title="% of buildings paying fines",
    yaxis_range=[0, 100],
    xaxis_tickangle=0,
    margin=dict(l=10, r=10, t=40, b=80),
)

st.plotly_chart(fig_borough, use_container_width=True)

st.markdown("---")

# =========================================================
# FINE SEVERITY BANDS AS PERCENTAGES – 2024 & 2030
# =========================================================
st.subheader("Fine severity distribution by building type (% of paying buildings)")

sev_bins = [0, 10_000, 50_000, 200_000, 1_000_000, np.inf]
sev_labels = ["0–10k", "10k–50k", "50k–200k", "200k–1M", "1M+"]

severity_color_map = {
    "0–10k":   "#FFD166",
    "10k–50k": "#06D6A0",
    "50k–200k": "#118AB2",
    "200k–1M": "#EF476F",
    "1M+":    "#9B5DE5",
}

# ---------- 2024 ----------
st.markdown("**2024 fines – severity mix within each building type**")

df_sev_2024 = df_viz[df_viz[fine_2024_col] > 0].copy()

if not df_sev_2024.empty:
    df_sev_2024["Fine_2024_band"] = pd.cut(
        df_sev_2024[fine_2024_col],
        bins=sev_bins,
        labels=sev_labels,
        include_lowest=True
    )

    sev_summary_2024 = (
        df_sev_2024
        .groupby([type_col, "Fine_2024_band"], dropna=False)
        .agg(
            n_buildings=("Property ID", "count")
        )
        .reset_index()
    )

    sev_summary_2024[type_col] = sev_summary_2024[type_col].astype(str)
    sev_summary_2024["n_buildings"] = sev_summary_2024["n_buildings"].astype(int)

    total_per_type_2024 = sev_summary_2024.groupby(type_col)["n_buildings"].transform("sum")
    sev_summary_2024["pct_buildings"] = np.where(
        total_per_type_2024 > 0,
        sev_summary_2024["n_buildings"] / total_per_type_2024 * 100,
        0
    )
    sev_summary_2024["pct_buildings"] = sev_summary_2024["pct_buildings"].round(0).astype(int)

    fig_sev_2024 = px.bar(
        sev_summary_2024,
        x=type_col,
        y="pct_buildings",
        color="Fine_2024_band",
        barmode="stack",
        category_orders={"Fine_2024_band": sev_labels},
        color_discrete_map=severity_color_map,
    )

    fig_sev_2024.update_traces(
        texttemplate="%{y:.0f}%",
        textposition="inside",
        hovertemplate=(
            "%{x}<br>"
            "Severity band: %{marker.color}<br>"
            "Share of paying buildings: %{y:.0f}%"
            "<extra></extra>"
        )
    )

    fig_sev_2024.update_layout(
        title="2024 LL97 fine severity distribution by building type (within-type % of paying buildings)",
        xaxis_title="Building type",
        yaxis_title="% of paying buildings (within type)",
        yaxis_range=[0, 100],
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=180),
    )

    st.plotly_chart(fig_sev_2024, use_container_width=True)
else:
    st.info("No 2024 fines found to compute severity bands.")

st.markdown("---")

# ---------- 2030 ----------
st.markdown("**2030 fines – severity mix within each building type**")

df_sev_2030 = df_viz[df_viz[fine_2030_col] > 0].copy()

if not df_sev_2030.empty:
    df_sev_2030["Fine_2030_band"] = pd.cut(
        df_sev_2030[fine_2030_col],
        bins=sev_bins,
        labels=sev_labels,
        include_lowest=True
    )

    sev_summary_2030 = (
        df_sev_2030
        .groupby([type_col, "Fine_2030_band"], dropna=False)
        .agg(
            n_buildings=("Property ID", "count")
        )
        .reset_index()
    )

    sev_summary_2030[type_col] = sev_summary_2030[type_col].astype(str)
    sev_summary_2030["n_buildings"] = sev_summary_2030["n_buildings"].astype(int)

    total_per_type_2030 = sev_summary_2030.groupby(type_col)["n_buildings"].transform("sum")
    sev_summary_2030["pct_buildings"] = np.where(
        total_per_type_2030 > 0,
        sev_summary_2030["n_buildings"] / total_per_type_2030 * 100,
        0
    )
    sev_summary_2030["pct_buildings"] = sev_summary_2030["pct_buildings"].round(0).astype(int)

    fig_sev_2030 = px.bar(
        sev_summary_2030,
        x=type_col,
        y="pct_buildings",
        color="Fine_2030_band",
        barmode="stack",
        category_orders={"Fine_2030_band": sev_labels},
        color_discrete_map=severity_color_map,
    )

    fig_sev_2030.update_traces(
        texttemplate="%{y:.0f}%",
        textposition="inside",
        hovertemplate=(
            "%{x}<br>"
            "Severity band: %{marker.color}<br>"
            "Share of paying buildings: %{y:.0f}%"
            "<extra></extra>"
        )
    )

    fig_sev_2030.update_layout(
        title="2030 LL97 fine severity distribution by building type (within-type % of paying buildings)",
        xaxis_title="Building type",
        yaxis_title="% of paying buildings (within type)",
        yaxis_range=[0, 100],
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=200),
    )

    st.plotly_chart(fig_sev_2030, use_container_width=True)
else:
    st.info("No 2030 fines found to compute severity bands.")

st.markdown("---")

# =========================================================
# NEW: QUINTILE DISTRIBUTION BY BUILDING TYPE
# =========================================================
st.subheader("Fine distribution by building type – quintiles of fines (per paying building)")

# 🔍 Explanation block for quintiles
st.markdown(
    """
**How to read these quintile charts**

- We only look at buildings that **actually pay a fine**.
- For each building type, we:
  - sort all paying buildings from **smallest fine → largest fine**, then  
  - split them into **5 equal-sized groups (quintiles)**:
    - **Q1** = lowest 20% of fines  
    - **Q2** = next 20%  
    - **Q3** = middle 20%  
    - **Q4** = higher 20%  
    - **Q5** = **highest 20% of fines**
- Each stacked bar adds up to **100% of paying buildings** for that building type.
- Colors show **how many of that type** fall into low vs high fine ranges:
  - more Q1/Q2 → most buildings get **smaller fines**
  - more Q4/Q5 → many buildings get **very large fines**.
- On hover, you also see the **fine levels for that type & quintile**:
  - how many buildings are in that quintile  
  - min, max, and median fine in that group.
"""
)

quintile_labels = [
    "Q1 (lowest fines)",
    "Q2",
    "Q3",
    "Q4",
    "Q5 (highest fines)"
]

quintile_color_map = {
    "Q1 (lowest fines)": "#BEE3F8",
    "Q2": "#90CDF4",
    "Q3": "#63B3ED",
    "Q4": "#4299E1",
    "Q5 (highest fines)": "#3182CE",
}

# ---------- 2024 QUINTILES ----------
st.markdown("**2024 – fine quintiles within each building type (only paying buildings)**")

df_q_2024 = df_viz[df_viz[fine_2024_col] > 0].copy()

if not df_q_2024.empty and df_q_2024[fine_2024_col].nunique() > 1:
    df_q_2024["Fine_2024_quintile"] = pd.qcut(
        df_q_2024[fine_2024_col],
        5,
        labels=quintile_labels,
        duplicates="drop"
    )

    q_summary_2024 = (
        df_q_2024
        .groupby([type_col, "Fine_2024_quintile"], dropna=False)
        .agg(
            n_buildings=("Property ID", "count"),
            min_fine_2024=(fine_2024_col, "min"),
            max_fine_2024=(fine_2024_col, "max"),
            median_fine_2024=(fine_2024_col, "median"),
        )
        .reset_index()
    )

    q_summary_2024[type_col] = q_summary_2024[type_col].astype(str)
    q_summary_2024["n_buildings"] = q_summary_2024["n_buildings"].fillna(0).round(0).astype(int)

    for col in ["min_fine_2024", "max_fine_2024", "median_fine_2024"]:
        q_summary_2024[col] = q_summary_2024[col].fillna(0).round(0).astype(int)

    total_per_type_2024_q = q_summary_2024.groupby(type_col)["n_buildings"].transform("sum")
    q_summary_2024["pct_buildings"] = np.where(
        total_per_type_2024_q > 0,
        q_summary_2024["n_buildings"] / total_per_type_2024_q * 100,
        0
    )
    q_summary_2024["pct_buildings"] = q_summary_2024["pct_buildings"].round(0).astype(int)

    fig_q_2024 = px.bar(
        q_summary_2024,
        x=type_col,
        y="pct_buildings",
        color="Fine_2024_quintile",
        barmode="stack",
        category_orders={"Fine_2024_quintile": quintile_labels},
        color_discrete_map=quintile_color_map,
        custom_data=[
            "n_buildings",
            "min_fine_2024",
            "max_fine_2024",
            "median_fine_2024",
        ],
    )

    fig_q_2024.update_traces(
        texttemplate="%{y:.0f}%",
        textposition="inside",
        hovertemplate=(
            "%{x}<br>"
            "Quintile: %{marker.color}<br>"
            "Share of paying buildings: %{y:.0f}%<br>"
            "Count in this quintile: %{customdata[0]:,}<br>"
            "Fine range in this quintile: $%{customdata[1]:,.0f} – $%{customdata[2]:,.0f}<br>"
            "Median fine: $%{customdata[3]:,.0f}"
            "<extra></extra>"
        ),
    )

    fig_q_2024.update_layout(
        title="2024 LL97 fine quintile distribution by building type (within-type % of paying buildings)",
        xaxis_title="Building type",
        yaxis_title="% of paying buildings (within type)",
        yaxis_range=[0, 100],
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=200),
    )

    st.plotly_chart(fig_q_2024, use_container_width=True)
else:
    st.info("Not enough variation in 2024 fines to compute quintiles.")

st.markdown("---")

# ---------- 2030 QUINTILES ----------
st.markdown("**2030 – fine quintiles within each building type (only paying buildings)**")

df_q_2030 = df_viz[df_viz[fine_2030_col] > 0].copy()

if not df_q_2030.empty and df_q_2030[fine_2030_col].nunique() > 1:
    df_q_2030["Fine_2030_quintile"] = pd.qcut(
        df_q_2030[fine_2030_col],
        5,
        labels=quintile_labels,
        duplicates="drop"
    )

    q_summary_2030 = (
        df_q_2030
        .groupby([type_col, "Fine_2030_quintile"], dropna=False)
        .agg(
            n_buildings=("Property ID", "count"),
            min_fine_2030=(fine_2030_col, "min"),
            max_fine_2030=(fine_2030_col, "max"),
            median_fine_2030=(fine_2030_col, "median"),
        )
        .reset_index()
    )

    q_summary_2030[type_col] = q_summary_2030[type_col].astype(str)
    q_summary_2030["n_buildings"] = q_summary_2030["n_buildings"].fillna(0).round(0).astype(int)

    for col in ["min_fine_2030", "max_fine_2030", "median_fine_2030"]:
        q_summary_2030[col] = q_summary_2030[col].fillna(0).round(0).astype(int)

    total_per_type_2030_q = q_summary_2030.groupby(type_col)["n_buildings"].transform("sum")
    q_summary_2030["pct_buildings"] = np.where(
        total_per_type_2030_q > 0,
        q_summary_2030["n_buildings"] / total_per_type_2030_q * 100,
        0
    )
    q_summary_2030["pct_buildings"] = q_summary_2030["pct_buildings"].round(0).astype(int)

    fig_q_2030 = px.bar(
        q_summary_2030,
        x=type_col,
        y="pct_buildings",
        color="Fine_2030_quintile",
        barmode="stack",
        category_orders={"Fine_2030_quintile": quintile_labels},
        color_discrete_map=quintile_color_map,
        custom_data=[
            "n_buildings",
            "min_fine_2030",
            "max_fine_2030",
            "median_fine_2030",
        ],
    )

    fig_q_2030.update_traces(
        texttemplate="%{y:.0f}%",
        textposition="inside",
        hovertemplate=(
            "%{x}<br>"
            "Quintile: %{marker.color}<br>"
            "Share of paying buildings: %{y:.0f}%<br>"
            "Count in this quintile: %{customdata[0]:,}<br>"
            "Fine range in this quintile: $%{customdata[1]:,.0f} – $%{customdata[2]:,.0f}<br>"
            "Median fine: $%{customdata[3]:,.0f}"
            "<extra></extra>"
        ),
    )

    fig_q_2030.update_layout(
        title="2030 LL97 fine quintile distribution by building type (within-type % of paying buildings)",
        xaxis_title="Building type",
        yaxis_title="% of paying buildings (within type)",
        yaxis_range=[0, 100],
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=40, b=200),
    )

    st.plotly_chart(fig_q_2030, use_container_width=True)
else:
    st.info("Not enough variation in 2030 fines to compute quintiles.")

st.markdown("---")

# -----------------------------
# RAW DATA
# -----------------------------
with st.expander("Show raw data (first 500 rows)"):
    st.write(df_viz.head(500))

# =====================================================================
# TOP-20 INVESTOR DRILLDOWN (MASTER VIEW + TIMELINES + DOWNLOAD)
# =====================================================================
st.markdown("---")
st.header("Top 20 LL97 Buildings – Investor Drilldown")

st.markdown(
    """
This section focuses on the **top 20 LL97-exposed buildings by typology and overall**  
and brings together **fines, retrofit permit history, and ownership** into a single investor view.
"""
)

# Paths for the top-20 artifacts (already created in your notebook work)
TOP20_MASTER_PATH = "LL97_top20_master_investor_view.csv"
TOP20_TIMELINE_PATH = "LL97_top20_permit_timeline_20yr.csv"


@st.cache_data
def load_top20_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Ensure a unified ID column
    if "PropertyID" in df.columns and "Property ID" not in df.columns:
        df["Property ID"] = df["PropertyID"]
    return df


@st.cache_data
def load_top20_timeline(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, parse_dates=["permit_date"])
    # Mirror ID naming for joins/filtering
    if "PropertyID" not in df.columns and "Property ID" in df.columns:
        df["PropertyID"] = df["Property ID"]
    return df


# Try to load the master view
try:
    top20_master = load_top20_master(TOP20_MASTER_PATH)
except Exception as e:
    st.info(
        f"Top-20 master dataset not found or not readable at `{TOP20_MASTER_PATH}`.\n\n"
        f"Error: {e}"
    )
    top20_master = None

if top20_master is not None:
    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------
    st.subheader("Download full top-20 master dataset")

    st.markdown(
        """
This CSV combines, for each of the top-20 buildings:

- **LL97 fines** (2024 & 2030)  
- **Basic building attributes** (typology, address, postal code)  
- **Energy-related permit history** (count, first/last permit year, type mix)  
- **Ownership snapshot** (primary owner name + high-level category)
"""
    )

    st.download_button(
        label="⬇️ Download Top-20 LL97 Master Investor View (CSV)",
        data=top20_master.to_csv(index=False),
        file_name="LL97_top20_master_investor_view.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.subheader("Filter the universe (typology & ownership)")

    # Make sure the key columns exist, but fail gracefully if they don't
    if "Primary Property Type - Self Selected" in top20_master.columns:
        type_options = sorted(top20_master["Primary Property Type - Self Selected"].dropna().unique())
    else:
        type_options = []

    if "owner_category" in top20_master.columns:
        owner_cat_options = sorted(top20_master["owner_category"].fillna("Unknown").unique())
    else:
        owner_cat_options = []

    colf1, colf2 = st.columns(2)

    selected_types = colf1.multiselect(
        "Building typologies (from top-20 set)",
        options=type_options,
        default=type_options,
    ) if type_options else []

    selected_owner_cats = colf2.multiselect(
        "Owner categories",
        options=owner_cat_options,
        default=owner_cat_options,
    ) if owner_cat_options else []

    filtered = top20_master.copy()

    if selected_types and "Primary Property Type - Self Selected" in filtered.columns:
        filtered = filtered[filtered["Primary Property Type - Self Selected"].isin(selected_types)]

    if selected_owner_cats and "owner_category" in filtered.columns:
        filtered = filtered[filtered["owner_category"].fillna("Unknown").isin(selected_owner_cats)]

    st.caption(f"Filtered top-20 rows: **{len(filtered)}** (out of 140)")

    # -----------------------------
    # TABS FOR INVESTOR VIEWS
    # -----------------------------
    tab_summary, tab_timeline, tab_owners = st.tabs(
        ["🏢 Fines & building tables", "📈 Retrofit permit timelines", "🏦 Ownership mix"]
    )

    # -----------------------------------------------------------------
    # TAB 1 – FINES & BUILDING TABLES (TOP 20 BY 2024/2030 FINES)
    # -----------------------------------------------------------------
    with tab_summary:
        st.markdown(
            """
We rank the **top 20 buildings by LL97 fines** to show where  
**absolute $ exposure** is concentrated, within the already-filtered universe.
"""
        )

        # Helper: pick columns that exist to avoid KeyErrors
        base_cols_pref = [
            "PropertyID", "Property ID",
            "Primary Property Type - Self Selected",
            "Address 1", "Postal Code",
            "Fine_2024_$", "Fine_2030_$",
            "n_energy_permits",
            "first_permit_year", "last_permit_year",
            "primary_owner", "owner_category",
        ]

        existing_base_cols = [c for c in base_cols_pref if c in filtered.columns]

        # ---- Top 20 by 2024 fines ----
        if "Fine_2024_$" in filtered.columns:
            top_2024 = (
                filtered.sort_values("Fine_2024_$", ascending=False)
                .head(20)[existing_base_cols]
            )

            st.subheader("Top 20 by 2024 LL97 fines (within filtered set)")
            st.dataframe(top_2024, use_container_width=True)
        else:
            st.info("Column `Fine_2024_$` not found in top-20 master dataset.")

        st.markdown("---")

        # ---- Top 20 by 2030 fines ----
        if "Fine_2030_$" in filtered.columns:
            top_2030 = (
                filtered.sort_values("Fine_2030_$", ascending=False)
                .head(20)[existing_base_cols]
            )

            st.subheader("Top 20 by 2030 LL97 fines (within filtered set)")
            st.dataframe(top_2030, use_container_width=True)
        else:
            st.info("Column `Fine_2030_$` not found in top-20 master dataset.")

    # -----------------------------------------------------------------
    # TAB 2 – PERMIT TIMELINES (RETROFIT INTENSITY OVER LAST 20 YEARS)
    # -----------------------------------------------------------------
    with tab_timeline:
        st.markdown(
            """
We look at **DOB energy-related permits over the last ~20 years**  
for these top-20 LL97 buildings:

- Each dot/segment represents **actual capital/maintenance work** on mechanicals, boilers, DHW, etc.  
- Higher counts in recent years can indicate **active retrofit programs** or **ongoing churn** in systems.
"""
        )

        try:
            timeline = load_top20_timeline(TOP20_TIMELINE_PATH)
        except Exception as e:
            st.info(
                f"Permit timeline dataset not found or not readable at `{TOP20_TIMELINE_PATH}`.\n\n"
                f"Error: {e}"
            )
            timeline = None

        if timeline is not None and not filtered.empty:
            # Align IDs
            if "Property ID" not in filtered.columns and "PropertyID" in filtered.columns:
                filtered["Property ID"] = filtered["PropertyID"]

            if "Property ID" in filtered.columns and "Property ID" in timeline.columns:
                id_list = filtered["Property ID"].unique().tolist()
                tl_filtered = timeline[timeline["Property ID"].isin(id_list)].copy()
            else:
                tl_filtered = timeline.copy()

            if tl_filtered.empty:
                st.info("No matched permits for the current filtered top-20 selection.")
            else:
                # Aggregate permits by year & typology
                group_cols = ["permit_year"]
                if "Primary Property Type - Self Selected" in tl_filtered.columns:
                    group_cols.append("Primary Property Type - Self Selected")

                tl_agg = (
                    tl_filtered
                    .groupby(group_cols, dropna=False)
                    .agg(n_permits=("Job #", "count"))
                    .reset_index()
                )

                st.subheader("Energy-related permits per year (top-20 LL97 buildings)")

                if "Primary Property Type - Self Selected" in tl_agg.columns:
                    fig_tl = px.line(
                        tl_agg,
                        x="permit_year",
                        y="n_permits",
                        color="Primary Property Type - Self Selected",
                        markers=True,
                    )
                    fig_tl.update_layout(
                        xaxis_title="Permit year",
                        yaxis_title="# of energy-related permits",
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=40),
                    )
                else:
                    fig_tl = px.line(
                        tl_agg,
                        x="permit_year",
                        y="n_permits",
                        markers=True,
                    )
                    fig_tl.update_layout(
                        xaxis_title="Permit year",
                        yaxis_title="# of energy-related permits (all filtered buildings)",
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=40),
                    )

                st.plotly_chart(fig_tl, use_container_width=True)

                # Optional: show raw timeline slice
                with st.expander("Show raw permit timeline rows (filtered selection)"):
                    st.dataframe(tl_filtered.sort_values(["Property ID", "permit_date"]).head(1000))

        elif timeline is None:
            st.stop()
        else:
            st.info("No buildings available after filters; adjust filters above to see timeline.")

    # -----------------------------------------------------------------
    # TAB 3 – OWNERSHIP MIX (WHO OWNS THE TOP-20 FINE EXPOSURE)
    # -----------------------------------------------------------------
    with tab_owners:
        st.markdown(
            """
We classify **who actually owns the top-20 LL97 liability**:

- Public housing & government agencies  
- Universities & educational institutions  
- Corporate / fund / REIT-style vehicles  
- Smaller private owners / individuals  
- Unknown / ambiguous LLCs
"""
        )

        if "owner_category" in filtered.columns:
            own_group = (
                filtered
                .assign(owner_category=lambda d: d["owner_category"].fillna("Unknown"))
                .groupby("owner_category", dropna=False)
                .agg(
                    n_buildings=("Property ID", "nunique") if "Property ID" in filtered.columns else ("PropertyID", "nunique"),
                    total_fines_2024=("Fine_2024_$", "sum") if "Fine_2024_$" in filtered.columns else ("PropertyID", "count"),
                    total_fines_2030=("Fine_2030_$", "sum") if "Fine_2030_$" in filtered.columns else ("PropertyID", "count"),
                )
                .reset_index()
            )

            total_blds = own_group["n_buildings"].sum()
            own_group["pct_buildings"] = np.where(
                total_blds > 0,
                own_group["n_buildings"] / total_blds * 100,
                0,
            ).round(1)

            st.subheader("Ownership mix within the (filtered) top-20 LL97 buildings")

            fig_own = px.bar(
                own_group,
                x="owner_category",
                y="n_buildings",
                text="n_buildings",
                hover_data={
                    "pct_buildings": True,
                    "total_fines_2024": ":,.0f",
                    "total_fines_2030": ":,.0f",
                },
            )
            fig_own.update_traces(
                textposition="outside",
                hovertemplate=(
                    "Owner category: %{x}<br>"
                    "Buildings in top-20 set: %{y:,}<br>"
                    "Share of filtered top-20: %{customdata[0]:.1f}%<br>"
                    "Total 2024 fines: $%{customdata[1]:,.0f}<br>"
                    "Total 2030 fines: $%{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
            )
            fig_own.update_layout(
                xaxis_title="Owner category",
                yaxis_title="# of buildings in top-20 set",
                xaxis_tickangle=-30,
                margin=dict(l=10, r=10, t=40, b=80),
            )
            st.plotly_chart(fig_own, use_container_width=True)

            with st.expander("Show ownership table"):
                st.dataframe(own_group, use_container_width=True)
        else:
            st.info("Column `owner_category` not found in top-20 master dataset; cannot build ownership mix.")