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
# If it's inside "data/", change to: "data/LL97_cleaned_no_outliers.csv"
CSV_PATH = "LL97_cleaned_no_outliers.csv"

type_col = "Primary Property Type - Self Selected"
gfa_col = "Property GFA - Self-Reported (ft²)"
year_built_col = "Year Built"
city_col = "City"
fine_2024_col = "Fine_2024_$"
fine_2030_col = "Fine_2030_$"


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

# Safety check for City column
if city_col not in df_raw.columns:
    st.error(f"City column '{city_col}' not found in CSV. Available columns include: {list(df_raw.columns)[:20]}")
    st.stop()

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

# Summary by City
summary_city = (
    df_viz
    .groupby(city_col, dropna=False)
    .agg(
        n_buildings=("Property ID", "count"),
        n_paying_2024=("Fine2024_flag", "sum"),
        n_paying_2030=("Fine2030_flag", "sum"),
        fines_2024_total=(fine_2024_col, "sum"),
        fines_2030_total=(fine_2030_col, "sum"),
    )
    .reset_index()
)
summary_city = summary_city.dropna(subset=[city_col])
summary_city = add_pct_and_avg(summary_city, city_col)

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
    margin=dict(l=10, r=10, t=40, b=140),
)

st.plotly_chart(fig_age, use_container_width=True)

# -----------------------------
# BAR: % PAYING BY CITY (TOP 10 BY # PAYING 2024)
# -----------------------------
st.subheader("Share of buildings paying fines by city (top 10 cities by # paying in 2024)")

# Pick top 10 cities by n_paying_2024
top_cities = (
    summary_city.sort_values("n_paying_2024", ascending=False)
    .head(10)[city_col]
    .tolist()
)

summary_city_top = summary_city[summary_city[city_col].isin(top_cities)].copy()

# Preserve order on x-axis
summary_city_top[city_col] = pd.Categorical(
    summary_city_top[city_col],
    categories=top_cities,
    ordered=True
)

long_city = summary_city_top.melt(
    id_vars=[city_col, "n_buildings", "n_paying_2024", "n_paying_2030",
             "fines_2024_total", "fines_2030_total",
             "avg_fine_2024", "avg_fine_2030"],
    value_vars=["pct_2024", "pct_2030"],
    var_name="Fine Year",
    value_name="Percent Paying",
)

long_city["Fine Year"] = long_city["Fine Year"].map(year_label_map)
long_city["Avg Fine (per paying bldg)"] = long_city.apply(_choose_avg, axis=1)

fig_city = px.bar(
    long_city,
    x=city_col,
    y="Percent Paying",
    color="Fine Year",
    barmode="group",
    custom_data=[
        "n_buildings", "n_paying_2024", "n_paying_2030",
        "fines_2024_total", "fines_2030_total",
        "Avg Fine (per paying bldg)",
    ],
)

fig_city.update_traces(
    texttemplate="%{y:.0f}%",
    textposition="outside",
)

fig_city.update_traces(
    hovertemplate=(
        "City: %{x}<br>"
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

fig_city.update_layout(
    xaxis_title="City",
    yaxis_title="% of buildings paying fines",
    yaxis_range=[0, 100],
    xaxis_tickangle=45,
    margin=dict(l=10, r=10, t=40, b=160),
)

st.plotly_chart(fig_city, use_container_width=True)

st.markdown("---")

# =========================================================
# FINE SEVERITY BANDS AS PERCENTAGES – 2024 & 2030
# =========================================================
st.subheader("Fine severity distribution by building type (% of paying buildings)")

# Define severity bands and bright, happy colors
sev_bins = [0, 10_000, 50_000, 200_000, 1_000_000, np.inf]
sev_labels = ["0–10k", "10k–50k", "50k–200k", "200k–1M", "1M+"]

# bright / playful palette
severity_color_map = {
    "0–10k":   "#FFD166",  # warm yellow
    "10k–50k": "#06D6A0",  # bright teal/green
    "50k–200k": "#118AB2", # vivid blue
    "200k–1M": "#EF476F",  # hot pink/red
    "1M+":    "#9B5DE5",   # bright purple
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
        margin=dict(l=10, r=10, t=40, b=180),
    )

    st.plotly_chart(fig_sev_2030, use_container_width=True)
else:
    st.info("No 2030 fines found to compute severity bands.")

st.markdown("---")

# -----------------------------
# RAW DATA
# -----------------------------
with st.expander("Show raw data (first 500 rows)"):
    st.write(df_viz.head(500))