import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page config ---
st.set_page_config(page_title="Iris Explorer", layout="wide")

# --- Title ---
st.title("Iris Dataset — Interactive Explorer 🌸")
st.write("This app loads the Iris dataset from a public URL and provides interactive visualization and analysis.")

# --- Best public URL ---
IRIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv(IRIS_URL)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename species column properly
    if "species" in df.columns:
        df = df.rename(columns={"species": "Species"})

    return df

df = load_data()

# Numeric column ordering
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

species_list = ["All"] + sorted(df["Species"].unique().tolist())
selected_species = st.sidebar.selectbox("Select Species", species_list)

x_axis = st.sidebar.selectbox("X Axis", numeric_cols)
y_axis = st.sidebar.selectbox("Y Axis", numeric_cols)

min_x = float(df[x_axis].min())
max_x = float(df[x_axis].max())

x_range = st.sidebar.slider(
    f"{x_axis} Range",
    min_value=min_x,
    max_value=max_x,
    value=(min_x, max_x)
)

show_raw = st.sidebar.checkbox("Show Raw Data")

# --- Apply filters ---
filtered = df.copy()

if selected_species != "All":
    filtered = filtered[filtered["Species"] == selected_species]

filtered = filtered[
    (filtered[x_axis] >= x_range[0]) &
    (filtered[x_axis] <= x_range[1])
]

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Rows (Filtered)", len(filtered))
col2.metric("Species Count", filtered["Species"].nunique())
col3.metric(f"Mean {y_axis}", f"{filtered[y_axis].mean():.2f}")

# --- Layout: plots + summary ---
left, right = st.columns((2, 1))

with left:
    st.subheader("Scatter Plot")
    scatter = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color="Species",
        title=f"{y_axis} vs {x_axis}",
        hover_data=filtered.columns
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Histogram")
    hist = px.histogram(
        filtered,
        x=y_axis,
        color="Species",
        barmode="overlay",
        title=f"Distribution of {y_axis}"
    )
    st.plotly_chart(hist, use_container_width=True)

with right:
    st.subheader("Boxplot Summary")
    box = px.box(filtered, y=numeric_cols, points="all")
    st.plotly_chart(box, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(filtered.describe().T.style.format("{:.2f}"))

# --- Raw data ---
if show_raw:
    st.subheader("Filtered Raw Data")
    st.dataframe(filtered)

# --- Download ---
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered CSV", csv, "iris_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Dataset source: github.com/mwaskom/seaborn-data (public)")
