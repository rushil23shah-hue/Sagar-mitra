"""
Sagar Mitra - Phase 3: Historical Data Query Agent (Gemini Function Calling)
==============================================================================
Lets Gemini query the real historical Argo dataset directly and answer
questions like "when was the last marine heatwave near Ratnagiri?" or
generate trend graphs, e.g. "plot temperature over years".

Adapted from the notebook: uses app.data.store.final_dataset instead of a
bare global, and saves plots under ./static/plots (served by FastAPI's
StaticFiles mount) instead of /tmp, returning a URL the frontend can use
directly as an <img src>.
"""
import os
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from google.genai import types

from app.data import store
from app.advisory import generate_content

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Full base URL so plot links are directly clickable/pasteable during testing.
# Set SAGAR_MITRA_BASE_URL in .env once deployed (e.g. https://your-app.onrender.com).
BASE_URL = os.environ.get("SAGAR_MITRA_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# TOOL FUNCTIONS (Gemini calls these automatically when needed)
# ---------------------------------------------------------------------------

def query_historical_extreme(parameter: str, mode: str, region_lat: float = None,
                              region_lon: float = None, radius_deg: float = 2.0) -> dict:
    """
    Finds the historical max/min/last-occurrence of a parameter.
    parameter: one of 'TEMP','PSAL','PRES'
    mode: 'max' or 'min'
    """
    df = store.final_dataset.copy()
    if region_lat is not None and region_lon is not None:
        df = df[(df["LATITUDE"].between(region_lat - radius_deg, region_lat + radius_deg)) &
                (df["LONGITUDE"].between(region_lon - radius_deg, region_lon + radius_deg))]
    if df.empty or parameter not in df.columns:
        return {"error": "No matching data found."}

    row = df.loc[df[parameter].idxmax()] if mode == "max" else df.loc[df[parameter].idxmin()]
    return {
        "value": float(row[parameter]),
        "date": str(row.get("JULD", "unknown")),
        "lat": float(row["LATITUDE"]), "lon": float(row["LONGITUDE"]),
    }


def get_yearly_trend(parameter: str, region_lat: float = None,
                      region_lon: float = None, radius_deg: float = 2.0) -> dict:
    """Returns yearly average of a parameter for trend/graph questions."""
    df = store.final_dataset.copy()
    if region_lat is not None and region_lon is not None:
        df = df[(df["LATITUDE"].between(region_lat - radius_deg, region_lat + radius_deg)) &
                (df["LONGITUDE"].between(region_lon - radius_deg, region_lon + radius_deg))]
    df["YEAR"] = pd.to_datetime(df["JULD"]).dt.year
    yearly = df.groupby("YEAR")[parameter].mean().dropna()
    return {"years": yearly.index.tolist(), "values": yearly.values.round(2).tolist()}


def plot_yearly_trend(parameter: str, region_lat: float = None,
                       region_lon: float = None) -> dict:
    """Generates and saves a trend graph, returns the file path + public URL."""
    data = get_yearly_trend(parameter, region_lat, region_lon)
    plt.figure(figsize=(8, 4))
    plt.plot(data["years"], data["values"], marker="o")
    plt.title(f"{parameter} Trend Over Years")
    plt.xlabel("Year")
    plt.ylabel(parameter)
    plt.grid(True)

    filename = f"{parameter}_trend_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"file_path": path, "url": f"{BASE_URL}/static/plots/{filename}"}


def plot_category_distribution(parameter: str = "TEMP", region_lat: float = None,
                                region_lon: float = None) -> dict:
    """
    Generates a colored pie chart showing what % of historical SURFACE
    (<=30m) readings fall into each anomaly category (Cold / Normal / Warm /
    Hot), based on the real calibrated surface climatology. Use when user
    asks for a 'pie chart' or 'distribution' or 'breakdown' of conditions.
    """
    df = store.final_dataset.copy()
    df = df[df["PRES"] <= 30]  # match the surface-only climatology baseline
    if region_lat is not None and region_lon is not None:
        df = df[(df["LATITUDE"].between(region_lat - 2.0, region_lat + 2.0)) &
                (df["LONGITUDE"].between(region_lon - 2.0, region_lon + 2.0))]
    mu = store.calibrated_clim["temp"]["mu"]
    sigma = store.calibrated_clim["temp"]["sigma"] or 0.9
    z = (df["TEMP"] - mu) / sigma

    labels = ["Cold (z<-1)", "Normal (-1..1)", "Warm (1..2)", "Hot (z>2)"]
    counts = [
        int((z < -1).sum()),
        int(((z >= -1) & (z <= 1)).sum()),
        int(((z > 1) & (z <= 2)).sum()),
        int((z > 2).sum()),
    ]
    colors = ["#4A90D9", "#7ED957", "#F5A623", "#D0021B"]

    present = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if not present:
        present = [("No data", 1, "#CCCCCC")]
    plot_labels = [p[0] for p in present]
    plot_counts = [p[1] for p in present]
    plot_colors = [p[2] for p in present]

    plt.figure(figsize=(7, 6))
    wedges, _, autotexts = plt.pie(
        plot_counts, colors=plot_colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.8,
    )
    plt.legend(wedges, plot_labels, title="Condition", loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    plt.title(f"{parameter} Surface Condition Distribution")
    plt.tight_layout()

    filename = f"{parameter}_piechart_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"file_path": path, "url": f"{BASE_URL}/static/plots/{filename}", "counts": dict(zip(labels, counts))}


TOOLS = [
    {
        "name": "query_historical_extreme",
        "description": "Find historical max/min value of a parameter (TEMP, PSAL, PRES), e.g. 'when was the last marine heatwave' or 'highest recorded salinity'.",
        "parameters": {
            "type": "object",
            "properties": {
                "parameter": {"type": "string", "enum": ["TEMP", "PSAL", "PRES"]},
                "mode": {"type": "string", "enum": ["max", "min"]},
                "region_lat": {"type": "number"},
                "region_lon": {"type": "number"},
            },
            "required": ["parameter", "mode"],
        },
    },
    {
        "name": "get_yearly_trend",
        "description": "Get yearly averages of a parameter over time for a region, for trend questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "parameter": {"type": "string", "enum": ["TEMP", "PSAL", "PRES"]},
                "region_lat": {"type": "number"},
                "region_lon": {"type": "number"},
            },
            "required": ["parameter"],
        },
    },
    {
        "name": "plot_yearly_trend",
        "description": "Generate a graph image of a parameter's trend over years. Use when user asks for a 'graph', 'chart', or 'plot'.",
        "parameters": {
            "type": "object",
            "properties": {
                "parameter": {"type": "string", "enum": ["TEMP", "PSAL", "PRES"]},
                "region_lat": {"type": "number"},
                "region_lon": {"type": "number"},
            },
            "required": ["parameter"],
        },
    },
    {
        "name": "plot_category_distribution",
        "description": "Generate a colored PIE CHART showing the breakdown/distribution of ocean temperature conditions (Cold/Normal/Warm/Hot) as percentages. Use when user asks for a 'pie chart', 'distribution', or 'breakdown'.",
        "parameters": {
            "type": "object",
            "properties": {
                "parameter": {"type": "string", "enum": ["TEMP"]},
                "region_lat": {"type": "number"},
                "region_lon": {"type": "number"},
            },
            "required": [],
        },
    },
]

_FUNCTION_MAP = {
    "query_historical_extreme": query_historical_extreme,
    "get_yearly_trend": get_yearly_trend,
    "plot_yearly_trend": plot_yearly_trend,
    "plot_category_distribution": plot_category_distribution,
}


def sagar_mitra_historical_query(user_message: str, persona: str = "Researcher",
                                  language: str = "English", model: str = "gemini-3.6-flash") -> dict:
    """
    Answers historical/trend questions using function calling against the
    real dataset. Returns a dict: {reply, mode, plot_url (if a graph was made)}.
    """
    tool = types.Tool(function_declarations=TOOLS)
    config = types.GenerateContentConfig(
        tools=[tool],
        system_instruction=(
            f"You are Sagar Mitra, answering a {persona}'s historical ocean-data question "
            f"in {language}. Use the available tools to fetch real data before answering. "
            "Never guess numbers - always call a tool first."
        ),
    )

    response = generate_content(model=model, contents=user_message, config=config)

    part = response.candidates[0].content.parts[0]
    if part.function_call:
        fn_name = part.function_call.name
        fn_args = dict(part.function_call.args)
        result = _FUNCTION_MAP[fn_name](**fn_args)

        # plot_yearly_trend returns a dict with file_path/url; other tools
        # return plain dicts already JSON-safe for the follow-up call.
        tool_result_for_model = result
        if fn_name in ("plot_yearly_trend", "plot_category_distribution"):
            tool_result_for_model = {"status": "graph generated", "url": result["url"]}

        follow_up = generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": user_message}]},
                response.candidates[0].content,
                {"role": "user", "parts": [{"function_response": {
                    "name": fn_name, "response": {"result": tool_result_for_model}}}]},
            ],
            config=config,
        )

        answer = follow_up.text
        plot_url = result["url"] if fn_name in ("plot_yearly_trend", "plot_category_distribution") else None
        return {"reply": answer, "mode": "historical", "plot_url": plot_url}

    return {"reply": response.text, "mode": "historical", "plot_url": None}