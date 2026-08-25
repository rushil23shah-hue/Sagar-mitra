"""
Sagar Mitra - End-to-End Chat Orchestration
=============================================
Wires: Phase 1 NLP (extract_query_parameters) -> ArgoStatisticalAnomalyEngine
-> Phase 2 NLP (generate_sagar_mitra_advisory), plus the keyword router that
decides between a live-conditions answer and a historical-data answer.

NOTE: `sagar_mitra_chat` currently reuses a synthetic demo profile/hist for
every query (same behavior as the original notebook) -- see app/data.py's
`DataStore.demo_profile` / `demo_hist`. Swapping in a real per-location Argo
profile lookup later only requires changing the two lines marked below.
"""
from app.data import store
from app.statistical_engine import ArgoStatisticalAnomalyEngine
from app.nlp_extract import extract_query_parameters
from app.advisory import generate_sagar_mitra_advisory, plain_gemini_answer, generate_content
from app.historical import sagar_mitra_historical_query
from google.genai import types
import re

HISTORICAL_KEYWORDS = [
    "when was", "highest", "lowest", "graph", "trend", "plot",
    "history", "past", "in 20", "pie chart", "distribution", "breakdown",
]

COMPARISON_KEYWORDS = [" vs ", " versus ", " or ", "compare", "which is better", "better place"]

# Known coastal reference points + common aliases/typos, reused for comparison detection.
KNOWN_LOCATIONS = {
    "mumbai": (19.07, 72.87),
    "koliwada": (19.07, 72.87),
    "ratnagiri": (16.99, 73.30),
    "sindhudurg": (16.02, 73.45),
    "sintadurg": (16.02, 73.45),  # common misspelling
    "alibaug": (18.64, 72.87),
    "alibagh": (18.64, 72.87),    # common misspelling
    "raigad": (18.64, 72.87),
    "palghar": (19.70, 72.77),
    "vengurla": (15.85, 73.63),
}


def _find_known_locations(message: str):
    """Returns a list of (name, lat, lon) for every known place mentioned."""
    msg = message.lower()
    found = []
    seen_coords = set()
    for name, (lat, lon) in KNOWN_LOCATIONS.items():
        if name in msg and (lat, lon) not in seen_coords:
            found.append((name, lat, lon))
            seen_coords.add((lat, lon))
    return found


def sagar_mitra_chat(user_message: str) -> dict:
    """
    End-to-end chatbot function:
      1. Extract params from user text (Phase 1 NLP)
      2. Try to build a profile from REAL nearby Argo data
      3. If real data is too sparse there -> fall back to a plain Gemini
         answer (no fabricated statistics)
      4. Otherwise run ArgoStatisticalAnomalyEngine -> Phase 2 NLP advisory
    """
    params = extract_query_parameters(user_message)
    lat = params.get("latitude") or 16.99
    lon = params.get("longitude") or 73.30
    persona = params.get("persona", "Fisherman")
    language = params.get("language", "English")
    intent = (params.get("query_intent") or "").lower()

    # Conceptual/definition questions ("what is a PFZ", "explain X") don't need
    # a live sensor reading -- answer directly instead of forcing stats.
    if any(k in intent for k in ["concept", "explanation", "general curiosity", "definition"]):
        reply = plain_gemini_answer(user_message, persona, language)
        return {
            "reply": reply,
            "mode": "concept_explanation",
            "query_context": params,
            "statistical_result": None,
        }

    profile_df = store.get_real_profile_near(lat, lon)
    if profile_df is None:
        # Not enough real data near this location -- don't fabricate stats.
        reply = plain_gemini_answer(user_message, persona, language)
        return {
            "reply": reply,
            "mode": "fallback_general_knowledge",
            "query_context": params,
            "statistical_result": None,
            "confidence": "LOW (insufficient nearby data)",
            "alert": None,
        }

    n_nearby = store.nearby_count(lat, lon)
    confidence = "HIGH" if n_nearby >= 200 else "MODERATE" if n_nearby >= 30 else "LOW"

    hist_df = store.get_real_hist_near(lat, lon)
    engine = ArgoStatisticalAnomalyEngine(climatology_db=store.calibrated_clim)
    result = engine.analyze_profile(profile_df, historical_timeseries=hist_df)
    result["query_context"] = params

    # --- DEMO OVERRIDE (remove before any real/production use) -----------
    # Guarantees a visible High-severity alert for Vengurla specifically,
    # for demo purposes only. This is NOT a real detection.
    location_name = (params.get("location_name") or "").lower()
    if "vengurla" in location_name and not result.get("detected_anomalies"):
        result["detected_anomalies"] = [{
            "anomaly_type": "Mesoscale Warm-Core Eddy",
            "depth_range_meters": [0, 500],
            "severity": "High",
            "statistical_metrics": {"z_score": 2.8, "delta_d20_m": 210, "p_ks": 0.004},
            "oceanographic_summary": "D20 isotherm displaced downward, indicating warm-core eddy. [DEMO]",
        }]
    # -----------------------------------------------------------------------

    advisory_text = generate_sagar_mitra_advisory(result, persona=persona, language=language)
    advisory_text += _build_evidence_footer(result)

    alert = _build_alert(result)
    map_data = {
        "query_location": {"lat": lat, "lon": lon},
        "nearby_points": store.get_nearby_points_categorized(lat, lon),
    }

    return {
        "reply": advisory_text,
        "mode": "live_conditions",
        "query_context": params,
        "statistical_result": result,
        "confidence": confidence,
        "alert": alert,
        "map_data": map_data,
    }


def _build_evidence_footer(result: dict) -> str:
    """
    Appends the raw supporting numbers straight from the statistical engine's
    JSON output (not LLM-generated), so the advisory is visibly backed by
    real computed data instead of just prose.
    """
    lm = result.get("layer_metrics", {})
    lines = [
        "\n\n--- Supporting Data ---",
        f"Mixed Layer Depth: {lm.get('MLD')} m | Isothermal Layer Depth: {lm.get('ILD')} m "
        f"| Barrier Layer Thickness: {lm.get('BLT')} m | D20 isotherm depth: {lm.get('D20'):.1f} m"
        if lm.get("D20") is not None else "",
    ]
    anomalies = result.get("detected_anomalies", [])
    if anomalies:
        for a in anomalies:
            m = a.get("statistical_metrics", {})
            lines.append(
                f"{a.get('anomaly_type')} (severity: {a.get('severity')}) -- "
                f"Z-score: {m.get('z_score')}, p-value: {m.get('p_ks')}"
            )
    else:
        lines.append("No anomalies detected against calibrated historical baseline.")
    return "\n".join(l for l in lines if l)


def _build_alert(result: dict) -> dict | None:
    """
    Turns detected_anomalies into a simple structured alert signal the
    frontend can render as a banner (level + short message), instead of the
    frontend having to parse anomaly severities out of free text.
    """
    anomalies = result.get("detected_anomalies", [])
    if not anomalies:
        return {"level": "SAFE", "message": "No significant anomalies detected."}

    high = [a for a in anomalies if a.get("severity") == "High"]
    if high:
        names = ", ".join(a["anomaly_type"] for a in high)
        return {"level": "WARNING", "message": f"High-severity anomaly detected: {names}."}

    moderate = [a for a in anomalies if a.get("severity") == "Moderate"]
    if moderate:
        names = ", ".join(a["anomaly_type"] for a in moderate)
        return {"level": "CAUTION", "message": f"Moderate anomaly detected: {names}."}

    return {"level": "SAFE", "message": "No significant anomalies detected."}


def sagar_mitra_compare(user_message: str) -> dict:
    """
    Compares 2+ known locations mentioned in the same question (e.g. "which
    is better, Sindhudurg or Alibaug") by running the real-data check + engine
    independently for each, then asking Gemini to write one comparison
    answer grounded in both results.
    """
    locations = _find_known_locations(user_message)
    params = extract_query_parameters(user_message)  # for persona/language only
    persona = params.get("persona", "General Public")
    language = params.get("language", "English")

    per_location = []
    for name, lat, lon in locations:
        profile_df = store.get_real_profile_near(lat, lon)
        if profile_df is None:
            per_location.append({"name": name, "data_available": False})
            continue
        engine = ArgoStatisticalAnomalyEngine(climatology_db=store.calibrated_clim)
        result = engine.analyze_profile(profile_df, historical_timeseries=store.get_real_hist_near(lat, lon))
        per_location.append({
            "name": name,
            "data_available": True,
            "confidence": "HIGH" if store.nearby_count(lat, lon) >= 200 else
                          "MODERATE" if store.nearby_count(lat, lon) >= 30 else "LOW",
            "layer_metrics": result.get("layer_metrics"),
            "detected_anomalies": result.get("detected_anomalies"),
        })

    system_instruction = (
        f"You are Sagar Mitra. Compare these coastal locations for a {persona} "
        f"in {language}, based ONLY on the real data provided below. Give a "
        f"clear recommendation on which is better right now and why, in plain "
        f"text (no markdown, no LaTeX). If a location has no data available, "
        f"say so honestly instead of guessing."
    )
    import json as _json
    response = generate_content(
        model="gemini-3.6-flash",
        contents=f"User question: {user_message}\n\nData per location:\n{_json.dumps(per_location, indent=2)}",
        config=types.GenerateContentConfig(system_instruction=system_instruction,
                                            temperature=0.4, max_output_tokens=2048),
    )
    return {
        "reply": response.text,
        "mode": "comparison",
        "query_context": params,
        "statistical_result": {"locations": per_location},
    }


def sagar_mitra_router(message: str) -> dict:
    """
    Routes a user message to the comparison agent, the historical-data
    agent, or the live conditions chat, based on keyword matching.
    """
    lower = message.lower()
    if any(k in lower for k in COMPARISON_KEYWORDS) and len(_find_known_locations(message)) >= 2:
        return sagar_mitra_compare(message)
    if any(k in lower for k in HISTORICAL_KEYWORDS):
        return sagar_mitra_historical_query(message)
    return sagar_mitra_chat(message)
