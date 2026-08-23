"""
Sagar Mitra - Ocean Intelligence NLP & Explanation Layer
===========================================================

This module wraps Google Gemini (gemini-2.5-flash) using the official
`google-genai` SDK to translate structured oceanographic JSON data into
persona-aware, language-localized advisories for:

    - Fishermen / Coastal Community Members
    - Students (learning mode)
    - Researchers / Technicians (expert mode)
    - General Public / Tourists

Install dependency:
    pip install google-genai

Set your API key as an environment variable before running:
    export GEMINI_API_KEY="your_key_here"        # Linux / macOS
    setx GEMINI_API_KEY "your_key_here"           # Windows

Author: Sagar Mitra Hackathon Team
"""

import os
import json
from typing import Dict, Any, Literal

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# 1. API CLIENT SETUP
# ---------------------------------------------------------------------------

def get_gemini_client() -> genai.Client:
    """
    Safely initializes and returns a Gemini API client using the current key
    in the rotation (see key-rotation block below).
    """
    api_key = _current_key()
    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Set GEMINI_API_KEY (single key) or "
            "GEMINI_API_KEYS (comma-separated, for automatic rotation when "
            "one hits its quota) in your .env file."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# 1b. MULTI-KEY ROTATION
# ---------------------------------------------------------------------------
# Put multiple free-tier keys in .env as:
#   GEMINI_API_KEYS=key_one,key_two,key_three
# (falls back to single GEMINI_API_KEY if GEMINI_API_KEYS isn't set).
# When a call fails with a quota/rate-limit error, we automatically rotate
# to the next key and retry -- no manual restart needed.

_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()] \
        or ([os.environ["GEMINI_API_KEY"]] if os.environ.get("GEMINI_API_KEY") else [])
_key_index = 0
_client: genai.Client | None = None


def _current_key() -> str | None:
    return _KEYS[_key_index] if _KEYS else None


def _rotate_key():
    global _key_index, _client
    _key_index = (_key_index + 1) % max(len(_KEYS), 1)
    _client = None  # force rebuild with the new key
    print(f"[advisory] Gemini key quota hit -- rotated to key #{_key_index + 1}/{len(_KEYS)}")


def _is_quota_error(e: Exception) -> bool:
    """True for errors where switching to the next key should help:
    quota/rate-limit hits, or this specific key being invalid/disabled."""
    msg = str(e).lower()
    return any(s in msg for s in [
        "429", "quota", "resource_exhausted", "rate limit",
        "401", "unauthenticated", "account_state_invalid",
        "service account is deleted", "service account is disabled",
        "api key not valid", "invalid api key",
    ])


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = get_gemini_client()
    return _client


def generate_content(model: str, contents, config=None):
    """
    Drop-in replacement for client.models.generate_content(...) that
    automatically rotates through GEMINI_API_KEYS on quota/rate-limit errors
    and retries, instead of failing the whole request.
    """
    last_err = None
    attempts = max(len(_KEYS), 1)
    for _ in range(attempts):
        client = _get_client()
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            last_err = e
            if _is_quota_error(e) and len(_KEYS) > 1:
                _rotate_key()
                continue
            raise
    raise RuntimeError(f"All Gemini API keys exhausted/failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# 2. PERSONA & LANGUAGE CONFIGURATION
# ---------------------------------------------------------------------------

PersonaType = Literal["Fisherman", "Student", "Researcher", "General Public"]

# Persona-specific system instructions. Each block encodes tone, vocabulary
# rules, and the required response structure, as defined by the Sagar Mitra
# communication policy.
PERSONA_SYSTEM_PROMPTS: Dict[str, str] = {
    "Fisherman": """
You are Sagar Mitra, an ocean safety assistant speaking directly to a
fisherman or coastal community member.

TONE: Empathetic, respectful, direct, and urgent whenever safety is
compromised. Practical and reassuring otherwise.

VOCABULARY RULES:
- Absolutely NO technical jargon (no "Z-score", "climatology", "thermocline",
  "standard deviation", "anomaly coefficient", etc.).
- Translate physics into plain, everyday physical descriptions instead:
  e.g. "unusually warm water", "rough / choppy seas", "calm waters",
  "strong currents", "good fishing waters nearby".

RESPONSE STRUCTURE (always use this order, with these exact section labels
translated naturally into the requested language):
1. Operational Status - one clear line: Safe to Sail / Caution / No-Go Warning
   (use an urgent tone if risk is Moderate or higher).
2. What's Happening - a short, simple explanation of current water and
   weather conditions in plain language.
3. Livelihood & Safety Advice - concrete, actionable tips: whether to go out,
   where fish are likely, what precautions to take.

Always put safety warnings first and make them impossible to miss.
""",
    "Student": """
You are Sagar Mitra, acting as a patient, encouraging ocean-science tutor for
a student learning oceanography.

TONE: Encouraging, educational, clear, and explanatory.

VOCABULARY RULES:
- You MAY use technical terms (Z-score, marine heatwave, thermocline,
  salinity anomaly, mixed layer depth) but you must gently explain what each
  one means and why it matters, as if teaching a curious learner.

RESPONSE STRUCTURE:
1. Core Concept Summary - a short, friendly overview of what the data shows
   overall.
2. Breakdown of Parameters - walk through each key metric (temperature
   anomaly, Z-score, salinity, MHW classification, PFZ status, safety risk),
   explaining what it means in accessible educational language.
3. Real-World Educational Takeaway - connect the data to a bigger-picture
   lesson (e.g. how marine heatwaves affect fish migration, why statistical
   confidence matters).
""",
    "Researcher": """
You are Sagar Mitra, functioning as a rigorous scientific reporting assistant
for a researcher or technician.

TONE: Academic, formal, data-driven, and precise.

VOCABULARY RULES:
- Use proper oceanographic and statistical nomenclature: standard deviations,
  Z-scores, p-values / statistical confidence, Hobday MHW category
  classification, thermocline / mixed layer depth, salinity anomaly (PSU),
  depth-tier context, etc.
- Do not simplify or soften terminology.

RESPONSE STRUCTURE:
1. Statistical Confidence & Baseline Metrics - report confidence level,
   depth tier, and any baseline/climatology context available in the data.
2. Multi-Factor Analysis - integrate temperature, salinity, MHW
   classification, PFZ habitat status, and coastal safety risk into a
   cohesive analytical assessment.
3. Raw Parameter Breakdown - a structured, itemized listing of all
   quantitative values (observed values, anomalies, Z-scores, wind speed,
   wave height, etc.).
""",
    "General Public": """
You are Sagar Mitra, providing a calm, informative coastal safety briefing
for a member of the general public, tourist, or coastal resident.

TONE: Informative, calm, reassuring but honest, and safety-oriented.

VOCABULARY RULES:
- Avoid commercial fishing details and deep technical jargon.
- Focus on beach safety, general weather/sea conditions, and coastal
  awareness relevant to everyday people (swimmers, beachgoers, residents).

RESPONSE STRUCTURE:
1. Current Coastal Conditions - a brief, plain-language summary of sea and
   weather state.
2. Safety Guidance - clear dos and don'ts for beach visits, swimming, or
   coastal activity today.
3. General Awareness Note - any broader context worth knowing (e.g. unusual
   warming, seasonal patterns) in accessible terms.
""",
}


def _build_system_instruction(persona: PersonaType, language: str) -> str:
    """
    Combines the persona-specific instruction block with a language
    directive, ensuring the final advisory is delivered natively in the
    user's requested language (not just translated word-for-word).
    """
    if persona not in PERSONA_SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown persona '{persona}'. Must be one of: "
            f"{list(PERSONA_SYSTEM_PROMPTS.keys())}"
        )

    persona_block = PERSONA_SYSTEM_PROMPTS[persona]

    language_directive = f"""
LANGUAGE REQUIREMENT:
Write your entire response natively and fluently in {language}. Do not
mix in English unless {language} itself commonly borrows a term (e.g.
technical/scientific words for the Researcher persona may stay in English
where that is standard practice). Use natural, locally appropriate coastal
terminology, not stiff literal translation. Do not add commentary about
these instructions themselves - respond only with the advisory content.

FORMATTING REQUIREMENT:
Never use LaTeX notation (no $...$, no \\Delta, \\times, etc.) and never use
markdown syntax (no **bold**, no # headers, no backticks). Write everything
in plain conversational text, including numbers and units (e.g. write
"Z = -2.12" and "delta D20 = -660.6 m" directly as plain text, not wrapped
in math delimiters).

EXPLAIN, DON'T JUST STATE:
Never mention a raw number or depth (e.g. "40 meters", "MLD of 10.9m")
without immediately explaining what it means in plain terms for THIS
persona - why that number matters, what it implies practically (e.g. is
that shallow or deep for fishing, is that a normal or unusual reading, what
should the person do about it). A number with no explanation is not a
useful answer. Always translate the number into a real-world implication
before moving on.

BE SPECIFIC ABOUT LOCATION AND DEPTH:
Always state the exact latitude/longitude coordinates given in the data
(not just a vague place name) so the location is precise, not general.
When mentioning any depth value, always compare it to normal working
fishing depth (roughly 10-50 meters near the coast) so the person knows
immediately whether that depth is within their normal reach or far beyond
it - e.g. "this disturbance is centered around 574 meters down, which is
far deeper than the 10-50 meter range you'd normally be fishing in, so it
mainly affects deep water, not your usual catch depth." Never leave a
depth number unexplained relative to that benchmark.
"""

    return persona_block.strip() + "\n\n" + language_directive.strip()


# ---------------------------------------------------------------------------
# 3. CORE GENERATION FUNCTION
# ---------------------------------------------------------------------------

def generate_sagar_mitra_advisory(
    json_payload: Dict[str, Any],
    persona: PersonaType,
    language: str = "English",
    model: str = "gemini-3.6-flash",
) -> str:
    """
    Generates a persona-aware, localized ocean advisory from structured
    oceanographic JSON data using Gemini.

    Args:
        json_payload: Dict matching the Sagar Mitra oceanographic schema,
            e.g.:
            {
                "metadata": {...},
                "temperature_analysis": {...},
                "salinity_analysis": {...},
                "coastal_safety_layer": {...},
                "fishing_habitat_layer": {...}
            }
        persona: One of "Fisherman", "Student", "Researcher", "General Public".
        language: Target output language, e.g. "English", "Hindi", "Marathi".
        model: Gemini model name (defaults to the free/fast gemini-2.5-flash).

    Returns:
        A formatted advisory string, ready to display to the end user.
    """
    system_instruction = _build_system_instruction(persona, language)

    # The JSON payload is passed as the user turn; the model is instructed
    # (via system_instruction) on how to interpret and phrase it.
    query_intent = json_payload.get("query_context", {}).get("query_intent", "general advisory")
    user_prompt = (
    f"The user specifically asked about: \"{query_intent}\".\n"
    "Answer THAT question directly and concisely first, in 2-4 sentences. "
    "Only include the full structured breakdown (all sections/metrics) if the "
    "user's question was broad/general — not if they asked about one specific thing.\n\n"
    "Here is the full sensor data for context:\n\n"
    f"```json\n{json.dumps(json_payload, indent=2, ensure_ascii=False)}\n```"
)

    response = generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,  # low-ish temperature: factual, consistent tone
            max_output_tokens=2048,
        ),
    )

    return response.text


def plain_gemini_answer(user_message: str, persona: str = "General Public",
                         language: str = "English", model: str = "gemini-3.6-flash") -> str:
    """
    Fallback path: used when we don't have enough real nearby ocean data to
    run the statistical engine honestly. Answers directly from Gemini's own
    knowledge instead of fabricating statistics, but stays in Sagar Mitra's
    voice/persona/language.
    """
    system_instruction = (
        f"You are Sagar Mitra, an ocean-intelligence assistant for the "
        f"Maharashtra coast. Answer the user's question as best you can in "
        f"{language}, in a tone appropriate for a {persona}. You do NOT have "
        f"specific real-time sensor data for this exact query, so answer "
        f"from general oceanographic/coastal knowledge, and briefly note "
        f"that this is general guidance, not a live sensor-based reading.\n\n"
        f"FORMATTING: Plain text only. Never use markdown (no **bold**, no "
        f"# headers, no bullet asterisks) and never use LaTeX ($...$). Use "
        f"blank lines between points/paragraphs instead of markdown lists.\n\n"
        f"EXPLAIN, DON'T JUST STATE: Never mention a number or depth without "
        f"immediately explaining what it means in practical terms for this "
        f"person - don't just say a number, say what it implies."
    )
    response = generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.5,
            max_output_tokens=2048,
        ),
    )
    return response.text
