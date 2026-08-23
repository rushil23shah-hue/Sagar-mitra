"""
Sagar Mitra - Phase 1 NLP: User Query -> Structured Parameters
=================================================================
Takes free-text user input (any language) and extracts the structured
parameters needed to run the Statistical Anomaly Engine and Phase 2
(advisory generation).
"""
import json
import re
from datetime import date

from google.genai import types
from app.advisory import generate_content

EXTRACTION_SYSTEM_PROMPT = """
You are the input-parsing layer for Sagar Mitra, an ocean intelligence system
for the Maharashtra coast (Arabian Sea).

Extract structured query parameters from the user's message, regardless of
language (English/Hindi/Marathi/Konkani). Respond with ONLY valid JSON, no
markdown, no explanation, matching this exact schema:

{
  "location_name": "<place name mentioned, or nearest known coastal town if implied, else null>",
  "latitude": <float, best-guess coordinates for that Maharashtra coastal location, else null>,
  "longitude": <float, else null>,
  "date": "<YYYY-MM-DD if mentioned/implied (e.g. 'tomorrow'), else null>",
  "depth_m": <float, fishing/query depth if mentioned, else 10.0 as default surface>,
  "persona": "<one of: Fisherman, Student, Researcher, General Public - infer from tone/question, default Fisherman>",
  "language": "<detected input language name, e.g. Marathi, Hindi, English>",
  "query_intent": "<short phrase: e.g. 'fishing safety check', 'general curiosity', 'research query'>"
}

Known Maharashtra coastal reference points (use if user names these or nearby areas):
Mumbai (19.07,72.87), Koliwada/fishing village areas generically (19.07,72.87 - same as Mumbai,
since "Koliwada" without further qualification usually refers to a Mumbai coastal fishing hamlet),
Ratnagiri (16.99,73.30), Sindhudurg (16.02,73.45), Raigad/Alibaug (18.64,72.87),
Palghar (19.70,72.77), Vengurla (15.85,73.63).

If the user mentions ANY place name at all (even a generic or locally-known one like
"Koliwada", a creek, a fishing village, a beach), match it to the closest known
reference point above rather than leaving latitude/longitude null or silently
defaulting elsewhere. Only use the true default (Ratnagiri coordinates) when
absolutely NO location of any kind is mentioned in the message.
If no date, use today.
Today's date is {today}.
"""


def extract_query_parameters(user_message: str, model: str = "gemini-3.6-flash") -> dict:
    """
    Phase 1 NLP: Converts a free-text user query into structured parameters
    for the Statistical Anomaly Engine.
    """
    system_instruction = EXTRACTION_SYSTEM_PROMPT.replace("{today}", date.today().isoformat())

    response = generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=2048,
        ),
    )

    text = response.text.strip()
    text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse extraction output: {text}")
