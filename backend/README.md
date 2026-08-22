# Sagar Mitra Backend

## Setup
```
pip install -r requirements.txt
cp .env.example .env   # then paste your GEMINI_API_KEY inside
```
Cache real data once: save your notebook's `final_dataset` as
`data/final_dataset.parquet` (see app/data.py). Without it, boots on
synthetic data automatically so frontend integration isn't blocked.

## Run
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Test
```
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"Ratnagiri jawal udya machimari surakshit aahe ka?"}'
```
Response: `{reply, mode, query_context, statistical_result, plot_url}`.
`mode` is one of `live_conditions` / `fallback_general_knowledge` / `historical`.
