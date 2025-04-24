# === rag_utils.py ===

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re

# === CONFIG ===
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOGETHER_API_TOKEN = "tgp_v1_aZfuZPrKodkoRGR70eDQwZE_jqfeL5UwJEBYIccBylc"
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# === Loaders ===
def load_faiss_index(index_path="faissindexIDB.index"):
    return faiss.read_index(index_path)

def load_metadata(metadata_path="metadataIDB.json"):
    with open(metadata_path, "r") as f:
        return json.load(f)

def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

# === RAG Search ===
def search_index(query, embed_model, index, metadata, top_k=3):
    query_vector = embed_model.encode(query).astype(np.float32)
    D, I = index.search(np.array([query_vector]), top_k)
    return [metadata[i] for i in I[0]]

# === SQL Generation ===
def generate_sql_from_nl(query, table_name="dbo.Inventory"):
    query = query.lower()
    
    # Standardizing certain queries
    if "who are the employees" in query:
        query = "who are all the employees in the warehouse"

    prompt = f"""<s>[INST]
You are a T-SQL expert. Generate a valid T-SQL Server query using the table Inventory with the following columns:
entry_id, item_name, item_category, unit_price, quantity, action_type, employee_name, employee_role, shift, log_timestamp, warehouse_location, reason, supplier_name, customer_name, status

### SQL Query Guidelines:
- Do NOT explain the query.
- Do NOT use markdown or comments.
- Do NOT say "Here is the SQL..."
- Do NOT add anything outside the SQL.
- Use TOP properly after SELECT, NOT after ORDER BY.
- Do NOT use LIMIT. Only use TOP for row limits.
- Do NOT use HAVING unless there's an aggregation.
- Do NOT use the TOP keyword unless it is needed to limit the result set.
- Ensure LIKE '%Electronics%' is used instead of category = 'Consumer Electronics'.
- If fetching the oldest inventory, use ORDER BY date_received ASC with TOP 1.

### User Question:
{query}

### SQL:
[/INST]"""


    headers = {
        "Authorization": f"Bearer {TOGETHER_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.3
    }

    response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

    if response.status_code == 200:
        raw = response.json()["choices"][0]["text"].split("[/INST]")[-1]
        sql = re.split(r";|--|#", raw.strip())[0]
        sql = re.sub(r"[^a-zA-Z0-9_.*,()=' \n><-]", "", sql)
        sql = sql.replace("\n", " ").strip()
        return sql
    else:
        return None


# === LLM Answer Generation ===
def generate_answer(query, context):
    prompt = f"""<s>[INST] You are a helpful assistant. Use the context to answer user's question.
- Use markdown and bullet points
- Be concise and clear

Context:
{context}

Question:
{query}

Answer:
[/INST]"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": 0.7
    }

    response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["text"].split("[/INST]")[-1].strip()
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"
