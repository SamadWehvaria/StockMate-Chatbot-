import streamlit as st
import csv
import os
import re
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO

from rag_utils import (
    load_faiss_index,
    load_metadata,
    load_embedder,
    search_index,
    generate_answer,
    generate_sql_from_nl
)
from DB_connect import run_sql_query

# === CONFIG ===
LOG_FILE = "chatbot_logs.csv"

# === Init Log File ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerow(["timestamp", "question", "answer", "feedback"])

# === Streamlit Page ===
st.set_page_config(page_title="Chatbot", page_icon="💬")

st.title("🤖 StockMate")
st.markdown("Ask anything about inventory policies or inventory data.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# === Load RAG Components ===
@st.cache_resource
def load_rag():
    return load_faiss_index(), load_metadata(), load_embedder()

index, metadata, embedder = load_rag()

# === Small Talk Handler (Regex + LLM Hybrid) ===
def is_small_talk(query):
    small_talk_patterns = [
        r"\b(hi|hello|hey|how are you|how’s it going|what's up|good morning|good evening|how are you doing|how r u|what's new|what's happening|good night)\b",
        r"\b(thank(s| you)|bye|see you|take care)\b",
        r"\b(what are you doing|did you eat|have you eaten|how's your day|what time is it|who made you)\b"
    ]
    return any(re.search(pattern, query.lower()) for pattern in small_talk_patterns)

# === Chat Input ===
query = st.chat_input("Ask me anything...")

if query:
    with st.spinner("Thinking... 🤔"):
        st.session_state.query_count += 1

        # === Handle Small Talk ===
        if is_small_talk(query):
            st.session_state.chat_history.append({
                "question": query,
                "answer": "Hello! I'm a chatbot, and I'm here to help! Ask me anything about inventory data or inventory policies.",
                "chunks": []
            })
            st.rerun()

        # === RAG: Retrieve context from PDF chunks
        top_chunks = search_index(query, embedder, index, metadata, top_k=3)
        policy_context = "\n\n".join([chunk["content"] for chunk in top_chunks]) if top_chunks else ""

        # === Attempt SQL Query
        sql_result = ""
        try:
            sql_query = generate_sql_from_nl(query)
            if sql_query:
                result_df = run_sql_query(sql_query)
                if isinstance(result_df, str):
                    sql_result = f"❌ SQL Error:\n\n{result_df}"
                elif not result_df.empty:
                    sql_result = f"🗃️ From Database:\n\n{result_df.to_markdown(index=False)}"
                else:
                    sql_result = "ℹ️ No matching records found in the database."
            else:
                sql_result = "❌ Could not generate SQL query."
        except Exception as e:
            sql_result = f"❌ Error running SQL query: {e}"

        # === Generate Answer from Policy
        llm_answer = generate_answer(query, policy_context) if policy_context else ""

        # === Combine Results
        final_answer = ""
        if sql_result:
            final_answer += sql_result
        if llm_answer:
            final_answer += f"\n\n📄 From Policy Document:\n\n{llm_answer}"

        st.session_state.chat_history.append({
            "question": query,
            "answer": final_answer.strip(),
            "chunks": top_chunks
        })

# === Display Chat History ===
for i, chat in enumerate(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {chat['question']}")
    st.markdown(f"**🤖 Bot:** {chat['answer']}")

    if chat["chunks"]:
        with st.expander("📄 View Policy Chunks"):
            for j, chunk in enumerate(chat["chunks"]):
                st.markdown(f"**Chunk {j+1} from `{chunk['source']}`**")
                st.write(chunk["content"])

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍 Helpful", key=f"like_{i}"):
            feedback = "👍 Helpful"  # Changed to a simple string
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f, quoting=csv.QUOTE_ALL).writerow([datetime.now(), chat["question"], chat["answer"], feedback])
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 Not Useful", key=f"dislike_{i}"):
            feedback = "👎 Not Useful" # Changed to a simple string
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f, quoting=csv.QUOTE_ALL).writerow([datetime.now(), chat["question"], chat["answer"], feedback])
            st.warning("Thanks for your feedback!")

    st.divider()

# === Sidebar Stats ===
with st.sidebar:
    st.markdown("### 📊 Chatbot Stats")
    st.markdown(f"**Total Queries:** `{st.session_state.query_count}`")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_threshold = datetime.now() - timedelta(days=30)
        recent_feedback = df[df['timestamp'] >= time_threshold]

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            recent_feedback.to_excel(writer, index=False, sheet_name='Feedback')
            workbook = writer.book
            worksheet = writer.sheets['Feedback']
            for col in worksheet.columns:
                max_length = max(len(str(cell.value or '')) for cell in col)
                worksheet.column_dimensions[col[0].column_letter].width = max_length + 2

        output.seek(0)
        st.download_button(
            "⬇️ Download Feedback Excel",
            output,
            file_name="chatbot_logs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
