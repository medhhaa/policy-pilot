import streamlit as st
from retrieval import build_faiss_index, query_faiss
from llm_agent import answer_query

st.set_page_config(page_title="policy‑pilot")
st.title("🧭 policy‑pilot: Compliance Q&A")

if st.button("🔨 Build / Refresh Index"):
    with st.spinner("Indexing chunks…"):
        build_faiss_index()
    st.success("Index built!")

query = st.text_input("Ask a compliance question…")
if st.button("💬 Get Answer"):
    if not query:
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving context…"):
            chunks = query_faiss(query, top_k=3)
        st.subheader("🔍 Retrieved Chunks")
        for i, c in enumerate(chunks, 1):
            st.write(f"**{i}.** {c['text']}")
        with st.spinner("Generating answer…"):
            answer = answer_query(query, chunks)
        st.subheader("🤖 policy‑pilot Answer")
        st.write(answer)
