import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL", "https://trademarkia-api.onrender.com/query")

st.title("Semantic Search Engine")
st.write("Search the 20 Newsgroups dataset")

query = st.text_input("Enter your query")

if st.button("Search"):

    if query:
        response = requests.post(API_URL, json={"query": query})

        if response.status_code == 200:
            st.subheader("Search Result")

            st.markdown("---")
            st.write(data["result"])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Similarity Score", round(data["similarity_score"], 3))

            with col2:
                st.metric("Cache Hit", data["cache_hit"])

            with col3:
                st.metric("Cluster", data["dominant_cluster"])

            st.markdown("---")

        else:
            st.error("API request failed")