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
            data = response.json()
            st.subheader("Result")

            st.write(data["result"])
            st.write("Similarity score:", data["similarity_score"])
            st.write("Cache hit:", data["cache_hit"])
            st.write("Cluster:", data["dominant_cluster"])

        else:
            st.error("API request failed")