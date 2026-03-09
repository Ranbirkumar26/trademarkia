import streamlit as st
import requests

API_URL = "http://localhost:8001/query"

st.title("Semantic Search Engine")
st.write("Search the 20 Newsgroups dataset")

query = st.text_input("Enter your query")

if st.button("Search"):

    if query:
        response = requests.post(API_URL, json={"query": query})

        if response.status_code == 200:
            data = response.json()

            st.subheader("Results")

            for i, r in enumerate(data["results"]):
                st.write(f"### Result {i+1}")
                st.write(r)
                st.write("---")

        else:
            st.error("API request failed")