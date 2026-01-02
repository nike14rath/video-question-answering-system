import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# CONFIG
# --------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_K = 5

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_resource
def load_dataframe():
    return joblib.load("Dataframe_with_embeddings.joblib")

df = load_dataframe()

# --------------------------------
# FUNCTIONS
# --------------------------------
def create_embeddings(text_list):
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text_list},
        timeout=120
    )
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return r.json()["response"]

# --------------------------------
# STREAMLIT UI
# --------------------------------
st.set_page_config(page_title="🎥 Video Q&A System", layout="wide")

st.title("🎥 Video Question Answering System")
st.markdown("Ask questions and get **video + timestamp based answers**")

query = st.text_input("🔍 Ask your question related to videos:")

if st.button("Search") and query.strip():

    with st.spinner("Creating query embedding..."):
        query_embedding = create_embeddings([query])[0]

    with st.spinner("Finding relevant video segments..."):
        similarity = cosine_similarity(
            np.vstack(df["embedding"]),
            [query_embedding]
        ).flatten()

        top_indices = similarity.argsort()[::-1][:TOP_K]
        results_df = df.iloc[top_indices]

    # st.subheader("📌 Top Relevant Video Segments")
    # for idx, row in results_df.iterrows():
    #     st.markdown(
    #         f"""
    #         **📺 Video:** `{row['Video']}`  
    #         ⏱ **Time:** {row['start']:.2f} min → {row['end']:.2f} min  
    #         📝 **Text:** {row['text']}
    #         ---
    #         """
    #     )

    prompt = f"""Here are the Vedios containting the text, Vedio, Time Stamp: {results_df[['Video','start', 'end', "text"]].to_json()}
The User asked this specific Questions related to Vedios,give the top three results in easy to navigate format if possible, you have the Answer how much content is taught, in which vedio title and at what Time Stamp(give the time in minutes formatW), Guide the User to that Particular Vedio. If User asked unrealted Questions tell them, that you can answer questions related to vedio only: {query}
"""

    with open("prompt.txt", "w") as f:
        f.write(prompt)

    with st.spinner("Generating answer using LLM..."):
        response = inference(prompt)

    with open("response.txt", "w") as f:
        f.write(response)

    st.subheader("🤖 Answer")
    st.write(response)

else:
    st.info("Enter a question and click **Search**")
