import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import streamlit as st
from dotenv import load_dotenv
import re
import os
import json

# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j connection
@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_neo4j_driver()

# Load dataset
file_path = "./data/flipkart_fashion_products_dataset.json"
df = pd.read_json(file_path).head(10000)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"\W", " ", str(text))
    return text.lower()

df["searchable_text"] = (
    df["title"].astype(str) + " " + df["description"].astype(str) + " " + df["brand"].astype(str)
).apply(preprocess_text)

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Caching embeddings
if os.path.exists("embeddings_cache.json"):
    with open("embeddings_cache.json", "r") as f:
        df["embeddings"] = pd.Series(json.load(f))
else:
    df["embeddings"] = df["searchable_text"].apply(lambda x: model.encode(x).tolist())
    with open("embeddings_cache.json", "w") as f:
        json.dump(df["embeddings"].tolist(), f)

print("Embeddings computed and dataset preprocessed.")

# Query expansion using Neo4j
def query_expansion(query):
    synonyms = []
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Keyword {name: $query})-[:RELATED_TO]->(related)
            RETURN related.name AS synonym
            """,
            {"query": query.lower()},
        )
        synonyms = [record["synonym"] for record in result]
    return synonyms

# Search products with filters
def search_products_with_filters(query, category=None, tag=None, top_n=5):
    expanded_queries = [query] + query_expansion(query)
    query_embedding = model.encode(expanded_queries, convert_to_tensor=True).cpu()
    filtered_df = df
    if category and category != "All":
        filtered_df = filtered_df[filtered_df["category"].astype(str) == category]
    if tag:
        filtered_df = filtered_df[filtered_df["brand"].astype(str).str.contains(tag, case=False, na=False)]
    if filtered_df.empty:
        return pd.DataFrame()
    product_embeddings = np.vstack(filtered_df["embeddings"].tolist())
    similarities = cosine_similarity(query_embedding.numpy(), product_embeddings)
    mean_similarities = similarities.mean(axis=0)
    top_indices = mean_similarities.argsort()[-top_n:][::-1]
    return filtered_df.iloc[top_indices][["title", "description", "category", "brand"]]

# Custom CSS for UI Beautification
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    /* Background image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1509631179647-0177331693ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main container styling */
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Title styling */
    h1 {
        font-family: 'Montserrat', sans-serif;
        color: #1a1a1a;
        text-align: center;
        font-size: 2.5em;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Subheader and text styling */
    h2, p {
        font-family: 'Montserrat', sans-serif;
        color: #333;
    }

    /* Button styling */
    .stButton>button {
        background-color: #ff4b5c;
        color: white;
        border-radius: 5px;
        font-family: 'Montserrat', sans-serif;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e04352;
    }

    /* Input field styling */
    .stTextInput>div>input {
        border: 2px solid #ff4b5c;
        border-radius: 5px;
        font-family: 'Montserrat', sans-serif;
    }

    /* Selectbox styling */
    .stSelectbox>div>div {
        border: 2px solid #ff4b5c;
        border-radius: 5px;
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title("VogueVista: Your Fashion Search Haven")

# User input in a container
with st.container():
    query = st.text_input("Enter your fashion quest:")
    category = st.selectbox(
        "Filter by category:", ["All"] + df["category"].astype(str).unique().tolist()
    )
    tag = st.text_input("Filter by brand (optional):")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            try:
                results = search_products_with_filters(
                    query, category=None if category == "All" else category, tag=tag
                )
                if results.empty:
                    st.info("No fabulous finds for now.")
                else:
                    st.write(f"Showing top {len(results)} stylish picks:")
                    for _, row in results.iterrows():
                        st.subheader(row["title"])
                        st.write(f"**Category**: {row['category']}")
                        st.write(f"**Description**: {row['description']}")
                        st.write(f"**Brand**: {row['brand']}")
                        st.write("---")
            except Exception as e:
                st.error(f"Oops, something went wrong: {e}")

# Close Neo4j connection
if st.button("Close Connection"):
    driver.close()
    st.write("Neo4j connection closed with flair.")