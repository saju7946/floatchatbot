import streamlit as st
import numpy as np
import faiss
import requests
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

# 🔐 ADD YOUR KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
client = Groq(api_key=GROQ_API_KEY)
if GROQ_API_KEY is None:
    st.error("GROQ API key not found.")
    st.stop()

st.set_page_config(page_title="Float Chat AI", page_icon="🌊", layout="wide")

# =============================
# 🎨 MODERN UI HEADER
# =============================
st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(90deg,#38bdf8,#0ea5e9);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
🌊 Float Chat AI Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:gray;'>AI-Powered Ocean Intelligence + Climate Analytics</p>", unsafe_allow_html=True)

# =============================
# LOAD DATASET
# =============================
@st.cache_resource
def load_sea_data():
    df = pd.read_csv("sea_levels_2015.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    df["Year"] = df["Time"].dt.year
    return df

sea_df = load_sea_data()

# =============================
# LOAD EMBEDDING MODEL
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =============================
# LOAD KNOWLEDGE + FAISS
# =============================
@st.cache_resource
def load_knowledge():
    with open("ocean_knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    documents = text.split("\n\n")
    embeddings = model.encode(documents)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return documents, index

documents, index = load_knowledge()

# =============================
# IMAGE FETCH
# =============================
def fetch_image(query):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query + " ocean",
        "client_id": UNSPLASH_ACCESS_KEY,
        "per_page": 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "results" in data and len(data["results"]) > 0:
        return data["results"][0]["urls"]["regular"]
    return None

# =============================
# TABS
# =============================
tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📊 Climate Dashboard", "📈 Prediction"])

# =====================================================
# 💬 TAB 1 — CHATBOT
# =====================================================
with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("Ask anything about oceans...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        query_vector = model.encode([query])
        D, I = index.search(np.array(query_vector), k=3)
        context = "\n".join([documents[i] for i in I[0]])

        prompt = f"""
        You are an ocean expert AI.
        Answer scientifically using context below.

        Context:
        {context}

        Question:
        {query}
        """

        with st.spinner("Thinking... 🌊"):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            image_url = fetch_image(msg["content"])
            if image_url:
                st.image(image_url, use_container_width=True)
            st.markdown(f"**🌊 Float Chat AI:** {msg['content']}")

# =====================================================
# 📊 TAB 2 — CLIMATE DASHBOARD
# =====================================================
with tab2:

    st.subheader("🌊 Animated Global Sea Level Rise")

    fig = px.line(
        sea_df,
        x="Time",
        y="GMSL",
        title="Global Mean Sea Level Over Time",
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Sea Level (mm)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡 Simulated Temperature vs Sea Level")

    # Simulated temperature trend
    sea_df["Temp"] = np.linspace(14, 15.5, len(sea_df))

    fig2 = px.scatter(
        sea_df,
        x="Temp",
        y="GMSL",
        title="Temperature vs Sea Level Correlation",
        trendline="ols"
    )

    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# 📈 TAB 3 — PREDICTION
# =====================================================
with tab3:

    st.subheader("📈 Sea Level Prediction (Linear Regression)")

    # Prepare data
    X = sea_df["Year"].values.reshape(-1, 1)
    y = sea_df["GMSL"].values

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_years = np.arange(sea_df["Year"].max(), sea_df["Year"].max() + 20).reshape(-1, 1)
    future_pred = model_lr.predict(future_years)

    future_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted_GMSL": future_pred
    })

    fig3 = px.line(
        future_df,
        x="Year",
        y="Predicted_GMSL",
        title="Predicted Sea Level Rise (Next 20 Years)"
    )

    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    st.info("Prediction is based on simple linear regression for demonstration purposes.")
