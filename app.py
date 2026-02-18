import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faiss
import requests
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================
# 🔐 LOAD API KEYS
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not GROQ_API_KEY:
    st.error("GROQ API key not found. Please check Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Float Chat AI", page_icon="🌊", layout="wide")

# =============================
# 🎨 MODERN HEADER
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
# 📂 LOAD DATASET
# =============================
@st.cache_resource
def load_sea_data():
    df = pd.read_csv("sea_levels_2015.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    df["Year"] = df["Time"].dt.year
    return df

sea_df = load_sea_data()

# =============================
# 🧠 LOAD EMBEDDING MODEL
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =============================
# 📚 LOAD KNOWLEDGE + FAISS
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
# 🖼 FETCH IMAGE FROM UNSPLASH
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
# 📑 TABS
# =============================
tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📊 Climate Dashboard", "📈 Prediction"])

# =====================================================
# 💬 TAB 1 — CHATBOT (UPDATED WITH IMAGE FEATURE)
# =====================================================
with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("Ask anything about oceans...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        # 🔥 NEW IMAGE FEATURE ADDED
        image_keywords = ["show", "image", "photo", "picture", "display"]

        if any(word in query.lower() for word in image_keywords):
            image_url = fetch_image(query)
            if image_url:
                st.image(image_url, use_container_width=True)
                st.success("Here is the ocean-related image you requested 🌊")
            else:
                st.warning("No image found.")

        # RAG Search
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
            st.markdown(f"**🌊 Float Chat AI:** {msg['content']}")

# =====================================================
# 📊 TAB 2 — CLIMATE DASHBOARD
# =====================================================
with tab2:

    st.subheader("🌊 Global Sea Level Rise")

    fig = px.line(
        sea_df,
        x="Time",
        y="GMSL",
        title="Global Mean Sea Level Over Time"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡 Simulated Temperature vs Sea Level")

    sea_df["Temp"] = np.linspace(14, 15.5, len(sea_df))

    fig2 = px.scatter(
        sea_df,
        x="Temp",
        y="GMSL",
        trendline="ols",
        title="Temperature vs Sea Level Correlation"
    )

    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# 📈 TAB 3 — PREDICTION + MODEL EVALUATION
# =====================================================
with tab3:

    st.subheader("📈 Sea Level Prediction & Model Evaluation")

    X = sea_df["Year"].values.reshape(-1, 1)
    y = sea_df["GMSL"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    y_train_pred = model_lr.predict(X_train)
    y_test_pred = model_lr.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    st.write(f"Training R² Score: {train_r2:.4f}")
    st.write(f"Testing R² Score: {test_r2:.4f}")
    st.write(f"Training MSE: {train_mse:.4f}")
    st.write(f"Testing MSE: {test_mse:.4f}")

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(name="Train R²", x=["Train"], y=[train_r2]))
    fig_acc.add_trace(go.Bar(name="Test R²", x=["Test"], y=[test_r2]))
    fig_acc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_acc, use_container_width=True)

    future_years = np.arange(
        sea_df["Year"].max(),
        sea_df["Year"].max() + 20
    ).reshape(-1, 1)

    future_pred = model_lr.predict(future_years)

    future_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted_GMSL": future_pred
    })

    fig_pred = px.line(
        future_df,
        x="Year",
        y="Predicted_GMSL",
        title="Predicted Sea Level Rise (Next 20 Years)"
    )

    fig_pred.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.info("Linear Regression model used. R² score is used as accuracy metric.")
