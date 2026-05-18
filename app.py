import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #102a43 45%, #1b4965 100%);
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    h1 {
        text-align: center;
        color: #38F8D4 !important;
        font-size: 56px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 18px rgba(56, 248, 212, 0.45);
    }

    h2, h3 {
        color: #38F8D4 !important;
        font-weight: 800 !important;
    }

    p, li, label {
        color: #E2E8F0 !important;
        font-size: 17px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
        border-right: 1px solid rgba(56, 248, 212, 0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.06);
        padding: 12px;
        border-radius: 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[role="radiogroup"] label:hover {
        background: rgba(56, 248, 212, 0.15);
        border: 1px solid rgba(56, 248, 212, 0.55);
    }

    .stButton>button {
        background: linear-gradient(90deg, #38F8D4, #38BDF8);
        color: #020617 !important;
        border: none;
        border-radius: 16px;
        padding: 0.9rem;
        font-size: 19px;
        font-weight: 900;
        width: 100%;
        box-shadow: 0 0 20px rgba(56, 248, 212, 0.35);
        transition: 0.25s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(56, 248, 212, 0.65);
    }

    textarea, input {
        background-color: #FFFFFF !important;
        color: #020617 !important;
        border-radius: 14px !important;
        border: 3px solid #38F8D4 !important;
        font-size: 18px !important;
    }

    textarea::placeholder {
        color: #64748B !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 18px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    div[data-testid="metric-container"] label {
        color: #CBD5E1 !important;
    }

    div[data-testid="metric-container"] div {
        color: #FFFFFF !important;
    }

    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
    }

    code {
        color: #38F8D4 !important;
        font-size: 15px !important;
    }

    .hero-card {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 28px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
    }

    .hero-subtitle {
        text-align: center;
        color: #E2E8F0;
        font-size: 23px;
        margin-top: -10px;
        margin-bottom: 28px;
    }

    .section-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 22px;
        margin-top: 16px;
        margin-bottom: 18px;
    }

    .positive-card {
        background: linear-gradient(90deg, #00C853, #64DD17);
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(0,255,0,0.45);
    }

    .negative-card {
        background: linear-gradient(90deg, #D50000, #FF1744);
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(255,0,0,0.45);
    }

    .neutral-card {
        background: linear-gradient(90deg, #2962FF, #00B0FF);
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(0,150,255,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO TITLE
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <h1>🛒 E-Commerce Review Sentiment Analysis</h1>
        <div class="hero-subtitle">
            AI-powered NLP dashboard for customer review sentiment, EDA, model comparison, and live prediction 🚀
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NLTK DOWNLOADS
# ============================================================

@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")


download_nltk()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("final_nlp_sentiment_eda_fe_dataset.csv")
    return df


df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📌 Dataset Overview",
        "📊 EDA Dashboard",
        "🤖 Model Training",
        "🔮 Predict Sentiment",
        "📈 Model Evaluation"
    ]
)


# ============================================================
# CLEANING FUNCTION
# ============================================================

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        ps.stem(word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# PREPARE DATA
# ============================================================

if "review" not in df.columns:
    if "title" in df.columns and "body" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
        df["body"] = df["body"].fillna("").astype(str)
        df["review"] = df["title"] + " " + df["body"]

if "clean_review" not in df.columns:
    df["clean_review"] = df["review"].fillna("").astype(str).apply(clean_text)

if "sentiment" not in df.columns:

    def create_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"

    df["sentiment"] = df["rating"].apply(create_sentiment)


df["review"] = df["review"].fillna("").astype(str)
df["clean_review"] = df["clean_review"].fillna("").astype(str)
df = df[df["clean_review"].str.strip() != ""]
df = df.dropna(subset=["sentiment"])
df.reset_index(drop=True, inplace=True)


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(data):

    X = data["clean_review"].fillna("").astype(str)
    y = data["sentiment"].fillna("").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SGD Classifier": SGDClassifier(),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "Linear SVC": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score",
        ascending=False
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return (
        tfidf,
        trained_models,
        best_model,
        best_model_name,
        results_df,
        X_test_tfidf,
        y_test
    )


try:
    (
        tfidf,
        trained_models,
        best_model,
        best_model_name,
        results_df,
        X_test_tfidf,
        y_test
    ) = train_models(df)

except Exception as e:
    st.error("Model training failed. Please check dataset columns and missing values.")
    st.exception(e)
    st.stop()


# ============================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================

if page == "📌 Dataset Overview":

    st.header("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Best Model", best_model_name)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Sentiment Distribution")
    st.dataframe(df["sentiment"].value_counts(), use_container_width=True)


# ============================================================
# PAGE 2: EDA DASHBOARD
# ============================================================

elif page == "📊 EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(data=df, x="sentiment", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(data=df, x="rating", ax=ax)
        ax.set_title("Rating Distribution")
        st.pyplot(fig)

    if "word_count" not in df.columns:
        df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))

    st.subheader("Review Word Count Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["word_count"], bins=50, ax=ax)
    ax.set_title("Review Word Count Distribution")
    st.pyplot(fig)

    st.subheader("Review Length by Sentiment")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="sentiment", y="word_count", ax=ax)
    ax.set_title("Review Length by Sentiment")
    st.pyplot(fig)

    st.subheader("Top 20 Most Common Words")

    all_words = " ".join(df["clean_review"])
    word_freq = Counter(all_words.split())

    common_words = pd.DataFrame(
        word_freq.most_common(20),
        columns=["Word", "Frequency"]
    )

    st.dataframe(common_words, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=common_words, x="Frequency", y="Word", ax=ax)
    ax.set_title("Top 20 Words")
    st.pyplot(fig)

    st.subheader("Word Cloud")

    if len(all_words.strip()) > 0:
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(all_words)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No words available for word cloud.")


# ============================================================
# PAGE 3: MODEL TRAINING
# ============================================================

elif page == "🤖 Model Training":

    st.header("🤖 Model Training Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Trained", len(trained_models))
    col2.metric("Best Model", best_model_name)
    col3.metric("Best F1 Score", round(results_df.iloc[0]["F1 Score"], 4))

    st.subheader("Models Used")
    st.write(list(trained_models.keys()))

    st.subheader("Model Performance Table")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Model Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x="F1 Score", y="Model", ax=ax)
    ax.set_title("Model Comparison by F1 Score")
    st.pyplot(fig)


# ============================================================
# PAGE 4: PREDICT SENTIMENT
# ============================================================

elif page == "🔮 Predict Sentiment":

    st.header("🔮 Predict Customer Review Sentiment")

    user_review = st.text_area(
        "Enter a customer review:",
        height=180,
        placeholder="Example: The product quality is excellent and delivery was very fast..."
    )

    selected_model_name = st.selectbox(
        "Choose Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )

    selected_model = trained_models[selected_model_name]

    if st.button("🚀 Predict Sentiment"):

        if user_review.strip() == "":
            st.warning("Please enter a review first.")

        else:
            cleaned_review = clean_text(user_review)
            review_vector = tfidf.transform([cleaned_review])
            prediction = selected_model.predict(review_vector)[0]

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == "positive":
                st.markdown(
                    '<div class="positive-card">😊 POSITIVE REVIEW</div>',
                    unsafe_allow_html=True
                )

            elif prediction == "negative":
                st.markdown(
                    '<div class="negative-card">😡 NEGATIVE REVIEW</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    '<div class="neutral-card">😐 NEUTRAL REVIEW</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Selected Model")
            st.write(selected_model_name)

            st.subheader("Cleaned Review")
            st.code(cleaned_review)


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.header("📈 Final Model Evaluation")

    selected_eval_model_name = st.selectbox(
        "Select Model for Evaluation",
        list(trained_models.keys())
    )

    selected_eval_model = trained_models[selected_eval_model_name]
    y_pred = selected_eval_model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(acc, 4))
    col2.metric("Precision", round(prec, 4))
    col3.metric("Recall", round(rec, 4))
    col4.metric("F1 Score", round(f1, 4))

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_eval_model_name}")

    st.pyplot(fig)
