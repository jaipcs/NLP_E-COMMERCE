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
    layout="wide"
)

st.title("🛒 E-Commerce Review Sentiment Analysis App")
st.write("This app performs NLP-based sentiment prediction on customer reviews.")


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

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "EDA Dashboard",
        "Model Training",
        "Predict Sentiment",
        "Model Evaluation"
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

if "clean_review" not in df.columns:
    df["clean_review"] = df["review"].apply(clean_text)

if "sentiment" not in df.columns:

    def create_sentiment(rating):

        if rating >= 4:
            return "positive"

        elif rating == 3:
            return "neutral"

        else:
            return "negative"

    df["sentiment"] = df["rating"].apply(create_sentiment)

# ============================================================
# FINAL SAFETY CLEANING BEFORE MODEL TRAINING
# ============================================================

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

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "SGD Classifier": SGDClassifier(),

        "Passive Aggressive Classifier": PassiveAggressiveClassifier(),

        "Ridge Classifier": RidgeClassifier(),

        "Linear SVC": LinearSVC(),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
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


# ============================================================
# TRAIN MODELS SAFELY
# ============================================================

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

    st.error("Model training failed. Please check your dataset columns and missing values.")

    st.exception(e)

    st.stop()



# ============================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Best Model", best_model_name)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Columns")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={
            "index": "Column",
            0: "Missing Values"
        }
    ))

    st.subheader("Sentiment Distribution")
    st.dataframe(df["sentiment"].value_counts())


# ============================================================
# PAGE 2: EDA DASHBOARD
# ============================================================

elif page == "EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.countplot(
            data=df,
            x="sentiment",
            ax=ax
        )

        ax.set_title("Sentiment Distribution")

        st.pyplot(fig)

    with col2:

        st.subheader("Rating Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.countplot(
            data=df,
            x="rating",
            ax=ax
        )

        ax.set_title("Rating Distribution")

        st.pyplot(fig)

    st.subheader("Review Length Analysis")

    if "word_count" not in df.columns:
        df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df["word_count"],
        bins=50,
        ax=ax
    )

    ax.set_title("Review Word Count Distribution")

    st.pyplot(fig)

    st.subheader("Review Length by Sentiment")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="sentiment",
        y="word_count",
        ax=ax
    )

    ax.set_title("Review Length by Sentiment")

    st.pyplot(fig)

    st.subheader("Top 20 Most Common Words")

    all_words = " ".join(df["clean_review"])

    word_freq = Counter(all_words.split())

    common_words = pd.DataFrame(
        word_freq.most_common(20),
        columns=["Word", "Frequency"]
    )

    st.dataframe(common_words)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=common_words,
        x="Frequency",
        y="Word",
        ax=ax
    )

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

elif page == "Model Training":

    st.header("🤖 Model Training Results")

    st.write("The app trains multiple NLP models using TF-IDF features.")

    st.subheader("Models Used")

    st.write(list(trained_models.keys()))

    st.subheader("Model Performance Table")

    st.dataframe(results_df)

    st.subheader("Model Comparison")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="F1 Score",
        y="Model",
        ax=ax
    )

    ax.set_title("Model Comparison by F1 Score")

    st.pyplot(fig)

    st.success(f"Best Model Selected: {best_model_name}")


# ============================================================
# PAGE 4: PREDICT SENTIMENT
# ============================================================

elif page == "Predict Sentiment":

    st.header("🔮 Predict Customer Review Sentiment")

    user_review = st.text_area(
        "Enter a customer review:",
        height=150
    )

    selected_model_name = st.selectbox(
        "Choose Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )

    selected_model = trained_models[selected_model_name]

    if st.button("Predict Sentiment"):

        if user_review.strip() == "":
            st.warning("Please enter a review first.")

        else:

            cleaned_review = clean_text(user_review)

            review_vector = tfidf.transform([cleaned_review])

            prediction = selected_model.predict(review_vector)[0]

            st.subheader("Prediction Result")

            if prediction == "positive":
                st.success(f"Predicted Sentiment: {prediction.upper()} 😊")

            elif prediction == "negative":
                st.error(f"Predicted Sentiment: {prediction.upper()} 😡")

            else:
                st.info(f"Predicted Sentiment: {prediction.upper()} 😐")

            st.write("Cleaned Review:")
            st.code(cleaned_review)


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("📈 Final Model Evaluation")

    selected_eval_model_name = st.selectbox(
        "Select Model for Evaluation",
        list(trained_models.keys())
    )

    selected_eval_model = trained_models[selected_eval_model_name]

    y_pred = selected_eval_model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

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

    st.dataframe(pd.DataFrame(report).transpose())

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
