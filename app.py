import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("💬 Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload your sentiment CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

        # Display dataset preview
        st.subheader("📊 Sample Data")
        st.write(df.head())

        # Check for required columns
        if 'Text' not in df.columns or 'Sentiment' not in df.columns:
            st.error("CSV must contain 'Text' and 'Sentiment' columns.")
            st.stop()

        # Prepare the data
        X = df['Text']
        y = df['Sentiment']

        # Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_vectorized = vectorizer.fit_transform(X)

        # Model training
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Input box
        st.subheader("✍️ Enter a Text")
        user_input = st.text_input("Type your comment below:")

        # Prediction
        if user_input:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            st.success(f"🔍 Predicted Sentiment: **{prediction}**")

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
