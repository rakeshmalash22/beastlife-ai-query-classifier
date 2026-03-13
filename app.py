import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("dataset.csv")

X = data["query"]
y = data["category"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

st.title("Beastlife AI Customer Query Classifier")

query = st.text_input("Enter customer query")

if query:
    q_vec = vectorizer.transform([query])
    prediction = model.predict(q_vec)
    st.write("Detected Category:", prediction[0])

st.subheader("Issue Distribution")

counts = data["category"].value_counts()

st.bar_chart(counts)