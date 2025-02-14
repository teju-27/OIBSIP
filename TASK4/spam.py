import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
@st.cache_data
def load_data():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data
data = load_data()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
st.title("Email Spam Detection")
st.write("This application uses machine learning to classify emails as spam or not spam (ham).")
st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter an email message to check if it's spam or ham:")
if st.sidebar.button("Check"):
    if user_input:
        new_email_transformed = vectorizer.transform([user_input])
        prediction = model.predict(new_email_transformed)
        result = "spam" if prediction[0] == 1 else "ham"
        st.success(f"The email is classified as: {result}")
    else:
        st.error("Please enter an email message to check.")
st.header("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_rep)