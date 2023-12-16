import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from keras.preprocessing.text import one_hot, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


# Constants for model features and padding
MAX_FEATURES = 10000
MAXLEN_SPAM = 3000
MAXLEN_SENTIMENT = 200  # Adjust as per your model

# Function to set the background color
def set_bg_color(hex_color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {hex_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load the tokenizer for spam detection
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer_spam = tokenizer_from_json(data)

# Load the sentiment and spam models
sentiment_model = load_model('nn_sentiment.h5')
spam_model = load_model('spam_v2.h5')

# Function to preprocess input text for sentiment analysis
def preprocess_text_for_sentiment(text):
    lemmatizer = WordNetLemmatizer()
    # Remove all special characters/ numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    text_words = text.split()
    # Lemmatize and remove stopwords
    processed_text = [lemmatizer.lemmatize(word) for word in text_words if word not in set(stopwords.words("english"))]
    # Join words back to string
    return " ".join(processed_text)

# Function for sentiment analysis
def analyze_sentiment(text):
    processed_text = preprocess_text_for_sentiment(text)
    encoded_input = one_hot(processed_text, n=MAX_FEATURES)
    padded_input = pad_sequences([encoded_input], maxlen=MAXLEN_SENTIMENT, padding="pre")
    prediction = sentiment_model.predict(padded_input)
    sentiment_label = np.argmax(prediction, axis=1)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust as per your model
    return sentiment_map.get(sentiment_label[0])

maxlen = 10865

def preprocess_data_for_spam(email_text, tokenizer):
    # Convert text to a sequence of integers using the loaded tokenizer
    email_seq = tokenizer.texts_to_sequences([email_text])
    
    # Pad the sequences to ensure consistent input size
    email_padded = pad_sequences(email_seq, maxlen=maxlen, padding='pre')[0]

    return email_padded

# Function to detect spam
def detect_spam(data, tokenizer):
    processed_data = preprocess_data_for_spam(data, tokenizer)
    prediction = spam_model.predict(np.array([processed_data]))
    is_spam = (prediction > 0.5).astype(int)[0][0]
    return is_spam

# Streamlit application start
def main():
    # Set background color
    set_bg_color("#A7C7E7")
    
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Pages', ['Home', 'About this App', 'Models', 'Contact'])

    # Now we check the page state to determine what to display
    if page == 'Home':
        st.title("Email Classifier")
        # Spam Detection Section
        st.subheader("Spam Detection:")
        user_email = st.text_area("Enter email text for spam detection:")
        if st.button("Detect Spam"):
            spam_result = detect_spam(user_email, tokenizer_spam)
            st.write(f"Email is {'Spam' if spam_result else 'Not Spam'}")
            
        # Sentiment Analysis Section
        st.subheader("Sentiment Analysis:")
        user_input = st.text_area("Enter text here for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            sentiment_result = analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment_result}")
            
        with st.expander("Disclaimer"):
            st.write("No financial advice. The contents of this website are not legally binding. This website is a student project. ")

            
    elif page == 'About this App':
        st.title("About the app")
        st.write("""
                 **Email Classifier** is designed to streamline the sorting and categorization of emails for financial businesses. Utilizing neural network algorithms, it efficiently processes and classifies emails by analyzing patterns found in extensive datasets of email communications and financial news. This smart tool enhances email management, ensuring relevant information is readily accessible.
                 """)
    
    elif page == 'Models':
        st.title("Models' Performances")

        st.subheader("Spam Detection:")
        # Display the confusion matrix for spam detection
        st.image('spam_matrix.png', caption='Confusion Matrix for Spam Detection')

        st.subheader("Sentiment Detection:")
        # Display the confusion matrix for sentiment detection
        st.image('sent_matrix.png', caption='Confusion Matrix for Sentiment Detection')
    
    
    elif page == 'Contact':
        st.title("Contact Us")
        st.markdown("We're here to help and answer any questions you might have!")
        st.markdown('**Email:** emailclassifier@gmail.com')
        st.markdown('**Phone:** +41 (0)22 157 16 21')

if __name__ == "__main__":
    main()


##############################################################################
