import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Gmail API imports
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import pickle as pk

# Gmail read-only scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('spam_detector_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'spam_detector_model.pkl' is in the same directory.")
        return None

# Predict single email text
def detect_spam(email_text, model):
    if model is None:
        return "Model not loaded", 0.0

    prediction = model.predict([email_text])[0]
    spam_probability = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([email_text])[0]
        spam_probability = proba[1]

    if prediction == 0:
        return "Ham (Not Spam)", spam_probability
    else:
        return "Spam", spam_probability

# Gmail Authentication
def get_credentials_dict():
    return {
        "installed": {
            "client_id": st.secrets["google"]["client_id"],
            "client_secret": st.secrets["google"]["client_secret"],
            "project_id": st.secrets["google"]["project_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [st.secrets["google"]["redirect_uri"]]
        }
    }

def authenticate_gmail():
    creds = None
    if os.path.exists('token.pkl') and os.access('token.pkl', os.R_OK):
        with open('token.pkl', 'rb') as token:
            creds = pk.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(get_credentials_dict(), SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.pkl', 'wb') as token:
            pk.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service


# Fetch recent 100 emails
def fetch_emails(service, max_results=100):
    result = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = result.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        snippet = msg_data.get('snippet', '')
        headers = msg_data.get("payload", {}).get("headers", [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        emails.append({'subject': subject, 'from': sender, 'text': snippet})
    return emails

# Bulk classify fetched emails
def classify_bulk_emails(emails, model):
    results = []
    for mail in emails:
        text = mail['text']
        pred = model.predict([text])[0]
        label = 'Spam' if pred == 1 else 'Not Spam'
        mail['label'] = label
        results.append(mail)
    return results

# Streamlit App
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

def main():
    st.title("📧 Email Spam Detection System")
    st.markdown("---")

    model = load_model()

    tab1, tab2 = st.tabs(["📥 Paste Email Manually", "📬 Gmail Auto Classification"])

    with tab1:
        st.write("Enter an email text below to check if it's spam or ham (legitimate email)")
        email_text = st.text_area(
            "Paste your email content here:",
            height=300,
            placeholder="Enter the email text you want to analyze..."
        )

        if st.button("🔍 Analyze Email", type="primary"):
            if email_text.strip():
                with st.spinner("Analyzing email..."):
                    result, probability = detect_spam(email_text, model)

                    st.subheader("Analysis Results")

                    if result == "Spam":
                        st.error(f"🚨 **{result}**")
                        st.error(f"Spam Probability: {probability:.2%}")
                    else:
                        st.success(f"✅ **{result}**")
                        st.success(f"Spam Probability: {probability:.2%}")

                    confidence = max(probability, 1 - probability)
                    st.info(f"Confidence Level: {confidence:.2%}")
            else:
                st.warning("Please enter some email text to analyze.")

    with tab2:
        st.subheader("📬 Gmail Integration")
        st.write("Authenticate and fetch the top 100 recent emails from your inbox and classify them.")
        if st.button("🔐 Login to Gmail & Classify"):
            try:
                with st.spinner("Authenticating and fetching emails..."):
                    service = authenticate_gmail()
                    emails = fetch_emails(service)
                    results = classify_bulk_emails(emails, model)
                    df = pd.DataFrame(results)
                st.success("Classification complete!")
                st.dataframe(df[['subject', 'from', 'label']])
            except Exception as e:
                st.error(f"❌ Error occurred: {e}")
                st.info("Ensure you have the correct Gmail API credentials in 'credentials.json'.")

    st.markdown("---")
    with st.expander("ℹ️ About the App"):
        st.info("""
        This spam detector uses:
        - **CountVectorizer** for text processing
        - **Multinomial Naive Bayes** for classification
        - Integrated with **Gmail API** to analyze real emails
        """)
        st.write("**Note:** No emails are stored or shared. This app only analyzes message snippets locally.")

if __name__ == "__main__":
    main()
