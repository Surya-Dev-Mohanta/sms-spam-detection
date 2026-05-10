import streamlit as st
import pandas as pd
import re
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from nltk.corpus import stopwords
import nltk
import plotly.graph_objects as go
import os

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="SMS Sentinel", page_icon="🛡️", layout="wide")

# Custom CSS to match the dark, neon-cyan aesthetic from the image
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0f141e;
        color: #e0e0e0;
    }
    
    /* Input Box Glowing Border */
    .stTextArea textarea {
        background-color: #1a2332 !important;
        color: white !important;
        border: 2px solid #00f0ff !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    
    /* Cyan Analyze Button */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #00f0ff 100%);
        color: #0a0e17;
        font-weight: bold;
        border: none;
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.6);
        color: #000;
    }
    
    /* Action Buttons Custom Styling */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) button { background-color: #ff4b4b; color: white; }
    div[data-testid="stVerticalBlock"] > div:nth-child(2) button { background-color: #ffa500; color: white; }
    div[data-testid="stVerticalBlock"] > div:nth-child(3) button { background-color: #555555; color: white; }
    
    /* Cards */
    .metric-card {
        background-color: #1a2332;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a3548;
        height: 100%;
    }
    
    /* Suspicious Word Pills */
    .suspicious-pill {
        display: inline-block;
        background-color: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# MODEL LOADING & CACHING
# ==============================
@st.cache_resource(show_spinner="Training AI Models... Please wait.")
def load_and_train_models():
    if not os.path.exists("dataset_with_researched.csv"):
        # Fallback dummy data if file is missing so the UI still loads
        df = pd.DataFrame({
            'label': [0, 1, 0, 1],
            'message': ["Hello how are you", "URGENT! You won a prize click here", "Call me later", "Your UPI is blocked verify KYC"]
        })
    else:
        df = pd.read_csv("dataset_with_researched.csv", encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    df['clean_msg'] = df['message'].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['clean_msg'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }
    
    # Pre-train models
    for name, model in models.items():
        model.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[('nb', models["Naive Bayes"]), ('lr', models["Logistic Regression"]), ('rf', models["Random Forest"])],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    return vectorizer, ensemble, clean_text

vectorizer, ensemble_model, clean_text_fn = load_and_train_models()

# ==============================
# SMART FEATURES LOGIC
# ==============================
suspicious_words = [
    "win", "winner", "won", "free", "urgent", "click", "offer", "limited", "credit", "debit", "loan",
    "upi", "bank", "account", "blocked", "suspended", "kyc", "aadhaar", "pan", "verify", "update",
    "otp", "pin", "cvv", "password", "login", "reward", "cashback", "gift", "prize", "lottery",
    "jackpot", "claim", "bonus", "coupon", "electricity", "bill", "disconnect", "recharge",
    "sim", "telecom", "refund", "income tax", "challan", "traffic", "parcel", "delivery",
    "customs", "courier", "job", "salary", "work from home", "part time", "investment",
    "crypto", "bitcoin", "trading", "casino", "bet", "gaming", "support", "customer care",
    "subscription", "renewal", "pay now", "call now", "immediately", "act now",
    "security alert", "suspicious activity", "wallet", "phonepe", "google pay", "paytm"
]

def detect_links(text):
    return re.findall(r'(https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+|is\.gd/\S+|cutt\.ly/\S+|ow\.ly/\S+)', text)

def extract_suspicious_words(text):
    found = []
    for word in text.replace(".", "").replace(",", "").replace("!", "").split():
        if word.lower() in suspicious_words and word.lower() not in found:
            found.append(word.lower())
    return found

def get_highlighted_html(text):
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if clean_word in suspicious_words:
            highlighted.append(f"<span style='color: #ff4b4b; font-weight: bold;'>{word}</span>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

def categorize_message(text):
    text_lower = text.lower()
    categories = {
        "UPI & Digital Wallet Scam": ["upi", "phonepe", "google pay", "paytm", "wallet", "collect request"],
        "Credit Card Fraud": ["credit card", "cvv", "credit limit"],
        "Debit Card / ATM Scam": ["debit card", "atm", "atm card"],
        "KYC Update Scam": ["kyc", "aadhaar", "pan", "verify kyc"],
        "Bank Account Suspension Scam": ["bank account", "account suspended", "account blocked", "bank verification"],
        "Electricity Bill Disconnection": ["electricity", "power", "disconnect", "electricity bill"],
        "Fake Job Offer Scam": ["job offer", "hiring", "interview", "work from home", "part time"],
        "Lottery & Sweepstakes Scam": ["lottery", "winner", "jackpot", "sweepstakes", "prize"],
        "Investment & Crypto Scam": ["investment", "crypto", "bitcoin", "trading", "profit"]
    }
    
    for cat, keywords in categories.items():
        if any(word in text_lower for word in keywords):
            return cat
    return "General Spam"

def suggest_action(text, category, links):
    actions = []
    text_lower = text.lower()
    if links: actions.append("⚠️ Do NOT click on any suspicious links.")
    if "Credit Card" in category or "Debit Card" in category:
        actions.append("💳 Never share your CVV, PIN, or OTP.")
    if "UPI" in category:
        actions.append("📲 Never approve unknown collect requests.")
    if "KYC" in category:
        actions.append("🪪 Verify KYC updates only on official websites.")
    if "Electricity" in category:
        actions.append("⚡ Contact your electricity board directly. Do not pay via SMS links.")
    if "otp" in text_lower or "urgent" in text_lower:
        actions.append("🔐 Scammers create urgency. Never share OTPs.")
    if not actions:
        actions.append("⚠️ Be cautious. Verify the sender before taking any action.")
    return actions

# ==============================
# UI RENDERING
# ==============================

st.title("🛡️ SMS SENTINEL")
st.markdown("<p style='color: #888; margin-top: -15px;'>AI-Powered Spam Detection</p>", unsafe_allow_html=True)

st.markdown("### SMS INPUT")
sms_input = st.text_area("Enter SMS content here...", height=100, label_visibility="collapsed")

if st.button("ANALYZE MESSAGE"):
    if sms_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Prediction
        cleaned = clean_text_fn(sms_input)
        vector = vectorizer.transform([cleaned])
        pred = ensemble_model.predict(vector)[0]
        prob = ensemble_model.predict_proba(vector)[0][1]
        
        is_spam = pred == 1
        confidence = prob * 100

        # Analysis Banner
        if is_spam:
            st.markdown(f"""
            <div style="background-color: rgba(255, 0, 0, 0.1); border: 1px solid #ff4b4b; border-radius: 8px; padding: 15px; margin-top: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center;">
                <h2 style="color: #ff4b4b; margin: 0;">⚠️ ANALYSIS RESULT: SPAM DETECTED</h2>
                <h2 style="color: #ff4b4b; margin: 0;">{confidence:.1f}%<span style="font-size: 0.5em; color: #888; display: block; text-align: right;">SPAM PROBABILITY</span></h2>
            </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
            <div style="background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; border-radius: 8px; padding: 15px; margin-top: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center;">
                <h2 style="color: #00ff00; margin: 0;">✅ ANALYSIS RESULT: SAFE MESSAGE</h2>
                <h2 style="color: #00ff00; margin: 0;">{confidence:.1f}%<span style="font-size: 0.5em; color: #888; display: block; text-align: right;">SPAM PROBABILITY</span></h2>
            </div>
            """, unsafe_allow_html=True)

        # Smart Highlighted Text
        st.markdown("**Highlighted Content:**")
        st.markdown(f"<div style='background: #1a2332; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>{get_highlighted_html(sms_input)}</div>", unsafe_allow_html=True)

        # 4-Column Layout Dashboard
        col1, col2, col3, col4 = st.columns(4)

        # Column 1: Suspicious Elements
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### 🚨 SUSPICIOUS ELEMENTS")
            found_words = extract_suspicious_words(sms_input)
            if found_words:
                pills_html = "".join([f"<span class='suspicious-pill'>{word.capitalize()}</span>" for word in found_words])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.write("No typical suspicious keywords found.")
            
            links = detect_links(sms_input)
            if links:
                st.markdown("<br><b>Detected Links:</b>", unsafe_allow_html=True)
                for link in links:
                    st.markdown(f"<span style='color: #ff4b4b; font-size: 0.9em;'>🔗 {link}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Column 2: Translate SMS
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### 🔤 TRANSLATE SMS")
            lang = st.selectbox("Select Language", ["Hindi (HI)", "Odia (OR)"], label_visibility="collapsed")
            
            target_lang = "hi" if "Hindi" in lang else "or"
            try:
                translated_text = GoogleTranslator(source='auto', target=target_lang).translate(sms_input)
                st.markdown(f"<p style='font-size: 0.9em; margin-top: 10px; color: #ccc;'>{translated_text}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error("Translation unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Column 3: Spam Confidence Gauge Chart
        with col3:
            st.markdown("<div class='metric-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("##### 🎯 SPAM CONFIDENCE")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                number = {'suffix': "%", 'font': {'color': '#e0e0e0', 'size': 30}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "#2a3548",
                    'steps': [
                        {'range': [0, 33], 'color': "#00ff00"},
                        {'range': [33, 66], 'color': "#ffa500"},
                        {'range': [66, 100], 'color': "#ff4b4b"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence
                    }
                }
            ))
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            category = categorize_message(sms_input)
            risk_level = "High Risk" if is_spam else "Low Risk"
            risk_color = "#ff4b4b" if is_spam else "#00ff00"
            st.markdown(f"<p style='color: {risk_color}; font-weight: bold; margin-top: -30px;'>{risk_level}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 0.8em; color: #888;'>Cat: {category}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Column 4: Recommended Actions
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("##### ⚡ RECOMMENDED ACTIONS")
            
            # Action Buttons
            st.button("REPORT SMS", use_container_width=True)
            st.button("BLOCK SENDER", use_container_width=True)
            st.button("IGNORE & DELETE", use_container_width=True)
            
            # Dynamic Code Actions
            st.markdown("<hr style='border-color: #2a3548; margin: 10px 0;'>", unsafe_allow_html=True)
            actions = suggest_action(sms_input, category, links)
            for act in actions:
                # Optional: translate the actions as requested
                try:
                    translated_act = GoogleTranslator(source='auto', target=target_lang).translate(act)
                    display_act = translated_act
                except:
                    display_act = act
                st.markdown(f"<p style='font-size: 0.8em; margin-bottom: 4px;'>• {display_act}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
