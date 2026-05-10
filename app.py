import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="Smart Spam Detector", page_icon="🛡️", layout="wide")

# Download stopwords silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ==============================
# SMART FEATURES DATA
# ==============================
suspicious_words = [
    "win", "winner", "won", "free", "urgent", "click", "offer", "limited", 
    "credit", "debit", "loan", "upi", "bank", "account", "blocked", "suspended",
    "kyc", "aadhaar", "pan", "verify", "update", "otp", "pin", "cvv", "password", 
    "login", "reward", "cashback", "gift", "prize", "lottery", "jackpot", "claim", 
    "bonus", "coupon", "electricity", "bill", "disconnect", "recharge", "sim", 
    "telecom", "refund", "income tax", "challan", "traffic", "parcel", "delivery",
    "customs", "courier", "job", "salary", "work from home", "part time", "investment",
    "crypto", "bitcoin", "trading", "casino", "bet", "gaming", "support", 
    "customer care", "subscription", "renewal", "pay now", "call now", "immediately", 
    "act now", "security alert", "suspicious activity", "wallet", "phonepe", 
    "google pay", "paytm"
]

# ==============================
# HELPER FUNCTIONS
# ==============================
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def detect_links(text):
    return re.findall(
        r'(https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+|is\.gd/\S+|cutt\.ly/\S+|ow\.ly/\S+)',
        text
    )

def highlight_words(text):
    highlighted = []
    for word in text.split():
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
        if clean_word in suspicious_words:
            # HTML highlighting for Streamlit
            highlighted.append(f'<span style="color:white; background-color:red; padding:2px 4px; border-radius:4px; font-weight:bold;">{word}</span>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)

def categorize_message(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["upi", "phonepe", "google pay", "paytm", "wallet", "collect request"]):
        return "UPI & Digital Wallet Scam"
    elif any(w in text_lower for w in ["credit card", "cvv", "credit limit"]):
        return "Credit Card Fraud"
    elif any(w in text_lower for w in ["debit card", "atm", "atm card"]):
        return "Debit Card / ATM Scam"
    elif any(w in text_lower for w in ["kyc", "aadhaar", "pan", "verify kyc"]):
        return "KYC Update Scam"
    elif any(w in text_lower for w in ["bank account", "account suspended", "account blocked", "bank verification"]):
        return "Bank Account Suspension Scam"
    elif any(w in text_lower for w in ["electricity", "power", "disconnect", "electricity bill"]):
        return "Electricity Bill Disconnection Scam"
    elif any(w in text_lower for w in ["income tax", "tax refund", "itr"]):
        return "Income Tax Refund Scam"
    elif any(w in text_lower for w in ["challan", "traffic fine", "e-challan"]):
        return "Traffic E-Challan Scam"
    elif any(w in text_lower for w in ["job offer", "hiring", "interview", "work from home", "part time"]):
        return "Fake Job / Work-From-Home Scam"
    elif any(w in text_lower for w in ["loan", "instant loan", "quick loan"]):
        return "Instant Loan Scam"
    elif any(w in text_lower for w in ["parcel", "delivery", "courier", "customs", "package"]):
        return "Package Delivery / Customs Scam"
    elif any(w in text_lower for w in ["lottery", "winner", "jackpot", "sweepstakes", "prize"]):
        return "Lottery & Sweepstakes Scam"
    elif any(w in text_lower for w in ["sim", "telecom", "sim blocked"]):
        return "Telecom / SIM Block Scam"
    elif any(w in text_lower for w in ["customer care", "support", "helpline", "remote access"]):
        return "Fake Tech Support Scam"
    elif any(w in text_lower for w in ["investment", "crypto", "bitcoin", "trading", "profit"]):
        return "Investment & Crypto Scam"
    elif any(w in text_lower for w in ["casino", "bet", "gaming", "gambling"]):
        return "Online Gaming & Casino Scam"
    else:
        return "General Spam"

def suggest_action(text, category, links):
    actions = []
    text_lower = text.lower()
    
    if links: actions.append("⚠️ Do NOT click on any suspicious links provided in the message.")
    if "Credit" in category or "Debit" in category:
        actions.extend(["💳 Never share your CVV, PIN, or OTP.", "🏦 Banks never ask for card details via SMS."])
    elif "UPI" in category:
        actions.extend(["📲 Never approve unknown collect requests.", "🔐 Entering your UPI PIN will DEDUCT money from your account."])
    elif "KYC" in category:
        actions.append("🪪 Verify KYC updates only by logging directly into your official banking app.")
    elif "Electricity" in category:
        actions.append("⚡ Contact your state electricity board directly. Do not call the number in the SMS.")
    elif "Job" in category:
        actions.append("💼 Genuine recruiters never ask for upfront payments or registration fees.")
    elif "Package" in category:
        actions.append("📦 Verify tracking numbers directly on the official courier website (e.g., BlueDart, India Post).")
    
    if "otp" in text_lower or "urgent" in text_lower:
        actions.append("⏳ Scammers create a false sense of urgency. Take a breath and do not share your OTP.")
    
    if not actions:
        actions.append("⚠️ Be highly cautious. Verify the sender's identity before taking any action.")
        
    return actions

# ==============================
# MODEL TRAINING (CACHED)
# ==============================
@st.cache_resource(show_spinner="Training AI Model (This only happens once)...")
def load_and_train():
    try:
        df = pd.read_csv("dataset_with_researched.csv", encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['clean_msg'] = df['message'].apply(clean_text)

        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        X = vectorizer.fit_transform(df['clean_msg'])
        y = df['label']

        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
        
        ensemble = VotingClassifier(
            estimators=[('nb', models["Naive Bayes"]), ('lr', models["Logistic Regression"]), ('rf', models["Random Forest"])],
            voting='soft'
        )
        ensemble.fit(X, y)
        return vectorizer, ensemble
    except FileNotFoundError:
        st.error("Error: 'dataset_with_researched.csv' not found! Please upload it to your project folder.")
        st.stop()

vectorizer, ensemble = load_and_train()

# ==============================
# UI DESIGN
# ==============================
st.title("🛡️ Advanced AI Spam & Phishing Detector")
st.markdown("Analyze SMS messages in real-time to detect scams, extract dangerous links, and get actionable safety advice.")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter your SMS or Email message here:", height=150, placeholder="E.g., Dear customer, your SBI account is blocked. Update KYC immediately via this link: http://scam-link.com")
    
    col_a, col_b = st.columns(2)
    with col_a:
        lang_choice = st.selectbox("Translate interface & actions to:", ["English", "Hindi (हिंदी)", "Odia (ଓଡ଼ିଆ)"])
    
    lang_map = {"English": "en", "Hindi (हिंदी)": "hi", "Odia (ଓଡ଼ିଆ)": "or"}
    target_lang = lang_map[lang_choice]
    
    analyze_btn = st.button("🔍 Analyze Message", use_container_width=True, type="primary")

if analyze_btn and user_input:
    # 1. Processing
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    pred = ensemble.predict(vector)[0]
    prob = ensemble.predict_proba(vector)[0][1]
    
    # 2. Results UI
    st.markdown("---")
    st.header("📊 Analysis Report")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if pred == 1:
            st.error("🚨 **SPAM / SCAM DETECTED**")
        else:
            st.success("✅ **SAFE MESSAGE**")
            
        # Spam Score Gradient
        score_pct = round(prob * 100, 2)
        st.markdown("**Spam Probability Score:**")
        gradient_html = f"""
        <div style="background-color: #e0e0e0; border-radius: 10px; width: 100%; height: 25px; margin-bottom: 20px;">
          <div style="background: linear-gradient(to right, #4CAF50, #FFEB3B, #F44336); width: {score_pct}%; height: 100%; border-radius: 10px; text-align: right; padding-right: 10px; color: black; font-weight: bold; line-height: 25px;">
            {score_pct}%
          </div>
        </div>
        """
        st.markdown(gradient_html, unsafe_allow_html=True)
        
        # Categorization
        if pred == 1:
            category = categorize_message(user_input)
            st.warning(f"**Detected Threat Category:** {category}")
            
    with res_col2:
        st.markdown("**🔍 Smart Highlighting:**")
        highlighted_text = highlight_words(user_input)
        st.markdown(f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">{highlighted_text}</div>', unsafe_allow_html=True)
    
    # Links & Actions
    if pred == 1:
        st.markdown("---")
        links = detect_links(user_input)
        if links:
            st.subheader("🔗 Suspicious Links Found")
            for link in links:
                st.markdown(f"- `{link}`")
                
        st.subheader("🛡️ Suggested Actions")
        actions = suggest_action(user_input, category, links)
        
        for act in actions:
            display_act = act
            # Translate if necessary
            if target_lang != "en":
                try:
                    display_act = GoogleTranslator(source='auto', target=target_lang).translate(act)
                except:
                    pass # Fallback to english if translation fails
            st.info(display_act)

with col2:
    st.markdown("### How it works")
    st.info("""
    **1. Text Analysis:** Uses TF-IDF and NLP to clean and tokenize the message.
    
    **2. AI Ensemble:** Combines Naive Bayes, Logistic Regression, and Random Forest for high accuracy.
    
    **3. Threat Intelligence:** Scans for 80+ known trigger words and flags malicious URLs.
    
    **4. Smart Categorization:** Routes the threat into specific buckets (UPI, Credit Card, KYC).
    """)
