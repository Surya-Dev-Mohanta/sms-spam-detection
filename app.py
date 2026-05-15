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
# PAGE CONFIGURATION & STATE
# ==============================
st.set_page_config(page_title="SMS SPAM DETECTION", page_icon="🛡️", layout="wide")

# Initialize session state so results don't disappear on language change
if "analyze_clicked" not in st.session_state:
    st.session_state.analyze_clicked = False

def trigger_analysis():
    st.session_state.analyze_clicked = True

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f141e; color: #e0e0e0; }
    .stTextArea textarea { background-color: #1a2332 !important; color: white !important; border: 2px solid #00f0ff !important; border-radius: 8px !important; box-shadow: 0 0 10px rgba(0, 240, 255, 0.2); }
    .stButton > button { background: linear-gradient(90deg, #00d4ff 0%, #00f0ff 100%); color: #0a0e17; font-weight: bold; border: none; width: 100%; border-radius: 8px; padding: 10px; transition: all 0.3s ease; }
    .stButton > button:hover { box-shadow: 0 0 15px rgba(0, 240, 255, 0.6); color: #000; }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) button { background-color: #ff4b4b; color: white; }
    div[data-testid="stVerticalBlock"] > div:nth-child(2) button { background-color: #ffa500; color: white; }
    div[data-testid="stVerticalBlock"] > div:nth-child(3) button { background-color: #555555; color: white; }
    
    /* FIX: Force Streamlit columns to perfectly wrap content in dark boxes */
    [data-testid="column"], [data-testid="stColumn"] { 
        background-color: #1a2332 !important; 
        padding: 15px !important; 
        border-radius: 10px !important; 
        border: 1px solid #2a3548 !important; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    
    .suspicious-pill { display: inline-block; background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; border: 1px solid #ff4b4b; padding: 4px 10px; border-radius: 15px; margin: 4px; font-size: 0.85em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================
# MODEL LOADING & CACHING
# ==============================
@st.cache_resource(show_spinner="Training AI Models... Please wait.")
def load_and_train_models():
    if not os.path.exists("dataset_with_researched.csv"):
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
    if any(word in text_lower for word in ["upi", "phonepe", "google pay", "paytm", "wallet", "collect request"]):
        return "UPI & Digital Wallet Scam"
    elif any(word in text_lower for word in ["credit card", "cvv", "credit limit"]):
        return "Credit Card Fraud"
    elif any(word in text_lower for word in ["debit card", "atm", "atm card"]):
        return "Debit Card / ATM Scam"
    elif any(word in text_lower for word in ["kyc", "aadhaar", "pan", "verify kyc"]):
        return "KYC Update Scam"
    elif any(word in text_lower for word in ["bank account", "account suspended", "account blocked", "bank verification"]):
        return "Bank Account Suspension Scam"
    elif any(word in text_lower for word in ["electricity", "power", "disconnect", "electricity bill"]):
        return "Electricity Bill Disconnection Scam"
    elif any(word in text_lower for word in ["income tax", "tax refund", "itr"]):
        return "Income Tax Refund Scam"
    elif any(word in text_lower for word in ["challan", "traffic fine", "e-challan"]):
        return "Traffic E-Challan Scam"
    elif any(word in text_lower for word in ["job offer", "hiring", "interview"]):
        return "Fake Job Offer Scam"
    elif any(word in text_lower for word in ["work from home", "part time", "earn daily", "earn money"]):
        return "Part-Time / Work-From-Home Scam"
    elif any(word in text_lower for word in ["loan", "instant loan", "quick loan"]):
        return "Instant Loan Scam"
    elif any(word in text_lower for word in ["parcel", "delivery", "courier", "customs", "package"]):
        return "Package Delivery / Customs Scam"
    elif any(word in text_lower for word in ["lottery", "winner", "jackpot", "sweepstakes", "prize"]):
        return "Lottery & Sweepstakes Scam"
    elif any(word in text_lower for word in ["sim", "telecom", "sim blocked"]):
        return "Telecom / SIM Block Scam"
    elif any(word in text_lower for word in ["customer care", "support", "helpline", "remote access"]):
        return "Fake Customer Care / Tech Support Scam"
    elif any(word in text_lower for word in ["subscription", "renewal", "netflix", "amazon prime"]):
        return "Subscription & Service Renewal Scam"
    elif any(word in text_lower for word in ["investment", "crypto", "bitcoin", "trading", "profit"]):
        return "Investment & Cryptocurrency Scam"
    elif any(word in text_lower for word in ["casino", "bet", "gaming", "gambling"]):
        return "Online Gaming & Casino Scam"
    elif any(word in text_lower for word in ["urgent help", "family emergency", "send money urgently"]):
        return "Emergency / Imposter Scam"
    elif any(word in text_lower for word in ["reward points", "cashback", "redeem reward"]):
        return "Fake Reward Points / Cashback Scam"
    else:
        return "General Spam"

def suggest_action(text, category, links):
    actions = []
    text_lower = text.lower()

    if links:
        actions.append("⚠️ Do NOT click on suspicious links.")

    if "Credit Card" in category:
        actions.append("💳 Never share your CVV, PIN, or OTP.")
        actions.append("🏦 Banks never ask for card details via SMS.")
    elif "Debit Card" in category:
        actions.append("🏧 Never share ATM PIN or OTP.")
        actions.append("🚫 Block your card if suspicious activity is found.")
    elif "UPI" in category:
        actions.append("📲 Never approve unknown collect requests.")
        actions.append("🔐 UPI scams can instantly steal money.")
    elif "KYC" in category:
        actions.append("🪪 Verify KYC updates only on official websites.")
        actions.append("🚫 Never upload Aadhaar/PAN on unknown links.")
    elif "Bank" in category:
        actions.append("🏦 Contact your bank directly using official numbers.")
        actions.append("🔐 Never share banking credentials.")
    elif "Electricity" in category:
        actions.append("⚡ Verify electricity bill alerts from official apps.")
        actions.append("🚫 Fake bill scams often create panic.")
    elif "Income Tax" in category:
        actions.append("💰 Check tax refunds only on official government portals.")
    elif "Traffic" in category:
        actions.append("🚦 Verify e-challans on official transport websites.")
    elif "Job" in category or "Work-From-Home" in category:
        actions.append("💼 Genuine jobs never ask for upfront payment.")
        actions.append("🚫 Avoid sharing personal documents carelessly.")
    elif "Loan" in category:
        actions.append("💰 Avoid instant loan scams with advance fees.")
        actions.append("📄 Verify RBI-registered lenders only.")
    elif "Package" in category:
        actions.append("📦 Verify courier updates from official delivery apps.")
    elif "Lottery" in category:
        actions.append("🎁 Genuine lotteries do not ask for processing fees.")
    elif "Telecom" in category:
        actions.append("📶 Contact telecom providers only through official apps.")
    elif "Customer Care" in category:
        actions.append("🛠️ Never install remote access apps for support.")
        actions.append("🚫 Fake support agents can steal banking details.")
    elif "Subscription" in category:
        actions.append("📺 Verify renewals from official service providers.")
    elif "Investment" in category:
        actions.append("📈 High guaranteed returns are usually scams.")
        actions.append("🚫 Be cautious with crypto investment messages.")
    elif "Gaming" in category:
        actions.append("🎮 Avoid gambling and betting links from SMS.")
    elif "Emergency" in category:
        actions.append("📞 Confirm emergencies directly with family members.")
    elif "Cashback" in category:
        actions.append("🎁 Fake cashback links can steal login credentials.")

    if "otp" in text_lower or "urgent" in text_lower:
        actions.append("🔐 Never share OTP with anyone.")
        actions.append("⏳ Scammers create urgency to trick victims.")

    if "call" in text_lower or "contact" in text_lower:
        actions.append("📞 Avoid calling unknown numbers from suspicious SMS.")

    if not actions:
        actions.append("⚠️ Be cautious. Verify sender before taking action.")

    return actions

# ==============================
# UI RENDERING
# ==============================

st.title("🛡️ SMS SPAM DETECTION")
st.markdown("<p style='color: #888; margin-top: -15px;'>AI-Powered Spam Detection</p>", unsafe_allow_html=True)

st.markdown("### SMS INPUT")
sms_input = st.text_area("Enter SMS content here...", height=100, label_visibility="collapsed")

# Connect the button to the session state function
st.button("ANALYZE MESSAGE", on_click=trigger_analysis)

# Only show results if the button has been clicked AND there is text
if st.session_state.analyze_clicked:
    if sms_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        cleaned = clean_text_fn(sms_input)
        vector = vectorizer.transform([cleaned])
        pred = ensemble_model.predict(vector)[0]
        prob = ensemble_model.predict_proba(vector)[0][1]
        
        is_spam = pred == 1
        confidence = prob * 100

        # Responsive & Super-Sized Banner
        if is_spam:
            st.markdown(f"""
            <div style="background-color: rgba(255, 0, 0, 0.1); border: 2px solid #ff4b4b; border-radius: 12px; padding: 25px 30px; margin-top: 15px; margin-bottom: 25px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 15px; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);">
                <div style="color: #ff4b4b; font-size: clamp(1.8rem, 3.5vw, 3rem); font-weight: 900; margin: 0; line-height: 1.1;">⚠️ ANALYSIS RESULT: SPAM DETECTED</div>
                <div style="color: #ff4b4b; font-size: clamp(3rem, 6vw, 5rem); font-weight: 900; margin: 0; text-align: right; line-height: 1;">{confidence:.1f}%<span style="font-size: clamp(0.9rem, 1.5vw, 1.2rem); color: #888; display: block; text-align: right; font-weight: 600; letter-spacing: 1.5px; margin-top: 8px;">SPAM PROBABILITY</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
            <div style="background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00ff00; border-radius: 12px; padding: 25px 30px; margin-top: 15px; margin-bottom: 25px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 15px; box-shadow: 0 4px 15px rgba(0, 255, 0, 0.2);">
                <div style="color: #00ff00; font-size: clamp(1.8rem, 3.5vw, 3rem); font-weight: 900; margin: 0; line-height: 1.1;">✅ ANALYSIS RESULT: SAFE MESSAGE</div>
                <div style="color: #00ff00; font-size: clamp(3rem, 6vw, 5rem); font-weight: 900; margin: 0; text-align: right; line-height: 1;">{confidence:.1f}%<span style="font-size: clamp(0.9rem, 1.5vw, 1.2rem); color: #888; display: block; text-align: right; font-weight: 600; letter-spacing: 1.5px; margin-top: 8px;">SPAM PROBABILITY</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Highlighted Content:**")
        st.markdown(f"<div style='background: #1a2332; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>{get_highlighted_html(sms_input)}</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        # Col 1: Elements
        with col1:
            st.markdown("##### 🚨 SUSPICIOUS ELEMENTS")
            found_words = extract_suspicious_words(sms_input)
            if found_words:
                pills_html = "".join([f"<span class='suspicious-pill'>{word.capitalize()}</span>" for word in found_words])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.write("No suspicious keywords found.")
            
            links = detect_links(sms_input)
            if links:
                st.markdown("<br><b>Detected Links:</b>", unsafe_allow_html=True)
                for link in links:
                    st.markdown(f"<span style='color: #ff4b4b; font-size: 0.9em;'>🔗 {link}</span>", unsafe_allow_html=True)

        # Col 2: Translate
        with col2:
            st.markdown("##### 🔤 TRANSLATE SMS")
            lang = st.selectbox("Select Language", ["English (EN)", "Hindi (HI)", "Odia (OR)"], label_visibility="collapsed")
            
            if "English" not in lang:
                target_lang = "hi" if "Hindi" in lang else "or"
                try:
                    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(sms_input)
                    st.markdown(f"<p style='font-size: 0.9em; margin-top: 10px; color: #ccc;'>{translated_text}</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Translation unavailable.")
            else:
                st.markdown("<p style='font-size: 0.9em; margin-top: 10px; color: #888;'>Translation off.</p>", unsafe_allow_html=True)

        # Col 3: Confidence
        with col3:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("##### 🎯 SPAM CONFIDENCE")
            
            if confidence < 33.33:
                risk_level = "Low Risk"
                risk_color = "#00ff00"  # Green
            elif confidence < 66.66:
                risk_level = "Medium Risk"
                risk_color = "#ffa500"  # Yellow
            else:
                risk_level = "High Risk"
                risk_color = "#ff4b4b"  # Red

            category = categorize_message(sms_input)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                number = {'suffix': "%", 'font': {'color': '#e0e0e0', 'size': 30}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': risk_color, 'thickness': 0.6},
                    'bgcolor': "#2a3548",
                    'steps': [
                        {'range': [0, 33], 'color': "rgba(0, 255, 0, 0.15)"},
                        {'range': [33, 66], 'color': "rgba(255, 165, 0, 0.15)"},
                        {'range': [66, 100], 'color': "rgba(255, 75, 75, 0.15)"}
                    ]
                }
            ))
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown(f"<p style='color: {risk_color}; font-weight: bold; margin-top: -30px;'>{risk_level}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 0.8em; color: #888;'>Cat: {category}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Col 4: Recommended Actions
        with col4:
            st.markdown("##### ⚡ RECOMMENDED ACTIONS")
            
            if is_spam:
                st.button("REPORT SMS", use_container_width=True)
                st.button("BLOCK SENDER", use_container_width=True)
                st.button("IGNORE & DELETE", use_container_width=True)
                
                st.markdown("<hr style='border-color: #2a3548; margin: 10px 0;'>", unsafe_allow_html=True)
                actions = suggest_action(sms_input, category, links)
                
                for act in actions:
                    if "English" not in lang:
                        try:
                            display_act = GoogleTranslator(source='auto', target=target_lang).translate(act)
                        except:
                            display_act = act
                    else:
                        display_act = act
                    
                    st.markdown(f"<p style='font-size: 0.8em; margin-bottom: 4px;'>{display_act}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #00ff00; margin-top: 20px; font-weight: bold; text-align: center;'>✅ Safe message.</p>", unsafe_allow_html=True)
                st.markdown("<p style='color: #888; text-align: center;'>No action required.</p>", unsafe_allow_html=True)
