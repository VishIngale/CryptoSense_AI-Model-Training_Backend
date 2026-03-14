from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_model import predict_sentiment
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests

from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import shap
import xgboost as xgb

# ✅ Email imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()

# CORS CONFIGURATION
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]


@app.get("/")
def home():
    return {"message": "Crypto Sentiment API Running 🚀"}

@app.post("/predict")
def predict(data: TextInput):
    return predict_sentiment(data.text)

@app.post("/predict-batch")
def predict_batch(data: BatchInput):
    results = []
    for text in data.texts:
        result = predict_sentiment(text)
        results.append(result)
    return results


# ======================================
# LOAD DATASET
# ======================================

combined_df = pd.read_csv("data/combined_dataset.csv")
combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

dl_model = load_model("price_model.h5", compile=False)

# ======================================
# PREPARE DATA FOR ALL 3 COINS
# ======================================

sequence_length = 10

COIN_NAME_MAP = {
    "bitcoin": "Bitcoin",
    "ethereum": "Ethereum",
    "solana": "Solana",
}

coin_data = {}

for frontend_name, dataset_name in COIN_NAME_MAP.items():
    df_coin = combined_df[combined_df["cryptocurrency"] == dataset_name].copy()
    df_coin = df_coin.sort_values("timestamp")
    raw_prices = df_coin["current_price_usd"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw_prices)
    coin_data[frontend_name] = {
        "raw_prices": raw_prices,
        "scaled_prices": scaled,
        "scaler": scaler,
    }


# ======================================
# XGBOOST + SHAP SETUP
# ======================================

FEATURES = [
    "social_sentiment_score",
    "news_sentiment_score",
    "fear_greed_index",
    "rsi_technical_indicator",
    "volatility_index",
    "price_change_24h_percent",
]

FEATURE_LABELS = {
    "social_sentiment_score": "Social Sentiment",
    "news_sentiment_score": "News Sentiment",
    "fear_greed_index": "Fear & Greed Index",
    "rsi_technical_indicator": "RSI Indicator",
    "volatility_index": "Volatility",
    "price_change_24h_percent": "24h Price Change",
}

xgb_models = {}
shap_explainers = {}

for frontend_name, dataset_name in COIN_NAME_MAP.items():
    df_coin = combined_df[combined_df["cryptocurrency"] == dataset_name].copy()
    df_coin = df_coin.sort_values("timestamp")
    X = df_coin[FEATURES]
    y = df_coin["current_price_usd"]
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    xgb_models[frontend_name] = model
    shap_explainers[frontend_name] = {
        "explainer": explainer,
        "X": X,
        "df": df_coin,
    }

print("✅ XGBoost + SHAP models ready for Bitcoin, Ethereum, Solana")


# ======================================
# HYBRID ANALYSIS — per coin
# ======================================

def hybrid_market_analysis(coin_key="bitcoin"):
    dataset_name = COIN_NAME_MAP.get(coin_key, "Bitcoin")
    coin_df = combined_df[combined_df["cryptocurrency"] == dataset_name]
    recent_data = coin_df.sort_values("timestamp").tail(50)

    avg_social = recent_data["social_sentiment_score"].mean()
    avg_news = recent_data["news_sentiment_score"].mean()
    avg_fear_greed = recent_data["fear_greed_index"].mean()
    avg_rsi = recent_data["rsi_technical_indicator"].mean()
    avg_volatility = recent_data["volatility_index"].mean()
    avg_price_change = recent_data["price_change_24h_percent"].mean()

    hybrid_score = (
        (avg_social * 0.25)
        + (avg_news * 0.25)
        + (avg_price_change * 0.20)
        + (avg_fear_greed * 0.15)
        + (avg_rsi * 0.10)
        - (avg_volatility * 0.05)
    )

    if hybrid_score > 40:
        signal = "STRONG BUY"
    elif hybrid_score > 15:
        signal = "BUY"
    elif hybrid_score < -15:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "coin": coin_key,
        "hybrid_score": round(hybrid_score, 2),
        "signal": signal,
        "avg_social_sentiment": round(avg_social, 2),
        "avg_news_sentiment": round(avg_news, 2),
        "avg_fear_greed": round(avg_fear_greed, 2),
        "avg_rsi": round(avg_rsi, 2),
        "avg_volatility": round(avg_volatility, 2),
    }

@app.get("/hybrid-analysis")
def get_hybrid_analysis(coin: str = "bitcoin"):
    return hybrid_market_analysis(coin.lower())


# ======================================
# PRICE PREDICTION API
# ======================================

class PricePredictionInput(BaseModel):
    days: int
    coin: str = "bitcoin"


@app.post("/predict-price")
def predict_price(data: PricePredictionInput):
    days_to_predict = data.days
    coin_key = data.coin.lower()

    if coin_key not in coin_data:
        coin_key = "bitcoin"

    scaled_prices = coin_data[coin_key]["scaled_prices"]
    scaler = coin_data[coin_key]["scaler"]
    raw_prices = coin_data[coin_key]["raw_prices"]

    last_sequence = scaled_prices[-sequence_length:]
    current_sequence = last_sequence.reshape(1, sequence_length, 1)
    predictions = []

    for _ in range(days_to_predict):
        next_price_scaled = dl_model.predict(current_sequence, verbose=0)
        predictions.append(next_price_scaled[0][0])
        next_value = next_price_scaled.reshape(1, 1, 1)
        current_sequence = np.concatenate(
            (current_sequence[:, 1:, :], next_value), axis=1
        )

    predicted_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )
    final_price = predicted_prices[-1][0]
    last_real_price = raw_prices[-1][0]

    trend = "UP 📈" if final_price > last_real_price else "DOWN 📉"
    confidence = round(
        100 - abs(final_price - last_real_price) / last_real_price * 100, 2
    )

    return {
        "predicted_price": round(float(final_price), 2),
        "trend": trend,
        "confidence": confidence,
    }


# ======================================
# SHAP EXPLANATION ENDPOINT
# ======================================

@app.get("/shap-explanation")
def shap_explanation(coin: str = "bitcoin"):
    coin_key = coin.lower()
    if coin_key not in shap_explainers:
        coin_key = "bitcoin"

    explainer_data = shap_explainers[coin_key]
    explainer = explainer_data["explainer"]
    X = explainer_data["X"]
    df_coin = explainer_data["df"]

    X_recent = X.tail(30)
    shap_values = explainer.shap_values(X_recent)
    mean_shap = np.abs(shap_values).mean(axis=0)
    last_shap = shap_values[-1]
    recent = df_coin.tail(30)

    result = []
    for i, feature in enumerate(FEATURES):
        result.append({
            "feature": FEATURE_LABELS[feature],
            "shap_value": round(float(last_shap[i]), 4),
            "importance": round(float(mean_shap[i]), 4),
            "current_value": round(float(recent[feature].iloc[-1]), 4),
            "avg_value": round(float(recent[feature].mean()), 4),
        })

    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    top_positive = [r for r in result if r["shap_value"] > 0]
    top_negative = [r for r in result if r["shap_value"] < 0]
    explanation_parts = []

    if top_positive:
        top = top_positive[0]
        explanation_parts.append(
            f"{top['feature']} is the strongest positive factor pushing {coin_key.capitalize()} price UP."
        )
    if top_negative:
        top = top_negative[0]
        explanation_parts.append(
            f"{top['feature']} is currently the biggest negative pressure pulling price DOWN."
        )

    total_shap = sum(r["shap_value"] for r in result)
    if total_shap > 0:
        explanation_parts.append(
            f"Overall market signals for {coin_key.capitalize()} are leaning BULLISH based on recent data."
        )
    else:
        explanation_parts.append(
            f"Overall market signals for {coin_key.capitalize()} are leaning BEARISH based on recent data."
        )

    return {
        "coin": coin_key,
        "shap_features": result,
        "explanation": " ".join(explanation_parts),
    }


# ======================================
# ASK SHAP QUESTION ENDPOINT
# ======================================

class ShapQuestionInput(BaseModel):
    coin: str = "bitcoin"
    question: str


@app.post("/ask-shap")
def ask_shap(data: ShapQuestionInput):
    coin_key = data.coin.lower()
    question = data.question.lower()

    if coin_key not in shap_explainers:
        coin_key = "bitcoin"

    explainer_data = shap_explainers[coin_key]
    explainer = explainer_data["explainer"]
    X = explainer_data["X"]
    df_coin = explainer_data["df"]

    X_recent = X.tail(30)
    shap_values = explainer.shap_values(X_recent)
    last_shap = shap_values[-1]
    recent = df_coin.tail(30)

    feature_context = {}
    for i, feature in enumerate(FEATURES):
        feature_context[feature] = {
            "label": FEATURE_LABELS[feature],
            "shap": round(float(last_shap[i]), 4),
            "value": round(float(recent[feature].iloc[-1]), 4),
        }

    coin_name = coin_key.capitalize()
    answer = ""

    if any(w in question for w in ["why", "reason", "cause", "explain"]):
        top = max(feature_context.values(), key=lambda x: abs(x["shap"]))
        direction = "pushing price UP" if top["shap"] > 0 else "pulling price DOWN"
        answer = (
            f"The main reason for {coin_name}'s current price movement is "
            f"{top['label']} (value: {top['value']}), which is {direction} "
            f"with a SHAP influence of {top['shap']}."
        )
    elif any(w in question for w in ["sentiment", "social", "news"]):
        s = feature_context["social_sentiment_score"]
        n = feature_context["news_sentiment_score"]
        answer = (
            f"For {coin_name}, Social Sentiment score is {s['value']} "
            f"(influence: {s['shap']}) and News Sentiment score is {n['value']} "
            f"(influence: {n['shap']}). "
            f"{'Both are positively influencing price.' if s['shap'] > 0 and n['shap'] > 0 else 'Sentiment signals are mixed.'}"
        )
    elif any(w in question for w in ["fear", "greed"]):
        fg = feature_context["fear_greed_index"]
        level = "high (Greed)" if fg["value"] > 60 else "low (Fear)" if fg["value"] < 40 else "neutral"
        answer = (
            f"The Fear & Greed Index for {coin_name} is currently {fg['value']} which is {level}. "
            f"It is {'positively' if fg['shap'] > 0 else 'negatively'} influencing price "
            f"with a SHAP value of {fg['shap']}."
        )
    elif any(w in question for w in ["rsi", "technical"]):
        rsi = feature_context["rsi_technical_indicator"]
        condition = "overbought (above 70)" if rsi["value"] > 70 else "oversold (below 30)" if rsi["value"] < 30 else "neutral range"
        answer = (
            f"The RSI for {coin_name} is {rsi['value']}, which is in {condition}. "
            f"RSI is {'supporting' if rsi['shap'] > 0 else 'pressuring'} the price "
            f"with a SHAP influence of {rsi['shap']}."
        )
    elif any(w in question for w in ["volatile", "volatility", "risk"]):
        vol = feature_context["volatility_index"]
        level = "high" if vol["value"] > 60 else "low" if vol["value"] < 30 else "moderate"
        answer = (
            f"{coin_name} currently has {level} volatility (index: {vol['value']}). "
            f"Volatility is {'adding upward' if vol['shap'] > 0 else 'adding downward'} pressure "
            f"with a SHAP value of {vol['shap']}."
        )
    elif any(w in question for w in ["bullish", "bearish", "trend", "direction", "going"]):
        total_shap = sum(f["shap"] for f in feature_context.values())
        top_factor = max(feature_context.values(), key=lambda x: x["shap"])
        answer = (
            f"{coin_name} is currently showing a {'BULLISH 📈' if total_shap > 0 else 'BEARISH 📉'} signal. "
            f"The strongest driving factor is {top_factor['label']} "
            f"with a SHAP influence of {top_factor['shap']}."
        )
    elif any(w in question for w in ["best", "strongest", "most important", "top factor"]):
        top = max(feature_context.values(), key=lambda x: abs(x["shap"]))
        answer = (
            f"The most influential factor for {coin_name} right now is {top['label']} "
            f"with a SHAP value of {top['shap']} and current value of {top['value']}."
        )
    else:
        total_shap = sum(f["shap"] for f in feature_context.values())
        top = max(feature_context.values(), key=lambda x: abs(x["shap"]))
        answer = (
            f"{coin_name} market analysis: Overall signal is {'BULLISH 📈' if total_shap > 0 else 'BEARISH 📉'}. "
            f"The most influential factor is {top['label']} (SHAP: {top['shap']}, value: {top['value']}). "
            f"Social Sentiment: {feature_context['social_sentiment_score']['value']}, "
            f"News Sentiment: {feature_context['news_sentiment_score']['value']}, "
            f"Fear & Greed: {feature_context['fear_greed_index']['value']}."
        )

    return {"coin": coin_key, "question": data.question, "answer": answer}


# ======================================
# ✅ EMAIL NOTIFICATION ENDPOINT
# ======================================

class EmailAlertInput(BaseModel):
    to_email: str        # user's email from Firebase
    coin: str            # e.g. "bitcoin"
    target: float        # target price user set
    current: float       # current price
    status: str          # "ABOVE" or "BELOW"


@app.post("/send-alert-email")
def send_alert_email(data: EmailAlertInput):
    try:
        sender_email = "vishakhaingaleavcoe@gmail.com"
        sender_password = "mrxg agoi ovxq nzth"

        # ✅ Build email content
        subject = f"🚨 CryptoSense Alert — {data.coin.upper()} Price Alert Triggered"

        if data.status == "ABOVE":
            headline = f"🚀 {data.coin.upper()} moved ABOVE your target!"
            color = "#22c55e"
            emoji = "📈"
        else:
            headline = f"📉 {data.coin.upper()} dropped BELOW your target!"
            color = "#ef4444"
            emoji = "📉"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #ffffff; padding: 30px;">
            <div style="max-width: 500px; margin: auto; background-color: #1e293b; border-radius: 12px; padding: 30px;">
                <h1 style="color: #f97316;">CryptoSense {emoji}</h1>
                <h2 style="color: {color};">{headline}</h2>
                <table style="width: 100%; margin-top: 20px;">
                    <tr>
                        <td style="color: #94a3b8; padding: 8px 0;">Coin</td>
                        <td style="font-weight: bold;">{data.coin.upper()}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8; padding: 8px 0;">Your Target Price</td>
                        <td style="font-weight: bold;">${data.target:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8; padding: 8px 0;">Current Price</td>
                        <td style="font-weight: bold; color: {color};">${data.current:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8; padding: 8px 0;">Status</td>
                        <td style="font-weight: bold; color: {color};">{data.status}</td>
                    </tr>
                </table>
                <p style="color: #64748b; margin-top: 30px; font-size: 12px;">
                    This is a one-time notification from CryptoSense. 
                    You will not receive another email for this alert unless you reset it.
                </p>
            </div>
        </body>
        </html>
        """

        # ✅ Build MIME message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = data.to_email
        msg.attach(MIMEText(body, "html"))

        # ✅ Send via Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, data.to_email, msg.as_string())

        return {"success": True, "message": f"Email sent to {data.to_email}"}

    except Exception as e:
        print(f"Email error: {e}")
        return {"success": False, "message": str(e)}


# ======================================
# REAL-TIME CRYPTO PRICES
# ======================================

@app.get("/crypto-prices")
async def get_crypto_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,solana",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}