
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import openai
import requests
import gdown

st.set_page_config(page_title="Maize-Doc", layout="wide")

st.markdown("""
    <style>
    /* Universal Text Visibility: Deep Charcoal for maximum contrast */
    .stApp, .main, .stMarkdown, p, span, h1, h2, h3, h4, li, div {
        color: #000000 !important;
    }

    /* GLASSMORPHISM: Works on any background (Light or Dark) */
    .weather-card, .quadrant-box, .advice-box {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    }

    /* Specific fix for high-priority labels */
    b, strong, .conf-text {
        color: #000000 !important;
        text-shadow: 0px 0px 1px rgba(255,255,255,1); /* Adds a tiny glow to keep it sharp */
    }

    .stButton>button {
        border-radius: 20px;
        background-color: #1b5e20;
        color: #ffffff !important;
        font-weight: bold;
        border: none;
        width: 100%;
    }

    /* Progress bar track visibility on any background */
    .bar-bg {
        background-color: rgba(0, 0, 0, 0.1) !important;
        height: 12px;
        border-radius: 6px;
        border: 1px solid rgba(0, 0, 0, 0.2);
        overflow: hidden;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

CORN_CLASSES = ['Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass']

@st.cache_resource
def load_model():
    filename = 'maize_model.tflite'
    if not os.path.exists(filename):
        file_id = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def get_weather(city):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        if not geo_res.get('results'): return None
        lat, lon = geo_res['results'][0]['latitude'], geo_res['results'][0]['longitude']
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
        w_res = requests.get(w_url).json()
        curr = w_res['current']
        return {"temp": f"{curr['temperature_2m']}°C", "hum": f"{curr['relative_humidity_2m']}%", "prec": f"{curr['precipitation']}mm"}
    except: return None

def analyze_quadrants(image, interpreter):
    w, h = image.size; mid_w, mid_h = w // 2, h // 2
    quads = {
        "Top-Left (Q1)": image.crop((0, 0, mid_w, mid_h)),
        "Top-Right (Q2)": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left (Q3)": image.crop((0, mid_h, mid_w, h)),
        "Bottom-Right (Q4)": image.crop((mid_w, mid_h, w, h))
    }
    results = {}
    for name, img_crop in quads.items():
        preds = predict_image(img_crop, interpreter)
        idx = np.argmax(preds); conf = np.max(preds) * 100
        results[name] = {"label": CORN_CLASSES[idx], "conf": conf, "img": img_crop}
    return results

def get_smart_advice(disease, weed, weather, location):
    try:
        client = openai.OpenAI(api_key=st.secrets["openai_key"])
        w_txt = f"{weather['temp']}, Hum: {weather['hum']}" if weather else "Unknown"
        issue = f"{disease} and {weed}" if (disease and weed) else (disease or weed or "Infection")
        prompt = f"Expert Agronomist Plan for Maize in {location}. Issue: {issue}. Weather: {w_txt}. Use bold keys."
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e: return f"Advice Error: {e}"

st.sidebar.title("🌱 Maize-Doc")
user_city = st.sidebar.text_input("City Name", value="Vellore")
enable_ai = st.sidebar.checkbox("AI Support", value=True)

st.header("🌽 Maize Health Quadrant Analysis")
uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
    with c2:
        w_data = get_weather(user_city)
        if w_data:
            st.markdown(f"""
            <div class="weather-card">
                <h3 style="margin-top:0;">🌤️ {user_city} Weather</h3>
                <h2 style="font-size:45px; margin:10px 0;">{w_data['temp']}</h2>
                <p><b>Humidity: {w_data['hum']} | Precip: {w_data['prec']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        if st.button('🚀 Analyze Quadrants'):
            with st.spinner('Running Scan...'):
                model = load_model()
                res = analyze_quadrants(img, model)
                q_cols = st.columns(2)
                d_list, w_list = [], []
                for i, (name, val) in enumerate(res.items()):
                    with q_cols[i%2]:
                        st.image(val['img'], use_column_width=True)
                        lbl, cnf = val['label'], val['conf']
                        clr = "#1b5e20" if "Healthy" in lbl else ("#e65100" if "Weed" in lbl else "#b71c1c")
                        st.markdown(f"""
                        <div class="quadrant-box">
                            <b>{name}</b><br>
                            <b style="color:{clr} !important; font-size:20px;">{lbl}</b><br>
                            <div class="bar-bg">
                                <div style="width:{cnf}%; background-color:{clr}; height:12px; border-radius:6px;"></div>
                            </div>
                            <span class="conf-text"><b>{cnf:.1f}% Confidence</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                        if "Healthy" not in lbl:
                            if "Weed" in lbl: w_list.append(lbl)
                            else: d_list.append(lbl)
                
                f_d = max(set(d_list), key=d_list.count) if d_list else None
                f_w = max(set(w_list), key=w_list.count) if w_list else None
                if (f_d or f_w) and enable_ai:
                    with st.spinner("Consulting AI..."):
                        advice = get_smart_advice(f_d, f_w, w_data, user_city)
                        st.markdown(f'<div class="advice-box"><h3 style="margin-top:0; color:#01579b !important;">🤖 AI Prescription</h3><div><b>{advice.replace(chr(10), "<br>")}</b></div></div>', unsafe_allow_html=True)
