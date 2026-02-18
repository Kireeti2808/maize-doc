
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
    .stButton>button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 25px;
        width: 100%;
    }
    img { border-radius: 15px; }
    .weather-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
        border-radius: 25px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #b2ebf2;
        color: #333;
        display: flex;
        align-items: center;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    .weather-temp { font-size: 32px; font-weight: 800; color: #00796b; margin: 0; }
    .result-box {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .advice-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1565c0;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        line-height: 1.6;
        font-size: 16px;
        color: #0d47a1;
    }
    </style>
""", unsafe_allow_html=True)

CORN_CLASSES = ['Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass']

@st.cache_resource
def load_model():
    filename = 'maize_model.tflite'
    if not os.path.exists(filename):
        file_id = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
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

def get_weather(location):
    if not location: return None
    try:
        url = f"https://wttr.in/{location}?format=j1"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            desc = current['weatherDesc'][0]['value']
            icon = "☀️"
            if "rain" in desc.lower(): icon = "🌧️"
            elif "cloud" in desc.lower(): icon = "☁️"
            return {
                "temp": f"{current['temp_C']}°C",
                "humidity": f"{current['humidity']}%",
                "precip": f"{current['precipMM']}mm",
                "condition": desc,
                "icon": icon
            }
    except: return None
    return None

def analyze_quadrants(image, interpreter):
    w, h = image.size; mid_w, mid_h = w // 2, h // 2
    quads = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)),
        "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)),
        "Bottom-Right": image.crop((mid_w, mid_h, w, h))
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
        w_txt = f"Temp: {weather['temp']}, Hum: {weather['humidity']}, Rain: {weather['precip']}" if weather else "Weather unavailable"
        issue = f"{disease} and {weed}" if disease and weed else (disease or weed or "Healthy")
        prompt = f"Expert Agronomist Plan for Maize. Issue: {issue}. Location: {location}. Weather: {w_txt}. Provide: Diagnosis, Chemical Action Plan (safe for current weather?), and Organic Alternative. Bold key terms."
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e: return f"AI Offline. {e}"

st.sidebar.title("Maize-Doc")
user_location = st.sidebar.text_input("Location", value="Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

st.header("Maize Health Analysis")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
    with col2:
        w_data = get_weather(user_location)
        if w_data:
            st.markdown(f'<div class="weather-card"><div><span style="font-size:40px;">{w_data["icon"]}</span></div><div><p class="weather-temp">{w_data["temp"]}</p><small>{w_data["condition"]}</small></div><div><small>Hum: {w_data["humidity"]}<br>Rain: {w_data["precip"]}</small></div></div>', unsafe_allow_html=True)
        
        if st.button('Run Quadrant Diagnosis'):
            with st.spinner('Analyzing...'):
                model = load_model()
                res = analyze_quadrants(img, model)
                q_cols = st.columns(2)
                d_list, w_list = [], []
                for i, (name, val) in enumerate(res.items()):
                    with q_cols[i%2]:
                        st.image(val['img'], width=150)
                        lbl, cnf = val['label'], val['conf']
                        clr = "#28a745" if "Healthy" in lbl else ("#fd7e14" if "Weed" in lbl else "#dc3545")
                        st.markdown(f'<div class="result-box"><b style="color:{clr};">{lbl}</b><br><small>{cnf:.0f}%</small></div>', unsafe_allow_html=True)
                        if "Healthy" not in lbl:
                            if "Weed" in lbl: w_list.append(lbl)
                            else: d_list.append(lbl)
                
                f_d = max(set(d_list), key=d_list.count) if d_list else None
                f_w = max(set(w_list), key=w_list.count) if w_list else None
                
                if f_d or f_w:
                    if enable_ai:
                        with st.spinner("Consulting AI..."):
                            advice = get_smart_advice(f_d, f_w, w_data, user_location)
                            st.markdown(f'<div class="advice-box"><h4>🤖 AI Prescription</h4>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                else: st.success("Crop appears Healthy")
