
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
    /* Force high contrast text colors */
    .stMarkdown, p, span, h1, h2, h3, h4, li {
        color: #1a1a1a !important;
    }
    
    .stButton>button {
        border-radius: 20px;
        background-color: #2e7d32;
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 10px 25px;
        width: 100%;
    }
    
    img { border-radius: 15px; }

    .weather-card {
        background-color: #ffffff !important;
        background: linear-gradient(135deg, #f1f8e9 0%, #ffffff 100%) !important;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border: 1px solid #c8e6c9;
        margin-bottom: 20px;
        text-align: center;
    }

    .quadrant-box {
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .advice-box {
        background-color: #ffffff !important;
        border-left: 8px solid #01579b;
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        color: #012d3a !important;
    }
    
    /* Small labels inside boxes */
    small {
        color: #555555 !important;
        font-weight: bold;
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
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
        w_res = requests.get(weather_url).json()
        current = w_res['current']
        return {
            "temp": f"{current['temperature_2m']}°C",
            "humidity": f"{current['relative_humidity_2m']}%",
            "precip": f"{current['precipitation']}mm",
            "icon": "🌤️" if current['precipitation'] == 0 else "🌧️"
        }
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
        w_info = f"{weather['temp']}, Humidity: {weather['humidity']}" if weather else "Weather unavailable"
        issue = f"{disease} and {weed}" if (disease and weed) else (disease or weed or "Infection")
        prompt = f"As an Agronomist, provide a treatment plan for Maize in {location}. Issue: {issue}. Weather: {w_info}. Include Diagnosis, Chemical Plan, and Organic Alternative. Use Bold for keys."
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e: return f"Advice unavailable: {e}"

st.sidebar.title("🌱 Maize-Doc")
user_city = st.sidebar.text_input("City Name", value="Vellore")
enable_ai = st.sidebar.checkbox("AI Support", value=True)

st.header("🌽 Maize Health Quadrant Analysis")
uploaded_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True, caption="Sample Image")
    
    with c2:
        w_data = get_weather(user_city)
        if w_data:
            st.markdown(f"""
            <div class="weather-card">
                <h3 style="color:#2e7d32 !important;">{w_data['icon']} {user_city} Weather</h3>
                <h2 style="color:#1b5e20 !important; margin:0; font-size:40px;">{w_data['temp']}</h2>
                <p style="color:#333 !important; font-weight:bold;">Humidity: {w_data['humidity']} | Precip: {w_data['precip']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Weather data unavailable for this city.")

        if st.button('🚀 Analyze Quadrants'):
            with st.spinner('Running AI Scan...'):
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
                            <small style="color:#555 !important;">{name}</small><br>
                            <b style="color:{clr} !important; font-size:18px;">{lbl}</b><br>
                            <div style="display:flex; align-items:center; margin-top:5px;">
                                <div style="flex-grow:1; background:#e0e0e0; height:12px; border-radius:6px; margin-right:10px;">
                                    <div style="width:{cnf}%; background:{clr}; height:12px; border-radius:6px;"></div>
                                </div>
                                <span style="font-size:14px; font-weight:bold; color:#333 !important;">{cnf:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "Healthy" not in lbl:
                            if "Weed" in lbl: w_list.append(lbl)
                            else: d_list.append(lbl)
                
                f_d = max(set(d_list), key=d_list.count) if d_list else None
                f_w = max(set(w_list), key=w_list.count) if w_list else None
                
                if f_d or f_w:
                    if enable_ai:
                        with st.spinner("Generating Treatment Plan..."):
                            advice = get_smart_advice(f_d, f_w, w_data, user_city)
                            st.markdown(f'<div class="advice-box"><h3 style="color:#01579b !important;">🤖 AI Prescription</h3><div style="color:#1a1a1a !important;">{advice.replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)
                else:
                    st.success("Analysis complete: Your crop is Healthy!")
