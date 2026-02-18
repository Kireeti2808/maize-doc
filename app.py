
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
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #b2ebf2;
        color: #333;
        display: flex;
        align-items: center;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    .weather-temp { font-size: 28px; font-weight: 800; color: #00796b; margin: 0; }
    .quadrant-box {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .advice-box {
        background-color: #f1f8ff;
        border-left: 6px solid #0056b3;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #002d5a;
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
        # Using a more direct format to avoid JSON parsing errors if API is slow
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C"
        headers = {'User-Agent': 'curl/7.64.1'}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200 and "|" in response.text:
            data = response.text.split("|")
            temp = data[0].strip()
            # Safety check for temperature unit
            if "F" in temp:
                 val = float(''.join(filter(str.isdigit, temp)))
                 temp = f"{int((val - 32) * 5/9)}°C"
            return {
                "temp": temp,
                "humidity": data[1].strip(),
                "precip": data[2].strip(),
                "condition": data[3].strip(),
                "icon": "🌤️"
            }
    except: return None
    return None

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
        w_txt = f"{weather['temp']}, Hum: {weather['humidity']}" if weather else "Local weather unknown"
        issue = f"{disease} and {weed}" if (disease and weed) else (disease or weed or "Maize Crop")
        prompt = f"Expert Agronomist Plan for Maize. Location: {location}. Issue: {issue}. Weather: {w_txt}. Provide: Diagnosis, Action Plan, and Organic Alternative. Bold key terms."
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e: return f"Advice unavailable: {e}"

st.sidebar.title("🌱 Maize-Doc")
user_location = st.sidebar.text_input("Enter City", value="Vellore")
enable_ai = st.sidebar.checkbox("AI Prescription", value=True)

st.header("🌽 Maize Health Quadrant Analysis")
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True, caption="Original Field Sample")
    
    with col_info:
        w_data = get_weather(user_location)
        if w_data:
            st.markdown(f"""
            <div class="weather-card">
                <div style="font-size:35px;">{w_data['icon']}</div>
                <div><p class="weather-temp">{w_data['temp']}</p><small>{w_data['condition']}</small></div>
                <div style="font-size:12px; border-left:1px solid #ccc; padding-left:10px;">
                    Hum: {w_data['humidity']}<br>Rain: {w_data['precip']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("📡 Weather station busy. Check location name.")

        if st.button('🚀 Start Deep Scan'):
            with st.spinner('Dividing image into quadrants and analyzing...'):
                model = load_model()
                res = analyze_quadrants(img, model)
                
                st.subheader("Quadrant Results")
                q_cols = st.columns(2)
                d_list, w_list = [], []

                for i, (name, val) in enumerate(res.items()):
                    with q_cols[i%2]:
                        st.image(val['img'], use_column_width=True)
                        lbl, cnf = val['label'], val['conf']
                        clr = "#28a745" if "Healthy" in lbl else ("#fd7e14" if "Weed" in lbl else "#dc3545")
                        
                        st.markdown(f"""
                        <div class="quadrant-box">
                            <small>{name}</small><br>
                            <b style="color:{clr}; font-size:16px;">{lbl}</b><br>
                            <div style="display:flex; align-items:center;">
                                <div style="flex-grow:1; background:#e0e0e0; height:8px; border-radius:4px; margin-right:10px;">
                                    <div style="width:{cnf}%; background:{clr}; height:8px; border-radius:4px;"></div>
                                </div>
                                <span style="font-size:12px; font-weight:bold;">{cnf:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "Healthy" not in lbl:
                            if "Weed" in lbl: w_list.append(lbl)
                            else: d_list.append(lbl)
                
                f_d = max(set(d_list), key=d_list.count) if d_list else None
                f_w = max(set(w_list), key=w_list.count) if w_list else None
                
                if f_d or f_w:
                    st.divider()
                    if enable_ai:
                        with st.spinner("Consulting AI Agronomist..."):
                            advice = get_smart_advice(f_d, f_w, w_data, user_location)
                            st.markdown(f'<div class="advice-box"><h3>🤖 Treatment Plan</h3>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                else:
                    st.balloons()
                    st.success("Analysis complete: Crop is 100% Healthy!")
