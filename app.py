
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import openai
import requests

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
    img {
        border-radius: 15px;
    }
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
    .weather-icon {
        font-size: 45px;
        margin-right: 15px;
    }
    .weather-temp {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        color: #00796b;
    }
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
    filename = 'models/maize_model.tflite'
    if not os.path.exists(filename):
        st.error(f"Model file not found: {filename}")
        return None
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

def get_weather_emoji(condition):
    condition = condition.lower()
    if "sun" in condition or "clear" in condition: return "☀️"
    elif "rain" in condition or "shower" in condition or "drizzle" in condition: return "🌧️"
    elif "cloud" in condition: return "☁️"
    elif "storm" in condition or "thunder" in condition: return "⛈️"
    elif "snow" in condition: return "❄️"
    else: return "🌤️"

def get_weather(location):
    if not location: return None
    try:
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C&M"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            temp = data[0].strip()
            if "F" in temp:
                 val = float(''.join(filter(str.isdigit, temp)))
                 temp = f"{int((val - 32) * 5/9)}°C"
            return {
                "temp": temp, 
                "humidity": data[1].strip(), 
                "precip": data[2].strip(), 
                "condition": data[3].strip(),
                "icon": get_weather_emoji(data[3].strip())
            }
    except: return None
    return None

def analyze_quadrants(image, interpreter):
    w, h = image.size; mid_w, mid_h = w // 2, h // 2
    quadrants = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)), "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)), "Bottom-Right": image.crop((mid_w, mid_h, w, h))
    }
    results = {}
    for name, img_crop in quadrants.items():
        preds = predict_image(img_crop, interpreter)
        idx = np.argmax(preds); conf = np.max(preds) * 100
        results[name] = {"label": CORN_CLASSES[idx], "conf": conf, "img": img_crop}
    return results

def get_smart_advice(disease_detected, weed_detected, weather, location):
    try:
        client = openai.OpenAI(api_key=st.secrets["openai_key"])
        weather_txt = "Unknown"
        if weather:
            weather_txt = f"Temp: {weather['temp']}, Humidity: {weather['humidity']}, Rain: {weather['precip']}, Sky: {weather['condition']}"

        issues = []
        if disease_detected: issues.append(f"Disease ({disease_detected})")
        if weed_detected: issues.append(f"Weed ({weed_detected})")
        diagnosis_str = " AND ".join(issues) if issues else "Healthy Crop"

        prompt = f"""
        You are an expert Agronomist AI.
        CROP: Maize (Corn)
        ISSUES DETECTED: {diagnosis_str}
        LOCATION: {location}
        CURRENT WEATHER: {weather_txt}
        
        TASK: Provide a comprehensive management plan.
        1. **DIAGNOSIS SUMMARY**: Briefly confirm what was found.
        2. **WEATHER CHECK**: Explicitly analyze if the current humidity ({weather.get('humidity', 'N/A')}) and rainfall ({weather.get('precip', 'N/A')}) make it safe to spray chemicals.
        3. **ACTION PLAN**:
           - If a DISEASE is present: Recommend a specific Fungicide/Bactericide.
           - If a WEED is present: Recommend a specific Herbicide.
           - *Important*: If both are present, mention if these chemicals can be mixed or need separate application.
        4. **ORGANIC ALTERNATIVE**: One non-chemical solution for the issues.
        
        Keep it structured, concise, and use bold formatting for key terms. Avoid using markdown points like '-'.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"AI Agronomist is offline. Error: {e}"

st.sidebar.title("Maize-Doc")
st.sidebar.markdown("---")

user_location = st.sidebar.text_input("Location", placeholder="e.g. Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Agronomist", value=True)

st.header("Maize Health Analysis")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        weather_data = get_weather(user_location)
        if weather_data:
            st.markdown(f"""
            <div class="weather-card">
                <div style="display:flex; align-items:center;">
                    <span class="weather-icon">{weather_data['icon']}</span>
                    <div>
                        <p class="weather-temp">{weather_data['temp']}</p>
                        <span style="color:#666; font-size:14px;">{weather_data['condition']}</span>
                    </div>
                </div>
                <div style="font-size:13px; border-left:1px solid #ddd; padding-left:20px;">
                    <div>Humidity: {weather_data['humidity']}</div>
                    <div>Rainfall: {weather_data['precip']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button('Run Diagnosis'):
            with st.spinner('Analyzing crop health...'):
                try:
                    interpreter = load_model()
                    if interpreter:
                        quad_results = analyze_quadrants(image, interpreter)
                        
                        st.write("### Quadrant Analysis")
                        q1, q2 = st.columns(2)
                        
                        detected_diseases = []
                        detected_weeds = []
                        
                        for i, (name, res) in enumerate(quad_results.items()):
                            target_col = q1 if i % 2 == 0 else q2
                            with target_col:
                                st.image(res['img'], width=150)
                                label = res['label']; conf = res['conf']
                                color = "#dc3545"
                                
                                is_weed = "Weed" in label
                                is_healthy = "Healthy" in label
                                
                                if is_healthy: color = "#28a745"
                                if is_weed: color = "#fd7e14"
                                
                                st.markdown(f"""
                                    <div class="result-box">
                                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                            <span style="font-weight:bold; color:{color};">{label}</span>
                                            <span style="color:#666; font-size:12px;">{conf:.0f}%</span>
                                        </div>
                                        <div style="width:100%; background:#e0e0e0; border-radius:5px; height:8px;">
                                            <div style="width:{conf}%; background:{color}; height:8px; border-radius:5px;"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if conf > 50:
                                    if is_healthy: pass
                                    elif is_weed: detected_weeds.append(label)
                                    else: detected_diseases.append(label)
                        
                        final_disease = max(set(detected_diseases), key=detected_diseases.count) if detected_diseases else None
                        final_weed = max(set(detected_weeds), key=detected_weeds.count) if detected_weeds else None
                        
                        if final_disease and final_weed:
                            st.error(f"Multiple Issues Detected: {final_disease} AND {final_weed}")
                        elif final_disease:
                            st.error(f"Disease Detected: {final_disease}")
                        elif final_weed:
                            st.warning(f"Weed Detected: {final_weed}")
                        else:
                            st.success("Crop appears Healthy")

                        if enable_ai:
                            if final_disease or final_weed:
                                with st.spinner("Consulting AI Agronomist..."):
                                    advice = get_smart_advice(final_disease, final_weed, weather_data, user_location)
                                    st.markdown(f"""
                                    <div class="advice-box">
                                        <h4 style="margin-top:0; color:#1565c0;">AI Agronomist Prescription</h4>
                                        {advice.replace(chr(10), '<br>')}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("Crop is healthy. No medical prescription needed. Keep monitoring!")
                except Exception as e: st.error(f"Error: {str(e)}")
