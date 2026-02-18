
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import openai
import requests
import gdown

# --- PAGE CONFIG ---
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
    .result-box {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .advice-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1565c0;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        line-height: 1.6;
        color: #0d47a1;
    }
    </style>
""", unsafe_allow_html=True)

CORN_CLASSES = ['Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass']

# --- MODEL LOADING (Auto-Download from Drive) ---
@st.cache_resource
def load_model():
    folder = 'models'
    filename = os.path.join(folder, 'maize_model.tflite')
    
    os.makedirs(folder, exist_ok=True)

    # Always check if model exists, if not download
    if not os.path.exists(filename):
        file_id = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner("Downloading Model from Drive..."):
            gdown.download(url, filename, quiet=False)
            
    try:
        interpreter = tf.lite.Interpreter(model_path=filename)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def predict_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize and Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # EfficientNet Preprocessing (Standard)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- WEATHER FUNCTIONS ---
def get_weather(location):
    if not location: return None
    try:
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C&M"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.text.split("|")
            temp = data[0].strip()
            # Convert F to C if needed
            if "F" in temp:
                 val = float(''.join(filter(str.isdigit, temp)))
                 temp = f"{int((val - 32) * 5/9)}°C"
            
            # Simple icon mapping
            cond = data[3].strip().lower()
            icon = "☀️"
            if "rain" in cond: icon = "🌧️"
            elif "cloud" in cond: icon = "☁️"
            
            return {
                "temp": temp, "humidity": data[1].strip(), 
                "precip": data[2].strip(), "condition": data[3].strip(), "icon": icon
            }
    except: return None
    return None

# --- AI ADVICE (With Safety Check) ---
def get_smart_advice(predicted_label, weather, location):
    try:
        client = openai.OpenAI(api_key=st.secrets["openai_key"])
        
        # SAFEGUARD: Handle missing weather to prevent crash
        if weather:
            weather_txt = f"Temp: {weather['temp']}, Humidity: {weather['humidity']}, Rain: {weather['precip']}"
            hum = weather.get('humidity', 'Unknown')
            rain = weather.get('precip', 'Unknown')
        else:
            weather_txt = "Weather data unavailable"
            hum = "Unknown"
            rain = "Unknown"

        prompt = f"""
        You are an expert Agronomist AI.
        CROP: Maize (Corn)
        ISSUE DETECTED: {predicted_label}
        LOCATION: {location if location else "Unknown"}
        WEATHER: {weather_txt}
        
        TASK: Provide a concise management plan.
        1. **DIAGNOSIS**: Confirm the issue.
        2. **WEATHER IMPLICATION**: Can they spray chemicals given Humidity ({hum}) and Rain ({rain})?
        3. **ACTION PLAN**:
           - Recommend specific Fungicide/Herbicide.
        4. **ORGANIC OPTION**: One non-chemical solution.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"AI Agronomist is offline. Error: {e}"

# --- MAIN APP UI ---
st.sidebar.title("Maize-Doc")
st.sidebar.markdown("---")

# Default location to prevent empty input issues
user_location = st.sidebar.text_input("Location", value="Hyderabad") 
enable_ai = st.sidebar.checkbox("Enable AI Agronomist", value=True)

st.header("Maize Health Analysis")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Fetch weather
        weather_data = get_weather(user_location)
        
        if weather_data:
            st.markdown(f"""
            <div class="weather-card">
                <span style="font-size:40px; margin-right:15px;">{weather_data['icon']}</span>
                <div>
                    <h3 style="margin:0; color:#00796b;">{weather_data['temp']}</h3>
                    <div style="color:#666;">{weather_data['condition']}</div>
                    <small>Humidity: {weather_data['humidity']} | Rain: {weather_data['precip']}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Weather unavailable. Proceeding with analysis.")

        if st.button('Run Diagnosis'):
            with st.spinner('Scanning crop (Fast Mode)...'):
                interpreter = load_model()
                if interpreter:
                    # Single fast prediction
                    preds = predict_image(image, interpreter)
                    idx = np.argmax(preds)
                    confidence = np.max(preds) * 100
                    label = CORN_CLASSES[idx]
                    
                    # Colors
                    color = "#28a745" if "Healthy" in label else "#dc3545"
                    if "Weed" in label: color = "#fd7e14"
                    
                    # Display Result
                    st.markdown(f"""
                    <div class="result-box">
                        <h2 style="color:{color}; margin:0;">{label}</h2>
                        <p style="font-size:18px; color:#555;">Confidence: <b>{confidence:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI Advice
                    if enable_ai and "Healthy" not in label:
                        with st.spinner("Consulting AI Agronomist..."):
                            advice = get_smart_advice(label, weather_data, user_location)
                            st.markdown(f"""
                            <div class="advice-box">
                                <h4>🤖 AI Agronomist Prescription</h4>
                                {advice.replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                    elif "Healthy" in label:
                        st.success("✅ Crop is healthy! No action needed.")
