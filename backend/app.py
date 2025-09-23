from flask import Flask, request, jsonify
import requests
import google.generativeai as genai
from flask_cors import CORS
import re
from deep_translator import GoogleTranslator  
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import traceback
import joblib
import json
from geopy.geocoders import Nominatim
import random
import sqlite3
from sklearn.cluster import KMeans


app = Flask(__name__)
CORS(app)

# API Keys
WEATHER_API_KEY = 'f28861253e574c589d5111924242807'
GEMINI_API_KEY = 'AIzaSyDNOtokPHTUm9WCJ1pOPaweUp_Rks9DhjI'
UNSPLASH_ACCESS_KEY = 'YAd-Af7cIyfplIFBCWaKRL1XiNE6VsFULmx-ln-_HfY'


DB_NAME = "complaints.db"


# Initialize Gemini Model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app.config['UPLOAD_FOLDER'] = './uploads' 

class ModifiedSVR:
    def __init__(self, C=1.0, base_epsilon=0.05, delta=0.02, lambda1=0.0, lr=0.005,
                 n_iter=3000, batch_size=16, random_state=0, early_stopping=False, stop_patience=500):
        self.C = C
        self.base_epsilon = base_epsilon
        self.delta = delta
        self.lambda1 = lambda1
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)
        self.w = None
        self.b = 0.0
        self.early_stopping = early_stopping
        self.stop_patience = stop_patience

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        if self.w is None:
            self.w = np.zeros(n_features)
        m = np.zeros_like(self.w); v = np.zeros_like(self.w)
        mb = 0.0; vb = 0.0
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        best_val_loss = np.inf; best_wb = None; patience = 0

        for it in range(1, self.n_iter + 1):
            idx = self.rng.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
            Xb = X[idx]; yb = y[idx]
            preds = Xb.dot(self.w) + self.b
            resid = yb - preds
            t = np.abs(resid) - self.base_epsilon
            dL_df = np.where(t <= 0, 0.0, np.where(t <= self.delta, -(t/self.delta)*np.sign(resid), -np.sign(resid)))
            grad_w = self.w.copy()
            if self.lambda1 != 0.0:
                grad_w += self.lambda1 * (self.w / (np.sqrt(self.w**2 + 1e-8)))
            grad_w += self.C * (dL_df[:,None] * Xb).sum(axis=0)
            grad_b = self.C * dL_df.sum()

            # Adam updates
            m = beta1*m + (1-beta1)*grad_w
            v = beta2*v + (1-beta2)*(grad_w**2)
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            self.w -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            mb = beta1*mb + (1-beta1)*grad_b
            vb = beta2*vb + (1-beta2)*(grad_b**2)
            mb_hat = mb / (1 - beta1**it)
            vb_hat = vb / (1 - beta2**it)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

            # validation
            if (it % 200 == 0 or it == 1) and (X_val is not None and y_val is not None):
                vpred = X_val.dot(self.w) + self.b
                vr = y_val - vpred
                vt = np.abs(vr) - self.base_epsilon
                vloss_terms = np.where(vt <= 0, 0.0, np.where(vt <= self.delta, vt**2/(2*self.delta), vt - self.delta/2))
                val_loss = 0.5*np.sum(self.w**2) + self.lambda1*np.sum(np.sqrt(self.w**2 + 1e-8)) + self.C*np.sum(vloss_terms)
                if val_loss < best_val_loss - 1e-9:
                    best_val_loss = val_loss
                    best_wb = (self.w.copy(), self.b)
                    patience = 0
                else:
                    patience += 200
                if self.early_stopping and patience >= self.stop_patience:
                    break
        if best_wb is not None:
            self.w, self.b = best_wb

    def predict(self, X):
        return X.dot(self.w) + self.b

stacked_pipeline = joblib.load("full_stacked_pipeline.pkl")
model_columns = stacked_pipeline["model_columns"]

# Helper: convert scaled logy -> raw yield
def scaled_logy_to_raw(y_scaled_array):
    inv = stacked_pipeline["scaler_y"].inverse_transform(np.array(y_scaled_array).reshape(-1,1)).ravel()
    return np.expm1(inv)

# A simple mapping for state climate data (can be enhanced)
states_climate = {
    "Punjab":         {"AvgTemperature(C)": 25, "AnnualRainfall(mm)": 650},
    "Haryana":        {"AvgTemperature(C)": 26, "AnnualRainfall(mm)": 600},
    "Uttar Pradesh":  {"AvgTemperature(C)": 27, "AnnualRainfall(mm)": 900},
    "Maharashtra":    {"AvgTemperature(C)": 28, "AnnualRainfall(mm)": 1000},
    "West Bengal":    {"AvgTemperature(C)": 29, "AnnualRainfall(mm)": 1500},
}




def get_db_connection():
    """
    Creates and returns a SQLite connection to 'complaints.db'.
    """
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # So we get rows as dictionaries
    return conn

def init_db():
    """
    Ensures the 'complaints' table is created with these columns:
      id, text, latitude, longitude, embedding, cluster_id, created_at
    """
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            embedding TEXT,
            cluster_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def run_clustering():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, embedding, latitude, longitude FROM complaints").fetchall()
    
    vectors = []
    ids = []

    for row in rows:
        if row["embedding"] is None:
            continue
        try:
            emb = json.loads(row["embedding"])  # parse JSON list
        except:
            continue
        
        # Optionally combine lat/long
        lat = float(row["latitude"] or 0.0)
        lon = float(row["longitude"] or 0.0)

        # If you want lat/long to matter, do something like:
        # combined_vector = emb + [lat/5.0, lon/5.0]
        # The /5.0 is a rough normalization—tweak as needed
        combined_vector = emb

        vectors.append(combined_vector)
        ids.append(row["id"])

    if not vectors:
        conn.close()
        return 0

    X = np.array(vectors)

    # Suppose we want exactly 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    # Update DB
    for i, label in enumerate(labels):
        cid = int(label)
        complaint_id = ids[i]
        conn.execute("UPDATE complaints SET cluster_id=? WHERE id=?", (cid, complaint_id))

    conn.commit()
    conn.close()
    return 3  # We forced 3 clusters

init_db()  # Make sure 'complaints' table exists



@app.route('/recommendations', methods=['POST'])
def recommendations():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    category = data.get('category')
    lang = data.get('language', 'en')  # Default to English

    if not lat or not lon or not category:
        return jsonify({'error': 'Latitude, Longitude, and Category are required'}), 400

    weather_data = get_weather_data(lat, lon)
    if not weather_data:
        return jsonify({'error': 'Failed to get weather data'}), 500

    recommendations = get_crop_recommendations(weather_data, category, lang)
    if not recommendations:
        return jsonify({'error': 'Failed to get crop recommendations'}), 500

    return jsonify({"Recommendations": recommendations})

def get_crop_recommendations(weather_data, category, lang):
    try:
        query = f"Generate 5 crop recommendations for the category: {category} given the weather conditions: {weather_data}. Give me only the name with one line information."
        
        # AI Model Response
        response = model.generate_content(query)
        recommendations_text = response.text.strip()

        # Extract names and descriptions using regex
        pattern = re.compile(r'\d+\.\s*([^\:]+):\s*([^\n]+)')
        matches = pattern.findall(recommendations_text)

        translated_recommendations = []
        for match in matches:
            name = match[0].strip().replace('**', '')
            ename = name  # English name
            description = match[1].strip().replace('**', '')
            image = get_crop_image(name)

            if lang != 'en':
                name = translate_text(name, lang)
                description = translate_text(description, lang)

            translated_recommendations.append({
                'name': name,
                'ename': ename,
                'description': description,
                'image': image
            })

        return translated_recommendations
    except Exception as e:
        print(f"Error generating crop recommendations: {e}")
        return []

def translate_text(text, lang):
    """Translate text using Deep Translator."""
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def get_crop_image(crop_name):
    """Fetch crop images from Unsplash API."""
    url = f"https://api.unsplash.com/search/photos?query={crop_name}&client_id={UNSPLASH_ACCESS_KEY}&per_page=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results'][0]['urls']['small'] if data['results'] else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image for {crop_name}: {e}")
        return None

def get_weather_data(lat, lon):
    """Fetch weather data from WeatherAPI."""
    url = f'http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    
@app.route('/crop_steps', methods=['POST'])
def crop_steps():
    data = request.json
    crop_name = data.get('crop_name')
    lang = data.get('language', 'en')
    category = data.get('category')

    if not crop_name:
        return jsonify({'error': 'Crop name is required'}), 400
    
    if lang != 'en':
        crop_name = translate_text(crop_name, lang)

    try:
        if category:
            queries = [f"Give me a small paragraph on {category} for {crop_name} in language {lang}"]

        response = model.generate_content(queries)
        recommendations_text = response.text 
        return(recommendations_text)
    

    except Exception as e:
        print(f"Error generating crop growing steps: {e}")
        return jsonify({'error': 'Failed to get crop growing steps'}), 500

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    try:
        image = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(image_path.replace('\\','/'))
        
        # Open the image and convert it to bytes
        with open(image_path, 'rb') as img_file:
            img_bytes = img_file.read()

        # Send image to Gemini AI model for disease prediction
        response = genai.upload_file(path=image_path)  # Ensure this function is defined
        prompt = "Identify the plant disease in this image and provide the result in plain text."
        prediction_response = model.generate_content([response, prompt]) 
        
        match = re.search(r'\\(.?)\\*', prediction_response.text)
        
        if match:
            return jsonify({'prediction': match.group(1)})
        else:
            return jsonify({'prediction': prediction_response.text})
        
    except Exception as e:
        print(f"Error making prediction with Gemini AI: {e}")
        return jsonify({'error': 'Failed to make prediction'}), 500
    
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    lang = data.get('language', 'en')
    query = data.get('query')
    
    if lang != 'en':
        query = translate_text(query, lang)

    try:
        if query:
            queries = [f"Give me the answer for this query {query} in language {lang}. The query should be related to only farming related stuff. In case of any other irrelevant question just say something like I am KrishiSahayak and will answer to only farming related stuff. In case of queries like diseases of crops, ask them to go to CNN Enabled Plant Disease Identification of the same website KrishiVikas"]

        response = model.generate_content(queries)
        recommendations_text = response.text 
        
        recommendations_text = recommendations_text.replace("**","")
        recommendations_text = recommendations_text.replace("*","-")
        return(recommendations_text)

    except Exception as e:
        print(f"Error generating answers: {e}")
        return jsonify({'error': 'Failed to get answers'}), 500

def get_weather_forecast(lat, lon):
    """Fetch 7-day weather forecast from WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={lat},{lon}&days=7"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather forecast: {e}")
        return None

@app.route("/weather-forecast", methods=["POST"])
def weather_forecast():
    """API endpoint to fetch 7-day weather forecast."""
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    forecast_data = get_weather_forecast(lat, lon)
    if forecast_data:
        return jsonify(forecast_data)
    else:
        return jsonify({"error": "Failed to fetch weather forecast"}), 500

@app.route('/crop-calendar', methods=['POST'])
def generate_crop_calendar():
    try:
        # Parse the request data from the React frontend
        data = request.json
        if not data or "cropName" not in data or "latitude" not in data or "longitude" not in data:
            return jsonify({"message": "Invalid data provided"}), 400

        # Extract crop name and location details
        crop_name = data["cropName"].strip()
        latitude = data["latitude"]
        longitude = data["longitude"]
        lang=data["lang"]

        if not crop_name:
            return jsonify({"message": "Crop name is required"}), 400
        
        weather_data = get_weather_data(latitude, longitude)
        if not weather_data:
            return jsonify({'error': 'Failed to get weather data'}), 500

        # Generate a precise and structured AI prompt
        prompt = (
            f"You are an expert agronomist. Generate a **detailed crop calendar** for {crop_name} based on {weather_data} in this language {lang}."
            f"based on the location (Latitude: {latitude}, Longitude: {longitude}) in India. "
            f"Ensure it follows best farming practices suited for the region. The response should "
            f"be **structured and formatted** so that each stage is clearly separated. Use double asterisks (**) "
            f"to highlight stage titles. Keep each section detailed and concise.\n\n"
            f"### Crop Calendar for {crop_name} ({latitude}, {longitude})\n"
            f"1. **Land Preparation**\n   - Time:\n   - Activities:\n"
            f"2. **Seed Treatment**\n   - Time:\n   - Method:\n   - Chemicals Used:\n"
            f"3. **Sowing Period**\n   - Best Months:\n   - Method:\n"
            f"4. **Irrigation Schedule**\n   - Frequency:\n   - Best Practices:\n"
            f"5. **Fertilization Schedule**\n   - Types of Fertilizers:\n   - Application Timing:\n"
            f"6. **Weed Management**\n   - Techniques:\n   - Chemicals Used:\n"
            f"7. **Pest & Disease Management**\n   - Common Pests & Diseases:\n   - Control Methods:\n"
            f"8. **Harvesting Time**\n   - Month:\n   - Harvesting Methods:\n"
            f"9. **Post-Harvest Handling**\n   - Storage & Processing Tips:\n\n"
            f"Make sure the response follows this structured format so that each stage is **clearly extractable**."
        )

        # Generate response using AI model
        response = model.generate_content([prompt])
        crop_calendar_text = response.text
        crop_calendar_text = crop_calendar_text.replace("**","")
        crop_calendar_text = crop_calendar_text.replace("###","")
        

        # Respond to the frontend with the structured crop calendar
        return jsonify({"cropCalendar": crop_calendar_text}), 200

    except Exception as e:
        print(f"Error while generating crop calendar: {e}")
        return jsonify({"message": "Error generating crop calendar"}), 500
    
df = pd.read_csv('../data/indian_crop_sales_data.csv', parse_dates=['Date'])


crop_sums = df.groupby('Crop')['Quantity Sold (kg)'].sum().sort_values(ascending=False)
top_3_crops = crop_sums.head(3)
bottom_3_crops = crop_sums.tail(3)


@app.route('/best_worst_sellers', methods=['GET'])
def best_worst_sellers():
    """
    Returns the top 3 and bottom 3 crops by total quantity sold.
    """
    best_sellers_df = top_3_crops.reset_index().rename(columns={'Quantity Sold (kg)': 'TotalSales'})
    worst_sellers_df = bottom_3_crops.reset_index().rename(columns={'Quantity Sold (kg)': 'TotalSales'})
    
    response = {
        'best_sellers': best_sellers_df.to_dict(orient='records'),
        'worst_sellers': worst_sellers_df.to_dict(orient='records')
    }
    return jsonify(response)


@app.route('/forecast', methods=['GET'])
def forecast():
    """
    Generate future forecast for a selected crop using SARIMAX.
    Query params: /forecast?crop=Rice&periods=7
    """
    crop_name = request.args.get('crop', 'Rice')
    periods = request.args.get('periods', 7, type=int)
    
    # Filter data for selected crop
    crop_data = df[df['Crop'] == crop_name].copy()
    crop_data.sort_values(by='Date', inplace=True)
    crop_data.set_index('Date', inplace=True)

    if len(crop_data) < 5:
        return jsonify({'error': 'Not enough data to forecast for this crop.'}), 400

    y = crop_data['Quantity Sold (kg)']
    
    # A simple, placeholder SARIMAX configuration
    model = SARIMAX(y, 
                    order=(1,1,1),
                    seasonal_order=(1,1,1,7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Forecast for given periods
    forecast_values = results.predict(start=len(y), end=len(y)+periods-1, typ='levels').tolist()
    
    # Generate future dates
    last_date = crop_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]
    
    forecast_output = []
    for date, val in zip(future_dates, forecast_values):
        forecast_output.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Forecast': round(val, 2)
        })

    # Return last 7 days of actual data + forecast for easier charting
    historical_df = crop_data.tail(7).reset_index()
    historical_df['Date'] = historical_df['Date'].dt.strftime('%Y-%m-%d')
    historical_output = historical_df[['Date', 'Quantity Sold (kg)']]\
                                     .rename(columns={'Quantity Sold (kg)': 'Actual'})

    response = {
        'crop': crop_name,
        'historical': historical_output.to_dict(orient='records'),
        'forecast': forecast_output
    }
    return jsonify(response)


def get_state_from_coordinates(latitude, longitude):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        if location and "state" in location.raw["address"]:
            return location.raw["address"]["state"]
    except Exception as e:
        print("Geolocation error:", e)
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Safe float parser
        def safe_float(val):
            try:
                return float(val)
            except:
                return 0.0

        crop_type = data.get("cropType", "")
        land_size = safe_float(data.get("landSize"))
        fertilizer = safe_float(data.get("fertilizer"))
        pesticide = safe_float(data.get("pesticide"))
        latitude = safe_float(data.get("latitude"))
        longitude = safe_float(data.get("longitude"))
        year = safe_float(data.get("year", 2024))

        # Simple climate mapping
        states_climate = {
            "Punjab": {"AvgTemperature(C)": 25, "AnnualRainfall(mm)": 650},
            "Haryana": {"AvgTemperature(C)": 26, "AnnualRainfall(mm)": 600},
            "Uttar Pradesh": {"AvgTemperature(C)": 27, "AnnualRainfall(mm)": 900},
            "Maharashtra": {"AvgTemperature(C)": 28, "AnnualRainfall(mm)": 1000},
            "West Bengal": {"AvgTemperature(C)": 29, "AnnualRainfall(mm)": 1500},
        }

        state = get_state_from_coordinates(latitude, longitude)
        climate = states_climate.get(state, {"AvgTemperature(C)": 25, "AnnualRainfall(mm)": 700})

        # Build input dataframe
        input_dict = {
            "Year": [year],
            "LandSize(ha)": [land_size],
            "FertilizerUsage(kg_ha)": [fertilizer],
            "PesticideUsage(kg_ha)": [pesticide],
            "AvgTemperature(C)": [climate["AvgTemperature(C)"]],
            "AnnualRainfall(mm)": [climate["AnnualRainfall(mm)"]],
            "State_Haryana": [1 if state=="Haryana" else 0],
            "State_Maharashtra": [1 if state=="Maharashtra" else 0],
            "State_Punjab": [1 if state=="Punjab" else 0],
            "State_Uttar Pradesh": [1 if state=="Uttar Pradesh" else 0],
            "State_West Bengal": [1 if state=="West Bengal" else 0],
            "CropType_Maize": [1 if crop_type=="Maize" else 0],
            "CropType_Rice": [1 if crop_type=="Rice" else 0],
            "CropType_Wheat": [1 if crop_type=="Wheat" else 0],
        }

        input_df = pd.DataFrame(input_dict)
        # Ensure correct column order
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # -------------------------
        # Preprocess & predict using full stacked pipeline
        # -------------------------
        X_s1 = stacked_pipeline["scaler_X1"].transform(input_df)
        X_poly = stacked_pipeline["poly"].transform(X_s1)
        X_final = stacked_pipeline["scaler_X2"].transform(X_poly)

        # Base model predictions
        ridge_pred = stacked_pipeline["ridge"].predict(X_final)
        rf_pred = stacked_pipeline["rf"].predict(X_final)
        gb_pred = stacked_pipeline["gb"].predict(X_final)
        svr_pred = stacked_pipeline["svr"].predict(X_final)

        # Convert RF & GB to scaled logy
        rf_pred_scaled = stacked_pipeline["scaler_y"].transform(np.log1p(rf_pred).reshape(-1,1)).ravel()
        gb_pred_scaled = stacked_pipeline["scaler_y"].transform(np.log1p(gb_pred).reshape(-1,1)).ravel()

        # Meta input
        meta_input = np.column_stack([ridge_pred, rf_pred_scaled, gb_pred_scaled, svr_pred])
        stack_pred_scaled = stacked_pipeline["meta_model"].predict(meta_input)
        stack_pred_raw = scaled_logy_to_raw(stack_pred_scaled)

        return jsonify({"predicted_yield": round(float(stack_pred_raw[0]), 2)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


    

@app.route('/submit-complaint', methods=['POST'])
def submit_complaint():
    """
    Expects JSON:
    {
      "text": "My crops are failing...",
      "latitude": 30.1234,
      "longitude": 76.2345
    }
    1. Generates a random embedding (for demonstration).
    2. Stores complaint in DB with cluster_id=None (initially).
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    complaint_text = data.get("text", "")
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not complaint_text or lat is None or lon is None:
        return jsonify({"error": "Please provide text, latitude, and longitude"}), 400

    # Generate a random 5D embedding as a placeholder
    random_embedding = [round(random.uniform(-1, 1), 3) for _ in range(5)]
    embedding_json = json.dumps(random_embedding)

    # Insert into DB
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO complaints (text, latitude, longitude, embedding)
        VALUES (?, ?, ?, ?)
    """, (complaint_text, lat, lon, embedding_json))
    conn.commit()
    conn.close()

    return jsonify({"message": "Complaint submitted successfully"}), 200

@app.route('/run-clustering', methods=['POST'])
def do_clustering():
    num_clusters = run_clustering()
    return jsonify({"message": f"Clustering complete. #clusters={num_clusters}"}), 200

@app.route('/complaints', methods=['GET'])
def get_complaints():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM complaints").fetchall()
    conn.close()

    complaints = []
    for row in rows:
        complaints.append({
            "id": row["id"],
            "text": row["text"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "embedding": row["embedding"],
            "cluster_id": row["cluster_id"],
            "created_at": row["created_at"]
        })
    return jsonify(complaints)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
