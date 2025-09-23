# 🌾 Krishi Vikas  - An AI-Powered Farmer Assist Platform
---

Krishi Vikas is an AI-powered agricultural platform built to support small and marginal farmers.
It offers personalized crop advice, disease diagnostics, market forecasts, and income alternatives using cutting-edge ML and NLP.
Designed for inclusivity, the platform ensures accessibility through multilingual and voice-enabled interfaces.

## 📚 Table of Contents

| Section | Description |
|--------|-------------|
|1. [ Overview](#overview) | Introduction to Krishi Vikas and its mission |
|2. [ Problem Statement](#problem-statement) | Challenges faced by small and marginal farmers |
|3. [ Features](#features) | Overview of all AI-powered modules like crop advisory, disease detection, and more |
|4. [ Architecture](#architecture) | System design and data flow architecture |
|5. [ Diversity & Tech Stack](#-why-diversity-is-our-topmost-priority) | Accessibility focus and complete tech stack overview |
|6. [ Installation](#installation) | Step-by-step guide to set up backend and frontend |
|7. [ Competitor Analysis](#-competitor-analysis) | Comparison with top AgriTech platforms |
|8. [ Business Model & Research Overview](#-business-model-and-research-overview) | Strategy, monetization, and research-backed innovation |

---

## Overview
**Krishi Vikas** is an end-to-end, AI-driven platform designed to empower small and marginal farmers with smart, data-backed advice at every stage of the crop lifecycle. From choosing the right crop variety for your agro-climatic zone to diagnosing plant diseases, forecasting yield and price, reporting in-field issues, and suggesting alternative income streams—Krishi Vikas brings cutting-edge ML/AI tools into the hands of those who need them most.

---

## Problem Statement

<div align="center">

 <img src="https://github.com/user-attachments/assets/cafffacd-a5e9-49f1-a08e-f025cd1ac6f8"  width="700"/>

</div>

Farmers, especially those managing less than 2 hectares of land, face multiple challenges:

1. **Sub-optimal Crop Selection**  
   Traditional intuition often drives crop choice, ignoring soil, weather, and market factors.

2. **Late or Missed Disease Diagnosis**  
   Visual symptoms go unnoticed or misidentified, leading to crop losses.

3. **Unpredictable Yield & Storage Planning**  
   Lack of accurate yield forecasts hampers storage, finance, and resource planning.

4. **Poor Post-Harvest Price Forecasting**  
   No reliable market trend data leads to sub-optimal selling times.

5. **Invisible In-Field Issues**  
   Pests, nutrient deficiencies, and flooding incidents often remain unreported and unaddressed.

6. **Income Instability**  
   Crop failures leave farmers without alternative livelihood guidance.

---

## Features
<div align="center">

 <img src="https://github.com/user-attachments/assets/b4ab4cfe-12a2-4423-866d-dadc41029e8a"  width="700"/>

</div>


### 1️⃣ Agro-Climatic Crop Recommendation

**Description:**  
Recommends the most suitable crop varieties for a farmer’s region based on agro-climatic conditions like temperature, rainfall, soil type, and seasonal variations.

**Tech Stack:**
- **NLP:** AgriBERT  
- **Frontend:** React, Ionic  
- **Backend:** Flask  
- **External APIs:** Weather APIs, Soil databases  

**How It Works:**
- AgriBERT interprets location-specific agricultural text and historical patterns.
- Real-time weather and soil data are parsed to identify optimal crop matches.
- Recommendations are visualized via an interactive crop calendar with planting/harvesting schedules.

---

### 2️⃣ Plant Disease Identification

**Description:**  
Allows farmers to upload images of infected crops/leaves to instantly detect diseases and receive treatment advice.

**Tech Stack:**
- **Deep Learning:** Convolutional Neural Networks (CNN with TensorFlow)  
- **Frontend:** Image Upload Interface (React/Ionic)  
- **Backend:** Flask API  
- **Model Training:** ImageNet-pretrained CNN fine-tuned on plant disease datasets  

**How It Works:**
- User uploads a photo of the plant or leaf.
- The image is processed by a CNN classifier trained to detect over 30 common crop diseases.
- Based on diagnosis, the system provides recommended pesticide and organic treatment options.

---

### 3️⃣ Precision Yield Forecasting

**Description:**  
Predicts expected crop yield per hectare using key farm-level inputs such as area, fertilizer usage, crop type, and weather data.

**Tech Stack:**
- **Machine Learning:** Support Vector Machines (SVM via scikit-learn)  
- **Backend:** Flask  
- **Frontend:** React dashboards  
- **Data Inputs:** Historical yield records, weather APIs, fertilizer logs  

**How It Works:**
- SVM regression models trained on multi-year agricultural datasets forecast yield potential.
- Forecasts are visualized for seasonal planning and irrigation scheduling.
- Suggestions for crop rotation and input optimization are provided alongside.

---

### 4️⃣ Market Price Forecasting

**Description:**  
Provides short and mid-term predictions of crop prices using time-series models to help farmers choose optimal selling times.

**Tech Stack:**
- **Time-Series Modeling:** SARIMA (statsmodels)  
- **Frontend:** Dynamic D3.js/Chart.js visualizations  
- **Backend:** Flask API  
- **Data Source:** Historical mandi prices, open agri-market data  

**How It Works:**
- SARIMA models analyze seasonal patterns in market price fluctuations.
- Users can explore price trends over days/weeks/months for selected crops.
- Forecasts are combined with region-specific market demand insights.

---

### 5️⃣ Crowdsourced Problem Reporting

**Description:**  
Allows farmers to report localized problems like pest outbreaks, nutrient deficiencies, or flood damage, which are clustered and prioritized using geospatial intelligence.

**Tech Stack:**
- **Clustering Algorithms:** DBSCAN for geo-tag clustering  
- **Categorization:** K-Means + NLP classifiers (Scikit-learn)  
- **Severity Scoring:** Hybrid Rule-Based + ML classification  
- **Frontend:** Map-based UI (Leaflet.js)  
- **Backend:** Flask  

**How It Works:**
- Reports submitted by farmers are geotagged and categorized using K-Means and NLP keyword extraction.
- DBSCAN clusters incidents to detect regional problem hotspots.
- Severity is auto-assessed to prioritize government or NGO response.

---

### 6️⃣ KrishiSahayak Chatbot – Alternative Income Advisor

**Description:**  
Conversational AI chatbot that guides farmers on diversifying income through activities like dairy, poultry, apiculture, and agro-tourism.

**Tech Stack:**
- **NLP Model:** Multilingual BERT (mBERT)  
- **TTS:** Google Text-to-Speech API  
- **Voice Interface:** Web Speech API / Ionic Native Plugins  
- **Backend:** Flask  
- **Frontend:** Chat UI (React/Ionic)  

**How It Works:**
- Farmers interact in their local language via text or voice.
- mBERT processes queries and maps them to curated income options and resources.
- TTS enables accessibility for non-literate users by reading chatbot responses aloud.  
---

## Architecture

<div align="center">

 <img src="https://github.com/user-attachments/assets/5a2f5bb6-0d59-4ec2-b27d-1cb994894183"  width="700"/>

</div>

**Flow:**  
1. **Frontend** (React Web + Ionic Mobile)  
2. **Backend** (Flask REST API)  
3. **ML/AI Models** (TensorFlow, scikit-learn, statsmodels)  
4. **Database** for user, farm, and report data  
5. **External APIs** for weather, geolocation, and TTS  


---

## 🌐 Why Diversity is Our Topmost Priority

We focus on inclusivity and accessibility through multilingual support, personalized user targeting, and a tech stack that ensures adaptability across use cases.


<div align="center">
  <img src="https://github.com/user-attachments/assets/2dfc3788-2f11-48db-a93f-94926160e2ec" alt="Diversity and Tech Stack Slide" width="700"/>
</div>

### 🔑 Key Focus Areas:
- **Multilingual Support** with `mBERT` for better regional understanding
- **Voice Assistant** using **Google Text to Speech** for non-literate users
- **User-Centric Design** mapping advice to regional weather and soil types
- **Diverse Feature Set** catering to every farmer's needs

### ⚙️ Tech Stack Overview:
- **Frontend:** React (Web), Ionic (Mobile)
- **Backend:** Flask
- **ML & AI Models:** TensorFlow, Scikit-learn, Python, Statsmodels

> This diverse and inclusive technology stack allows us to reach more users with tailored solutions.

---


## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/JainSneha6/KrishiVikas.git
   cd KrishiVikas
   ```
2. **Backend Setup**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   flask run --host=0.0.0.0 --port=5000
   ```
3. **Frontend Setup**
   ```bash
   cd ../frontend
   npm install
   npm start      # React web on http://localhost:3000
   ionic serve    # Mobile preview
   ```

---


## 📊 Competitor Analysis

We evaluated Krishi-Vikas alongside top AgriTech players—Plantix, Kindwise, and Cropin—across five core features. Our product stands out with comprehensive AI-driven solutions tailored for farmers' diverse needs.


<div align="center">
  <img src="https://github.com/user-attachments/assets/e4531ddb-816f-4f37-964e-6b364bc2c5d8" alt="Competitor Analysis Slide" width="700"/>
</div>

### ✅ Key Advantages of Krishi-Vikas:
- Full-stack AI coverage: from **crop recommendation** to **income advisory**
- Uniquely offers **geo-spatial issue mapping** using crowdsourced data
- Combines **CNN**, **time-series**, and **chatbot tech** in one solution

Krishi-Vikas clearly leads in providing an all-in-one platform for small and marginal farmers.

---


## 🧩 Business Model and Research Overview

The following slide presents our business model and research foundation for **Krishi Vikas**:

<div align="center">
  <img src="https://github.com/user-attachments/assets/a99c00f5-84e0-450f-9601-6586217422bd" alt="Business Model and Research Slide" width="700"/>
</div>

 **Key Highlights**  
 - Targeting 86% of Indian farmers (small & marginal).  
 - Cost-efficient model focusing on cloud and marketing.  
 - Research backed by cutting-edge models like AgriBERT, FinBERT, SARIMA, and CNN.  
 - Monetization through B2C access and ad revenue.

### 🔬 Research References  

- **Why AgriBERT?**  
  [Exploring New Frontiers in Agricultural NLP](https://ieeexplore.ieee.org/abstract/document/10637955)

- **Why finBERT?**  
  [Financial Sentiment Analysis on News and Reports Using FinBERT](https://ieeexplore.ieee.org/abstract/document/10796670)

- **Why SARIMA?**  
  [Study and Analysis of SARIMA and LSTM in Forecasting Time Series Data](https://www.sciencedirect.com/science/article/abs/pii/S2213138821004847)

- **Why CNN?**  
  [Classification of Plant Diseases Using Pretrained CNN on ImageNet](https://openagriculturejournal.com/VOLUME/18/ELOCATOR/e18743315305194/FULLTEXT/)


---


