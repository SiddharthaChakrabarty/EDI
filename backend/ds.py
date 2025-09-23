import csv
import random
import numpy as np

# Define possible states and approximate climate data
# (In a real scenario, lat/long would more precisely map to the state and climate.)
states_info = [
    {
        "state": "Punjab", 
        "lat_range": (30.0, 31.5),
        "lon_range": (74.0, 76.0),
        "avg_temp": 25,  # degrees Celsius
        "annual_rainfall": 650  # mm
    },
    {
        "state": "Haryana", 
        "lat_range": (28.5, 30.5),
        "lon_range": (76.0, 77.5),
        "avg_temp": 26,
        "annual_rainfall": 600
    },
    {
        "state": "Uttar Pradesh", 
        "lat_range": (25.0, 28.0),
        "lon_range": (80.0, 83.0),
        "avg_temp": 27,
        "annual_rainfall": 900
    },
    {
        "state": "Maharashtra", 
        "lat_range": (18.0, 20.0),
        "lon_range": (73.0, 75.0),
        "avg_temp": 28,
        "annual_rainfall": 1000
    },
    {
        "state": "West Bengal", 
        "lat_range": (22.0, 24.0),
        "lon_range": (87.0, 89.0),
        "avg_temp": 29,
        "annual_rainfall": 1500
    },
]

crops = ["Rice", "Wheat", "Maize"]
years = list(range(2015, 2025))  # 2015 to 2024

# Decide how many rows you want
num_samples = 1000

with open("crop_yield_dataset.csv", mode='w', newline='') as csv_file:
    fieldnames = [
        "State", "CropType", "Year", "LandSize(ha)", 
        "FertilizerUsage(kg_ha)", "PesticideUsage(kg_ha)",
        "AvgTemperature(C)", "AnnualRainfall(mm)",
        "Latitude", "Longitude", "Yield(kg_ha)"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    for _ in range(num_samples):
        state_data = random.choice(states_info)
        
        # Random lat/long within the state range
        lat = round(random.uniform(*state_data["lat_range"]), 4)
        lon = round(random.uniform(*state_data["lon_range"]), 4)
        
        crop = random.choice(crops)
        year = random.choice(years)
        
        land_size = round(random.uniform(1, 10), 2)
        fertilizer = round(random.uniform(50, 300), 2)
        pesticide = round(random.uniform(1, 50), 2)
        
        # Slight random variations around the state's average climate
        avg_temp = round(state_data["avg_temp"] + random.uniform(-2, 2), 2)
        annual_rainfall = round(state_data["annual_rainfall"] + random.uniform(-100, 100), 2)
        
        # Simplified yield formula (you can refine):
        # Base yield depends on crop type, plus factors for fertilizer, pesticide, climate
        base_yield_dict = {"Rice": 3000, "Wheat": 2800, "Maize": 2500}
        base_yield = base_yield_dict[crop]
        
        # Influence from fertilizer and pesticide (random coefficients)
        fert_influence = 0.5 * fertilizer  
        pest_influence = 1.0 * pesticide  
        
        # Influence from temperature and rainfall
        temp_factor = -10 * abs(avg_temp - 25)  # Far from optimal 25°C => negative effect
        rain_factor = -0.5 * abs(annual_rainfall - 800)  # difference from some ideal
         
        # Combine influences
        predicted_yield = base_yield + fert_influence + pest_influence + temp_factor + rain_factor
        
        # Random noise
        predicted_yield += random.uniform(-200, 200)
        
        # Ensure yield is within a plausible range
        if predicted_yield < 800:
            predicted_yield = 800 + random.uniform(0, 100)
        if predicted_yield > 8000:
            predicted_yield = 8000 - random.uniform(0, 100)
        
        writer.writerow({
            "State": state_data["state"],
            "CropType": crop,
            "Year": year,
            "LandSize(ha)": land_size,
            "FertilizerUsage(kg_ha)": fertilizer,
            "PesticideUsage(kg_ha)": pesticide,
            "AvgTemperature(C)": avg_temp,
            "AnnualRainfall(mm)": annual_rainfall,
            "Latitude": lat,
            "Longitude": lon,
            "Yield(kg_ha)": round(predicted_yield, 2)
        })

print("Synthetic dataset generated: crop_yield_dataset.csv")

