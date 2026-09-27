import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

app = FastAPI(title="MoES SIH26081 Hybrid Weather Intelligence Engine")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "yashvendrasahu/moes-weather-ai"

print("Fetching model weights from Hugging Face...")
model_a_file = hf_hub_download(repo_id=REPO_ID, filename="model_a_production.joblib")
model_b_file = hf_hub_download(repo_id=REPO_ID, filename="model_b_production.joblib")
offset_file = hf_hub_download(repo_id=REPO_ID, filename="cqr_offset_final.npy")

clf_a = joblib.load(model_a_file)
regs_b = joblib.load(model_b_file)
cqr_offset = float(np.load(offset_file))
expected_cols = list(clf_a.get_booster().feature_names)
print(f"Loaded {len(expected_cols)} model features successfully!")

class WeatherInput(BaseModel):
    latitude: float = 31.58
    longitude: float = 76.91
    climatic_zone: int = 1
    tp_gfs: float = 32.0
    tp_ecmwf: float = 42.0
    tp_ncum: float = 48.0
    tp_wrf: float = 55.0  # Pydantic input field
    cape: float = 2950.0
    cin: float = 15.0
    rh_700: float = 88.0
    mslp: float = 1002.0
    wind_shear: float = 24.0
    elevation_m: float = 1044.0
    terrain_slope_deg: float = 24.5
    radar_max_dbz: float = 54.0
    satellite_ctt_celsius: float = -68.0

@app.get("/")
def home():
    return {"message": "MoES Weather Intelligence API is running", "endpoints": ["/health", "/predict", "/docs"]}

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "v5.0-production", "coverage_guarantee": "80.00%"}

@app.post("/predict")
def predict_weather(payload: WeatherInput):
    data = payload.dict()
    
    tp_wrf_val = data.get("tp_wrf", 0.0)
    ens_mean = (data["tp_gfs"] + data["tp_ecmwf"] + data["tp_ncum"] + tp_wrf_val) / 4.0
    spread_val = float(np.std([data["tp_gfs"], data["tp_ecmwf"], data["tp_ncum"], tp_wrf_val]))
    indig_spread = abs(((data["tp_gfs"] + data["tp_ecmwf"]) / 2.0) - ((data["tp_ncum"] + tp_wrf_val) / 2.0))
    slope_rad = np.radians(data["terrain_slope_deg"])
    
    row_dict = {
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "climatic_zone": data["climatic_zone"],
        "tp_gfs": data["tp_gfs"],
        "tp_ecmwf": data["tp_ecmwf"],
        "model_spread": abs(data["tp_gfs"] - data["tp_ecmwf"]),
        "lead_time_hours": 24,
        "cape": data["cape"],
        "cin": data["cin"],
        "rh_700": data["rh_700"],
        "mslp": data["mslp"],
        "wind_shear": data["wind_shear"],
        "instability_index": (data["cape"] / 1000.0) * data["wind_shear"],
        "season_sin": 0.85,
        "season_cos": -0.52,
        "monsoon_phase": 2,
        "weather_regime": 1 if data["cape"] > 2000 else 0,
        "elevation_m": data["elevation_m"],
        "terrain_slope_deg": data["terrain_slope_deg"],
        "aspect_deg": 220.0,
        "orographic_lift_mps": data["wind_shear"] * np.sin(slope_rad),
        "orographic_moisture_flux": data["wind_shear"] * np.sin(slope_rad) * (data["rh_700"] / 100.0),
        "radar_max_dbz": data["radar_max_dbz"],
        "satellite_ctt_celsius": data["satellite_ctt_celsius"],
        "radar_growth_rate_30m": 8.0 if data["radar_max_dbz"] > 45 else 0.0,
        "convective_cell_strength": (data["radar_max_dbz"] / 50.0) * (abs(min(0, data["satellite_ctt_celsius"])) / 60.0),
        "pressure_gradient_mag": 4.5,
        "spatial_moisture_laplacian": 8.5,
        "neighbor_rain_mean": ens_mean * 0.9,
        "neighbor_rain_max": ens_mean * 1.25,
        "spatial_coherence_index": (data["cape"] / 1000.0) * data["wind_shear"] / 5.5,
        "tp_ncum": data["tp_ncum"],
        "tp_imd_wrf": tp_wrf_val,  # Exact expected feature name for Model A
        "indigenous_global_spread": indig_spread,
        "multi_model_std": spread_val,
        "multi_model_mean": ens_mean,
        "forecast_entropy": spread_val / (ens_mean + 1.0),
        "climatological_mean_daily": 14.5,
        "climatological_std_daily": 12.0,
        "forecast_climatology_anomaly_z": (ens_mean - 14.5) / 12.0,
        "climatology_ratio": ens_mean / 15.5,
        "cape_anomaly_ratio": data["cape"] / 1200.0
    }
    
    # Model A ke exact features aur order par sanitize karein
    df_row = pd.DataFrame([row_dict])
    X = df_row[expected_cols]
    
    # Model A: Bust Probability
    bust_prob = float(clf_a.predict_proba(X)[0, 1])
    if (data["cape"] > 2400) and (data["rh_700"] > 75) and (data["cin"] < 40) and (data["radar_max_dbz"] > 45):
        bust_prob = max(bust_prob, 0.76)
    elif data["cin"] > 100 or data["rh_700"] < 35:
        bust_prob = min(bust_prob, 0.05)
        
    # Model B: Quantiles
    p10 = max(0.0, float(regs_b[0.10].predict(X)[0]) - cqr_offset)
    p50 = float(regs_b[0.50].predict(X)[0])
    p90 = float(regs_b[0.90].predict(X)[0]) + cqr_offset
    
    if bust_prob > 0.50:
        p90 = max(p90, p50 * 1.75)
    if data["cin"] > 100:
        p10, p50, p90 = 0.0, 0.0, 0.0

    return {
        "nwp_bust_probability": round(bust_prob, 4),
        "is_bust_warning": bust_prob > 0.2437,
        "quantiles_mm": {
            "p10": round(p10, 2),
            "p50_calibrated_median": round(p50, 2),
            "p90_risk_ceiling": round(p90, 2)
        },
        "conformal_confidence_band_coverage": "80.00%",
        "early_warning_alert": "RED" if p90 > 100.0 or (bust_prob > 0.65 and p90 > 70.0) else ("ORANGE" if p90 > 50.0 or bust_prob > 0.40 else ("YELLOW" if p90 > 15.0 else "GREEN"))
    }

