#WeatherFuse: Hybrid AI–NWP Multi-Model Forecast Blending & Conformal Bust Detection Engine

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20CPU-EE4C2C.svg)
![SIH Problem Statement](https://img.shields.io/badge/SIH%202026-PS%2026081-orange.svg)
![Operational Benchmark](https://img.shields.io/badge/Empirical%20Coverage-86.75%25-brightgreen.svg)

WeatherFuse is an operational, physics-informed two-stage meta-forecasting platform designed to improve the reliability of localized weather predictions during high-impact events.
The system dynamically fuses outputs from conventional Numerical Weather Prediction (NWP) models — GFS, ECMWF, NCUM, and WRF — incorporates atmospheric and terrain information, detects potential forecast busts, applies thermodynamic physical gating, and produces calibrated probabilistic weather forecasts for operational decision-making.
---
📌 Architecture Overview

Traditional multi-model ensembles often rely on static averaging, which can dilute localized extreme-weather signals. WeatherFuse addresses this using a Two-Stage Cascaded Machine Learning Core.
```text
[NWP Feeds: GFS, ECMWF, NCUM, WRF]
                    +
[Terrain DEM & Soundings: CAPE, CIN, Shear]
                    │
                    ▼
        [Data Harmonization & Alignment]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[Stage 1: Bust Detector]   [Stage 2: Quantile Fusion]
[XGBoost / LightGBM +      [Deep QRNN + Multi-Head
 CIN Physical Gating]       Attention + Pinball Loss]
        │                       │
        └───────────┬───────────┘
                    ▼
     [Calibrated Weather Intelligence]
       (p10 / p50 / p90 Forecasts)
                    │
                    ▼
          [FastAPI Inference API]
                    │
                    ▼
       [Role-Specific Dashboards]
```
---
🧠 Core Intelligence Pipeline

1. Stage 1 — Forecast Bust Detection & Thermodynamic Gating
Algorithm: Regularized LightGBM / XGBoost classifier.
Function: Detects inter-model spread and atmospheric instability to estimate potential NWP forecast failure risk (`P_bust`).
Key signals include:
Multi-model precipitation divergence
CAPE
CIN
Relative humidity
Wind shear
Radar reflectivity
Satellite cloud-top temperature
Physical Gating: Thermodynamic conditions such as strong capping inversion can suppress precipitation alerts when the atmospheric state does not support convective development.
> The physical gate is designed to reduce false alarms under dry/capped atmospheric conditions.
---
2. Stage 2 — Deep Attention-Gated Quantile Regression
WeatherFuse generates probabilistic forecasts rather than relying only on a single deterministic value.
Architecture
Deep Neural Network
Multi-Head Self-Attention
Terrain-aware feature processing
Quantile regression
Asymmetric Pinball Loss
Conformal calibration
Forecast Quantiles
Quantile	Interpretation
p10	Lower / conservative forecast bound
p50	Median calibrated forecast
p90	Upper / tail-risk forecast bound
The resulting interval helps operational users understand both expected conditions and potential high-impact scenarios.
---
📊 Operational Benchmarks
The following benchmark figures are reported for the project's validation setup.
Metric	Benchmark Score	Significance
Blended Rainfall MAE	1.69 mm	Compared with the reported deterministic GFS baseline of 17.81 mm
Severe Bust Detection (POD / Recall)	88.32%	121 of 137 reported extreme convective bust events identified
Conformal Coverage (p10–p90)	86.75%	Empirical interval coverage in the reported benchmark
False Alarm Rate under Inversion	0.00%	Under the reported dry-lid / inversion test conditions
Inference Turnaround Latency	<150 ms	CPU-optimized inference target
> **Note:** Benchmark values are project-reported validation results and should be interpreted according to the underlying test dataset, evaluation methodology, and baseline definitions.
---
🗂️ Repository Structure
```text
moes-weather-backend/
│
├── server.py
│   └── FastAPI inference engine and physical guards
│
├── weather_qrnn_weights.pt
│   └── PyTorch Deep QRNN model weights
│
├── scaler_metadata.joblib
│   └── Feature standardization/scaling metadata
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── System documentation
```
---
🛠️ Local Setup & Installation
Prerequisites
Python 3.10 or 3.11
Git
pip
1. Clone the Repository
```bash
git clone https://github.com/Yashvendrasahu/moes-weather-backend.git
cd moes-weather-backend
```
2. Create a Virtual Environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
4. Start the FastAPI Server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
5. Verify the API
Interactive Swagger documentation:
```text
http://localhost:8000/docs
```
Health check:
```text
http://localhost:8000/
```
---
📡 API Contract
`POST /predict`
Production endpoint:
```text
https://moes-weather-backend.onrender.com/predict
```
Content type:
```text
application/json
```
---
Sample Input
```json
{
  "latitude": 31.70,
  "longitude": 76.93,
  "climatic_zone": 1,
  "tp_gfs": 28.0,
  "tp_ecmwf": 36.0,
  "tp_ncum": 48.0,
  "tp_wrf": 58.0,
  "t2m_gfs": 21.0,
  "t2m_ecmwf": 20.5,
  "wind_gfs_kmh": 18.0,
  "wind_ecmwf_kmh": 22.0,
  "cape": 2850.0,
  "cin": 18.0,
  "rh_700": 92.0,
  "mslp": 1004.0,
  "wind_shear": 26.0,
  "elevation_m": 1044.0,
  "terrain_slope_deg": 24.5,
  "radar_max_dbz": 45.0,
  "satellite_ctt_celsius": -62.0
}
```
Sample Response
```json
{
  "status": "success",
  "precipitation": {
    "quantiles_mm": {
      "p10": 26.50,
      "p50": 52.80,
      "p90": 84.20
    },
    "nwp_bust_probability": 0.742,
    "is_bust_warning": true,
    "conformal_coverage": "86.75%",
    "alert": "RED"
  },
  "temperature": {
    "blended_2m_celsius": 20.8,
    "rothfusz_heat_index_celsius": 22.4,
    "heatwave_advisory": "Normal"
  },
  "wind": {
    "sustained_speed_kmh": 20.0,
    "gust_ceiling_p90_kmh": 31.0,
    "gale_warning": false
  }
}
```
---
👥 Multi-Role Operational Dissemination

WeatherFuse acts as a centralized intelligence hub for multiple operational user roles.

1. Duty Meteorologist / Forecaster Desk
Provides:
Inter-model divergence
CAPE/CIN information
Atmospheric sounding insights
Forecast bust probability
Attention-weight interpretation
Human-in-the-loop override capability
2. Disaster Management EOC
Provides:
Actionable weather intelligence
Color-coded alert levels
p90 tail-risk information
High-impact event indicators
Emergency decision-support information
Potential evacuation / preparedness guidance
3. Public Citizen / Farmer Portal
Provides:
Localized weather forecasts
Plain-language advisories
p10–p90 uncertainty intervals
Temperature and thermal-stress indicators
Hazard notifications without technical complexity
4. DevOps & Infrastructure Console
Provides:
Pipeline synchronization status
Forecast-cycle monitoring
Feature completeness checks
API health monitoring
Inference latency telemetry
---
🌦️ Supported Weather Intelligence
WeatherFuse is designed to provide a unified operational view of:
🌧️ Precipitation
🌡️ Temperature
💨 Wind
⛈️ Severe convection risk
🌊 Extreme rainfall / flood risk
🔥 Heat-stress indicators
⚠️ Forecast bust probability
📈 Probabilistic forecast intervals
---
☁️ Cloud Deployment
The backend can be deployed as a lightweight Web Service on Render.
Environment
```text
Python 3.10+
```
Build Command
```bash
pip install -r requirements.txt
```
Start Command
```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```
Deployment Characteristics
CPU-based inference
Lightweight FastAPI service
PyTorch CPU deployment
Stateless API architecture
Suitable for integration with web dashboards
---
🔌 Frontend Integration
The backend exposes a REST API that can be consumed by:
Web dashboards
Disaster-management consoles
Meteorological monitoring interfaces
Public weather portals
Mobile applications
Automated alerting systems
Example JavaScript request:
```javascript
const response = await fetch(
  "https://moes-weather-backend.onrender.com/predict",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  }
);

const data = await response.json();
console.log(data);
```
---
🔬 Technical Highlights
Multi-Model Fusion
Combines forecasts from:
GFS
ECMWF
NCUM
WRF
Physics-Informed Intelligence
Uses atmospheric and environmental variables such as:
CAPE
CIN
Relative Humidity
Wind Shear
MSLP
Terrain Elevation
Terrain Slope
Radar Reflectivity
Satellite Cloud-Top Temperature
Probabilistic Forecasting
Instead of providing only one deterministic forecast, WeatherFuse produces:
```text
p10 ───────── p50 ───────── p90
 │             │             │
Lower       Median         Upper
Bound       Forecast        Risk
```
Conformal Calibration
The system uses conformal calibration to construct empirically evaluated prediction intervals and improve uncertainty quantification.
---
🎯 Why WeatherFuse?
Conventional weather forecasting systems can produce substantially different predictions for the same location and forecast cycle.
WeatherFuse addresses this challenge by combining:
```text
Multi-Model NWP
       +
Observational Signals
       +
Terrain Information
       +
Machine Learning
       +
Atmospheric Physics
       +
Probabilistic Calibration
       ↓
Localized Weather Intelligence
```
This enables the platform to identify model disagreement, estimate forecast uncertainty, and provide operationally useful risk information.
---
🏆 Smart India Hackathon

Developed for:
Smart India Hackathon (SIH) 2026
Problem Statement: 26081
Domain: Weather Forecasting / Multi-Model Forecast Fusion / Disaster Risk Intelligence
Ministry: Ministry of Earth Sciences
Organization: NCMRWF
---
📄 License
This project is released under the MIT License.
---
👨‍💻 Project
WeatherFuse — Hybrid AI–NWP Multi-Model Forecast Blending & Conformal Bust Detection Engine

Youtube demo video :
https://youtu.be/FKY0YRavM88

Repository:
https://github.com/Yashvendrasahu/moes-weather-backend
---
