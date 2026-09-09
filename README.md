# 🏙️ AI Real Estate Analyzer

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=flat&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Machine Learning and **Basic Agentic AI** web application for real estate investment analysis. The system predicts property prices in Bengaluru, provides property recommendations from Kaggle dataset, computes key financial investment metrics (ROI, Rental Yield, Score), classifies investment risk, runs an automated 11-step agentic decision workflow, logs records to SQLite, generates downloadable PDF reports, and visualizes insights via Chart.js with a modern dark glassmorphism UI.

---

## 🎯 Key Features & Capabilities

1. **Data Preprocessing & Outlier Removal**: Handling missing values, parsing sqft ranges, filtering price/sqft standard deviation outliers per location.
2. **Supervised Regression Learning**: Training & comparing **Linear Regression** vs **Random Forest Regressor**.
3. **Model Evaluation Metrics**: MAE, RMSE, and R² Score.
4. **Recommendation System**: Top-N property matching using proximity vector distance and similarity scoring.
5. **Basic Agentic AI**: Autonomous multi-step decision pipeline in pure Python.
6. **Full-Stack Integration**: FastAPI backend, SQLite database CRUD, ReportLab PDF exporter, Chart.js visualizer.

---

## 📊 Dataset (Kaggle Bengaluru House Price Data)

- **Dataset**: Bengaluru House Price Prediction Dataset (`Bengaluru_House_Data.csv`)
- **Source**: [Kaggle Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- **Feature Columns**:
  - `location`: Neighborhood location in Bengaluru
  - `total_sqft`: Built-up area in square feet
  - `bhk`: Bedrooms Count (derived from `size`)
  - `bath`: Number of bathrooms
  - `balcony`: Number of balconies
  - `price`: Target property price in Lakhs (₹)

---

## 🤖 Basic Agentic AI Workflow

When the user clicks **"Analyze Property"**, the AI Agent automatically executes an 11-step pure Python workflow:

```
[1. Read Input] ➔ [2. Search Kaggle Data] ➔ [3. Predict ML Price] ➔ [4. Top 5 Similar Props]
                                                                              │
[8. Classify Risk] ◄─ [7. Investment Score] ◄─ [6. Rental Yield] ◄─ [5. Calculate ROI]
        │
        ▼
[9. AI Insights] ➔ [10. BUY/HOLD/AVOID] ➔ [11. Generate PDF & Save SQLite Record]
```

---

## 📈 Machine Learning Performance Comparison

| Model | MAE (Lakhs) | RMSE (Lakhs) | R² Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | ~15.39 | ~24.35 | ~0.8888 | Baseline Model |
| **Random Forest Regressor** | **~7.27** | **~14.11** | **~0.9626** | **Selected Best Model** |

---

## 📁 Project Structure

```
final/
├── data/
│   └── download_data.py          # Script to fetch/generate Bengaluru_House_Data.csv
├── ml/
│   ├── data_preprocessing.py     # Data cleaning, sqft parser, outlier filtering
│   └── train_model.py            # Model training (LR vs RF), metrics, joblib export
├── saved_models/
│   ├── best_model.joblib         # Saved best ML model pipeline
│   ├── cleaned_dataset.csv       # Preprocessed dataset for recommendation search
│   ├── locations.json            # Supported Bengaluru locations
│   └── metrics.json              # Model evaluation metrics
├── backend/
│   ├── database.py               # SQLite database setup & CRUD helpers
│   ├── agent.py                  # Basic Agentic AI engine & ReportLab PDF generator
│   └── main.py                   # FastAPI app routes & static file server
├── static/
│   ├── css/
│   │   └── style.css             # Glassmorphic Dark Theme (Purple & Cyan accents)
│   ├── js/
│   │   ├── app.js                # App controller, agent progress, history API
│   │   └── charts.js             # Chart.js visualization logic
│   └── index.html                # Single Page App template
├── pdf_reports/                  # Generated PDF analysis reports
├── analysis_history.db           # SQLite Database file
├── requirements.txt              # Python dependencies
├── Procfile                      # Render deployment configuration
└── README.md                     # Documentation Guide
```

---

## 🛠️ Local Setup & Run Instructions

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Machine Learning Models
```bash
python ml/train_model.py
```

### 5. Launch FastAPI Web Application
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Open your browser and visit: `http://localhost:8000`

---

## 🚀 Deployment Instructions

### Deployment to Render
1. Push your repository to **GitHub**.
2. Log into [Render.com](https://render.com/) and create a **New Web Service**.
3. Connect your GitHub repository.
4. Set Build Command: `pip install -r requirements.txt && python ml/train_model.py`
5. Set Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Click **Deploy**.

---

## 📜 License
This project is open-source under the MIT License.
