# Car Price Prediction — Starter Project

This project implements your *A1: Predicting Car Price* assignment end-to-end:
- Clean & preprocess the dataset exactly per the assignment rules
- Train a regression model with a robust `scikit-learn` Pipeline
- Save and load the trained pipeline
- Serve predictions via a Dash web app
- Optional Docker deployment

## Quickstart

### 0) Prereqs
- Python 3.10 or newer
- VS Code with the **Python** and **Jupyter** extensions

### 1) Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Jupyter kernel (optional but recommended)
```bash
python -m ipykernel install --user --name car-price-venv --display-name "Python (car-price)"
```

### 4) Put your dataset
Copy your dataset file (e.g., `Cars (1).csv`) into the project root (same folder as this README).

### 5) Train the model
```bash
python train_model.py --csv "Cars (1).csv" --out app/model.pkl
```
This will: 
- Clean the data as per assignment
- Train a Ridge regression with cross-validation
- Save the full preprocessing+model **Pipeline** to `app/model.pkl`

### 6) Run the app
```bash
python app/app.py
```
Open http://127.0.0.1:8050 in your browser.

### 7) (Optional) Docker
Build & run with Docker:
```bash
docker build -t car-price-app -f app/Dockerfile .
docker run -p 8050:8050 car-price-app
```
Or with Docker Compose:
```bash
docker compose -f app/docker-compose.yaml up --build
```

---

## Repo layout
```
car-price-project/
├── A1_Car_Price_Prediction.ipynb   # Starter notebook (EDA + modeling)
├── train_model.py                  # Script to train and save the pipeline
├── requirements.txt
├── README.md
└── app/
    ├── app.py                      # Dash web app (loads app/model.pkl)
    ├── model.pkl                   # (created after training)
    ├── Dockerfile                  # For containerized serving
    └── docker-compose.yaml
```

---

## Notes
- The pipeline uses `OneHotEncoder(handle_unknown="ignore")` so new categories (e.g., unseen brand) won't crash the app.
- The target `selling_price` is log-transformed during training; predictions are exponentiated back in the app.
- If a user leaves fields blank in the app, the pipeline imputes sensible defaults.
