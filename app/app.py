# app/app.py — Dash + Bootstrap car price predictor (assignment-aligned)
# ---------------------------------------------------------------
# Notes:
# - Exposes only assignment-required features
# - Filters fuel to Petrol/Diesel, owner to 1..4 (no Test Drive Cars)
# - Converts model output from log(price) -> price with np.exp
# - Brand normalized to the first word
# - Clean, commented, beginner-friendly

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# -----------------------------
# Paths & model/data loading
# -----------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"
CSV_PATH   = HERE / "Cars.csv"

# Your model was trained on log(selling_price)
MODEL_OUTPUT_IS_LOG = True  # <- keep True for your assignment training

# Load trained pipeline (preprocessing + regressor)
with open(MODEL_PATH, "rb") as f:
    pipe = pickle.load(f)

# Load dataset to build dropdown options and sensible defaults
df = pd.read_csv(CSV_PATH)

# Ensure 'brand' exists: first word of 'name' if needed
if "brand" not in df.columns:
    if "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.strip().str.split().str[0]
    else:
        df["brand"] = "Unknown"

# Clean helper to extract numeric portion if units exist
def _clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")

for col in ["mileage", "engine", "max_power"]:
    if col in df.columns:
        df[col] = _clean_numeric(df[col])

# -----------------------------
# Build UI options (per assignment)
# -----------------------------
# Fuel must be only Petrol/Diesel
fuel_options = ["Petrol", "Diesel"]

# If CSV has others (CNG/LPG), exclude them
if "fuel" in df.columns:
    found = sorted(df["fuel"].dropna().astype(str).unique().tolist())
    # intersect with allowed list
    fuel_options = [f for f in found if f in {"Petrol", "Diesel"}] or ["Petrol", "Diesel"]

# Transmission options from data or default
trans_options = (
    sorted(df["transmission"].dropna().astype(str).unique().tolist())
    if "transmission" in df.columns else ["Manual", "Automatic"]
)
if not trans_options:
    trans_options = ["Manual", "Automatic"]

# Owner mapping is numeric 1..4 (exclude 5 = Test Drive Car)
owner_options = [1, 2, 3, 4]

# Years from data (or a reasonable range)
year_options = sorted(
    pd.to_numeric(df.get("year", pd.Series([], dtype=float)), errors="coerce")
    .dropna().astype(int).unique().tolist()
) or list(range(2005, 2025))

# Brands from data
brand_options = sorted(df["brand"].dropna().astype(str).unique().tolist()) or ["Maruti", "Hyundai", "Honda"]

# -----------------------------
# Defaults (median/mode after cleaning)
# -----------------------------
def _median(col, fallback):
    if col not in df.columns: return fallback
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.median()) if not s.empty else fallback

def _mode(col, fallback):
    if col not in df.columns: return fallback
    s = df[col].dropna().astype(str)
    return s.mode().iloc[0] if not s.empty and not s.mode().empty else fallback

defaults = {
    "year": int(_median("year", 2018)),
    "km_driven": int(_median("km_driven", 45000)),
    "mileage": _median("mileage", 18.5),
    "engine": _median("engine", 1197.0),
    "max_power": _median("max_power", 82.0),
    "fuel": fuel_options[0],
    "brand": _mode("brand", brand_options[0]),
    "transmission": _mode("transmission", trans_options[0]),
    "owner": 1,
}

# -----------------------------
# Dash app (Bootstrap)
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def dd(id_, label, options, value):
    return dbc.Col([
        dbc.Label(label),
        dcc.Dropdown(id=id_, options=[{"label": str(o), "value": o} for o in options], value=value, clearable=False)
    ], md=4)

def num(id_, label, value, min_=0, step=None):
    return dbc.Col([
        dbc.Label(label),
        dbc.Input(id=id_, type="number", value=value, min=min_, step=step)
    ], md=4)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2(" Car Price Prediction "), width=12), className="mt-3"),
    dbc.Row(dbc.Col(html.P("Enter details below"), width=12)),
    dbc.Card(dbc.CardBody([
        # Row 1
        dbc.Row([
            dd("brand", "Brand (first word)", brand_options, defaults["brand"]),
            dd("year", "Year", year_options, defaults["year"]),
            dd("fuel", "Fuel (Petrol/Diesel only)", fuel_options, defaults["fuel"]),
        ], className="gy-3"),
        # Row 2
        dbc.Row([
            dd("transmission", "Transmission", trans_options, defaults["transmission"]),
            dd("owner", "Owner (1=First ... 4=Fourth+)", owner_options, defaults["owner"]),
            num("km_driven", "Kilometers Driven", defaults["km_driven"], min_=0, step=500),
        ], className="gy-3"),
        # Row 3
        dbc.Row([
            num("engine", "Engine (CC)", defaults["engine"], min_=0, step=10),
            num("mileage", "Mileage (km/l)", defaults["mileage"], min_=0, step=0.1),
            num("max_power", "Max Power (bhp)", defaults["max_power"], min_=0, step=0.1),
        ], className="gy-3"),

        dbc.Button("Predict Price", id="predict", color="primary", className="mt-3"),
        html.Hr(),
        html.H4("Predicted Selling Price:", className="mt-2"),
        html.Div(id="out", className="display-6 text-success"),
    ]), className="shadow p-3 mt-3"),
], fluid=True)

# -----------------------------
# Callback: predict (8 features)
# -----------------------------
@app.callback(
    Output("out", "children"),
    Input("predict", "n_clicks"),
    State("year", "value"),
    State("km_driven", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("fuel", "value"),
    State("brand", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    prevent_initial_call=True
)
def predict_price(n, year, km_driven, mileage, engine, max_power, fuel, brand, transmission, owner):
    # Normalize brand to first word (defensive)
    brand_norm = None
    if isinstance(brand, str) and brand.strip():
        brand_norm = brand.split()[0]
    else:
        brand_norm = defaults["brand"]

    # Build feature row matching training schema
    row = {
        "year": year or defaults["year"],
        "km_driven": km_driven if (km_driven is not None and km_driven >= 0) else defaults["km_driven"],
        "mileage": mileage if (mileage is not None and mileage > 0) else defaults["mileage"],
        "engine": engine if (engine is not None and engine > 0) else defaults["engine"],
        "max_power": max_power if (max_power is not None and max_power > 0) else defaults["max_power"],
        "fuel": fuel if fuel in {"Petrol", "Diesel"} else defaults["fuel"],  # enforce assignment rule
        "brand": brand_norm,
        "transmission": transmission or defaults["transmission"],
        "owner": owner if owner in {1, 2, 3, 4} else 1,  # exclude 5 (Test Drive Car)
    }

    X = pd.DataFrame([row])

    # Predict (your pipeline handles preprocessing). Model output is log(price).
    y_pred = pipe.predict(X)
    val = float(np.asarray(y_pred).ravel()[0])

    price = np.exp(val) if MODEL_OUTPUT_IS_LOG else val
    return f"₹ {price:,.2f}"

# -----------------------------
# Run (Dash v3+)
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
