from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from utils.validation import validate_input
from utils.preprocess import preprocess_input
from services.predictor import predict_price, predict_price_batch
from services.explain import shap_explain, lime_explain

app = Flask(__name__)

# ================= LOAD MODELS =================

rf = joblib.load("models/rf.pkl")
xgb = joblib.load("models/xgb.pkl")
lgbm = joblib.load("models/lgbm.pkl")
ridge = joblib.load("models/ridge.pkl")
ann = joblib.load("models/ann.pkl")
meta_model = joblib.load("models/meta.pkl")

scaler = joblib.load("models/scaler.pkl")

model_encoder = joblib.load("models/label_model.pkl")

FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")

# ================= EXPLAINERS =================

shap_explainer = shap.TreeExplainer(rf)

lime_training_data = joblib.load("models/lime_training_data.pkl")

lime_explainer = LimeTabularExplainer(
    training_data=lime_training_data,
    feature_names=FEATURE_COLUMNS,
    mode="regression"
)

# ================= ERROR RANGE =================

MODEL_MAE = 75000
UNCERTAINTY_FACTOR = 0.4
BASE_PRICE_ADJUSTMENT = 50000
AGE_DEPRECIATION_PER_YEAR = 20000
MAX_AGE_DEPRECIATION = 300000


def _calculate_adjusted_price(raw_price, car_age):

    normalized_age = max(0.0, float(car_age))
    age_depreciation = min(
        normalized_age * AGE_DEPRECIATION_PER_YEAR,
        MAX_AGE_DEPRECIATION
    )

    total_deduction = BASE_PRICE_ADJUSTMENT + age_depreciation
    adjusted_price = max(0.0, float(raw_price) - total_deduction)

    return adjusted_price, {
        "raw_price": round(float(raw_price), 2),
        "base_adjustment": round(float(BASE_PRICE_ADJUSTMENT), 2),
        "age_depreciation": round(float(age_depreciation), 2),
        "total_deduction": round(float(total_deduction), 2)
    }


def ensemble_predict_for_lime(x_array):

    X_lime = pd.DataFrame(x_array, columns=FEATURE_COLUMNS)

    preds = predict_price_batch(
        X_lime,
        scaler,
        rf,
        xgb,
        lgbm,
        ridge,
        ann,
        meta_model
    )

    age_values = X_lime["car_age"].clip(lower=0).to_numpy(dtype=float)
    age_depreciation = np.minimum(
        age_values * AGE_DEPRECIATION_PER_YEAR,
        MAX_AGE_DEPRECIATION
    )

    adjusted = np.maximum(0, preds - BASE_PRICE_ADJUSTMENT - age_depreciation)

    return adjusted


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # ---------- VALIDATION ----------
    valid, message = validate_input(data)

    if not valid:
        return jsonify({"error": message}), 400

    # ---------- PREPROCESS ----------
    try:
        X = preprocess_input(
            data,
            model_encoder,
            FEATURE_COLUMNS
        )
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    # ---------- PREDICT ----------
    price = predict_price(
        X,
        scaler,
        rf, xgb, lgbm, ridge, ann,
        meta_model
    )

    adjusted_price, adjustments = _calculate_adjusted_price(
        raw_price=price,
        car_age=data.get("car_age", 0)
    )

    # ---------- RANGE ----------
    uncertainty = MODEL_MAE * UNCERTAINTY_FACTOR

    lower = max(0, adjusted_price - uncertainty)
    upper = adjusted_price + uncertainty

    # ---------- EXPLANATIONS ----------
    shap_data, shap_diagnostics = shap_explain(X, shap_explainer, FEATURE_COLUMNS)

    lime_data, lime_diagnostics = lime_explain(X, lime_explainer, ensemble_predict_for_lime)

    age_depreciation = float(adjustments.get("age_depreciation", 0))
    if age_depreciation > 0:
        if "Car Age" in lime_data:
            del lime_data["Car Age"]
        lime_data["Age Depreciation Rule"] = -age_depreciation

    lime_diagnostics["selected_contribution_sum"] = float(sum(lime_data.values()))
    lime_diagnostics["final_adjusted_price"] = round(float(adjusted_price), 2)
    lime_diagnostics["age_depreciation_rule"] = round(float(age_depreciation), 2)

    # ---------- RESPONSE ----------
    return jsonify({
        "predicted_price": round(float(adjusted_price), 2),

        "price_range": {
            "min": round(lower, 2),
            "max": round(upper, 2)
        },

        "shap_plot_data": shap_data,
        "shap_diagnostics": shap_diagnostics,
        "lime_explanation": lime_data,
        "lime_diagnostics": lime_diagnostics,
        "adjustments": adjustments
    })


@app.route("/")
def home():
    return "🚗 Car Price Prediction API Running"


if __name__ == "__main__":
    app.run(debug=True)