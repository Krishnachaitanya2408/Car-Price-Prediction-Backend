import numpy as np


def predict_price_batch(X, scaler, rf, xgb, lgbm, ridge, ann, meta_model):

    X_scaled = scaler.transform(X)

    pred_rf = rf.predict(X)
    pred_xgb = xgb.predict(X)
    pred_lgbm = lgbm.predict(X)
    pred_ridge = ridge.predict(X_scaled)
    pred_ann = ann.predict(X_scaled)

    meta_input = np.column_stack([
        pred_rf,
        pred_xgb,
        pred_lgbm,
        pred_ridge,
        pred_ann
    ])

    final_log = meta_model.predict(meta_input)

    return np.expm1(final_log)


def predict_price(X, scaler, rf, xgb, lgbm, ridge, ann, meta_model):

    return predict_price_batch(
        X,
        scaler,
        rf,
        xgb,
        lgbm,
        ridge,
        ann,
        meta_model
    )[0]