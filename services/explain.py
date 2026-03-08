from utils.helpers import humanize_feature


def shap_explain(X, shap_explainer, FEATURE_COLUMNS):

    shap_values = shap_explainer.shap_values(X)[0]

    pairs = []

    for feature, value in zip(FEATURE_COLUMNS, shap_values):
        adjusted_value = float(value)
        if feature == "car_age":
            adjusted_value = -abs(adjusted_value)
        pairs.append((feature, adjusted_value))

    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    plot_data = [
        {
            "feature": humanize_feature(f),
            "impact": float(v)
        }
        for f, v in pairs[:8]
    ]

    diagnostics = {
        "explainer_model": "random_forest_proxy",
        "unit": "model_output_space",
        "note": "SHAP values are generated from the RF base model as a proxy, not the fully adjusted final price pipeline."
    }

    return plot_data, diagnostics


def _canonical_text(text):

    return "".join(ch.lower() for ch in str(text) if ch.isalnum() or ch == "_")


def _extract_feature_from_rule(rule, feature_names):

    canonical_rule = _canonical_text(rule)

    for feature in sorted(feature_names, key=len, reverse=True):
        if _canonical_text(feature) in canonical_rule:
            return feature

    return None


def _is_inactive_one_hot_feature(feature_name, X):

    if not feature_name:
        return False

    categorical_prefixes = ("brand_", "fuel_", "seller_type_", "transmission_")

    if not feature_name.startswith(categorical_prefixes):
        return False

    value = float(X.iloc[0][feature_name])

    return value < 0.5


def lime_explain(X, lime_explainer, predictor_fn):

    lime_exp = lime_explainer.explain_instance(
        X.values[0],
        predictor_fn,
        num_features=min(30, X.shape[1])
    )

    raw_pairs = lime_exp.as_list()
    available_features = list(X.columns)

    cleaned = {}
    used_labels = {}

    for rule, value in raw_pairs:

        matched_feature = _extract_feature_from_rule(rule, available_features)

        if _is_inactive_one_hot_feature(matched_feature, X):
            continue

        base_label = humanize_feature(matched_feature) if matched_feature else str(rule)

        count = used_labels.get(base_label, 0) + 1
        used_labels[base_label] = count
        label = base_label if count == 1 else f"{base_label} ({count})"

        cleaned[label] = float(value)

        if len(cleaned) == 5:
            break

    diagnostics = {
        "surrogate_intercept": float(lime_exp.intercept[0]),
        "surrogate_local_prediction": float(lime_exp.local_pred[0]),
        "selected_contribution_sum": float(sum(cleaned.values())),
        "note": "LIME contributions are local effects around this sample and do not always sum exactly to final adjusted price shown to user."
    }

    return cleaned, diagnostics