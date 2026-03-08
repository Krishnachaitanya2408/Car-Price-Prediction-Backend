import pandas as pd
import difflib


def _canonical_model_name(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _resolve_model_name(raw_model, classes):

    class_lookup = {
        _canonical_model_name(class_name): class_name
        for class_name in classes
    }

    canonical_raw = _canonical_model_name(raw_model)

    exact = class_lookup.get(canonical_raw)
    if exact is not None:
        return exact

    canonical_classes = list(class_lookup.keys())
    if not canonical_classes:
        raise ValueError("Model encoder has no known classes")

    matches = difflib.get_close_matches(
        canonical_raw,
        canonical_classes,
        n=1,
        cutoff=0.0
    )

    if matches:
        return class_lookup[matches[0]]

    return classes[0]


def preprocess_input(data, model_encoder, FEATURE_COLUMNS):

    df = pd.DataFrame([data])

    # Encode model
    raw_model = df.loc[0, "model"]
    classes = list(model_encoder.classes_)

    normalized_model = _resolve_model_name(raw_model, classes)

    df["model"] = model_encoder.transform([normalized_model])

    # One hot encoding
    df = pd.get_dummies(
        df,
        columns=["brand", "fuel", "seller_type", "transmission"],
        drop_first=True
    )

    # Match training columns
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df