import math


def _coerce_numeric(value, field_name, integer=False):

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    if value is None:
        raise ValueError(f"{field_name} is required")

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned == "":
            raise ValueError(f"{field_name} is required")
        parsed = float(cleaned)
    elif isinstance(value, (int, float)):
        parsed = float(value)
    else:
        raise ValueError(f"{field_name} must be numeric")

    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")

    if parsed < 0:
        raise ValueError(f"{field_name} cannot be negative")

    if integer:
        if not float(parsed).is_integer():
            raise ValueError(f"{field_name} must be an integer")
        return int(parsed)

    return float(parsed)


def validate_input(data):

    if not isinstance(data, dict):
        return False, "Invalid JSON body"

    required_fields = [
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "brand",
        "model",
        "car_age"
    ]

    for field in required_fields:
        if field not in data or data[field] is None or (isinstance(data[field], str) and data[field].strip() == ""):
            return False, f"Missing field: {field}"

    numeric_fields = {
        "km_driven": True,
        "mileage": False,
        "engine": False,
        "max_power": False,
        "seats": True,
        "car_age": True
    }

    for field, integer in numeric_fields.items():

        try:
            data[field] = _coerce_numeric(
                data[field],
                field_name=field,
                integer=integer
            )
        except ValueError as err:
            return False, str(err)

    categorical_fields = ["fuel", "seller_type", "transmission", "brand", "model"]
    for field in categorical_fields:
        data[field] = str(data[field]).strip()

    return True, "Valid input"