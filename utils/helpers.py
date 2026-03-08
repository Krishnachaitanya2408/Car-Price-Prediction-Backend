def humanize_feature(feature):

    if feature.startswith("fuel_"):
        return f"Fuel Type: {feature.split('_')[1]}"

    if feature.startswith("transmission_"):
        return f"Transmission: {feature.split('_')[1]}"

    if feature.startswith("seller_type_"):
        return f"Seller Type: {feature.split('_')[2]}"

    if feature.startswith("brand_"):
        return f"Brand: {feature.split('_')[1]}"

    if feature == "km_driven":
        return "Kilometers Driven"

    if feature == "car_age":
        return "Car Age"

    if feature == "max_power":
        return "Engine Power"

    if feature == "mileage":
        return "Mileage"

    if feature == "engine":
        return "Engine Capacity"

    if feature == "seats":
        return "Number of Seats"

    return feature.replace("_", " ").title()