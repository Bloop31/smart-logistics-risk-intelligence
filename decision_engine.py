import numpy as np


# ======================================
# TRAFFIC LEVEL RECONSTRUCTION
# ======================================

def get_traffic_level(traffic_heavy, traffic_detour):
    if traffic_heavy == 1:
        return "Heavy"
    elif traffic_detour == 1:
        return "Detour"
    else:
        return "Clear"


# ======================================
# RISK CLASSIFICATION
# ======================================

def classify_risk(delay_probability):

    if delay_probability < 0.40:
        return "Low"
    elif delay_probability <= 0.70:
        return "Medium"
    elif delay_probability <= 0.90:
        return "High"
    else:
        return "Critical"


# ======================================
# RISK → ACTION MAPPING
# ======================================

def get_action(risk, asset_utilization):

    if risk == "Low":
        return "Normal"

    elif risk == "Medium":
        return "Monitor"

    elif risk == "High":
        if asset_utilization > 90:
            return "Reroute_Notify_Redistribute"
        else:
            return "Reroute_Notify"

    elif risk == "Critical":
        return "Reroute_Notify_Redistribute"


# ======================================
# BASELINE ETA CALCULATION
# ======================================

def calculate_baseline_eta(delay_probability, operational_base_time):
    return operational_base_time + (delay_probability * operational_base_time)


# ======================================
# DATA-DRIVEN REROUTE OPTIMIZATION
# ======================================

def calculate_optimized_eta(delay_probability,
                             risk,
                             operational_base_time,
                             clear_factor,
                             improvement_rate=0.5):

    baseline_eta = calculate_baseline_eta(delay_probability,
                                          operational_base_time)

    if risk in ["High", "Critical"]:

        optimized_factor = (
            delay_probability
            - improvement_rate * (delay_probability - clear_factor)
        )

        optimized_eta = (
            operational_base_time
            + (optimized_factor * operational_base_time)
        )

        return optimized_eta

    else:
        return baseline_eta


# ======================================
# CUSTOMER NOTIFICATION
# ======================================

def generate_notification(risk,
                          baseline_eta,
                          optimized_eta,
                          traffic_level):

    baseline = round(baseline_eta, 2)
    optimized = round(optimized_eta, 2)

    if risk in ["High", "Critical"]:

        if optimized < baseline:
            return (
                f"Due to {traffic_level} traffic conditions, your shipment faced elevated delay risk. "
                f"We have optimized the route. Updated ETA: {optimized} minutes "
                f"(earlier estimate was {baseline} minutes)."
            )
        else:
            return (
                f"Due to {traffic_level} traffic conditions, your shipment is under high delay risk. "
                f"Our operations team is actively monitoring the situation. "
                f"Current ETA: {baseline} minutes."
            )

    elif risk == "Medium":
        return (
            f"Your shipment is experiencing moderate traffic conditions. "
            f"Our system is monitoring for potential delays."
        )

    else:
        return "Your shipment is on schedule."
# ======================================
# EXPECTED DELAY HOURS (FOR COST MODEL)
# ======================================

def estimate_delay_hours(delay_probability, operational_base_time):
    """
    Estimate delay hours based on probability.
    We use operational base time as severity scaler.
    """
    return delay_probability * operational_base_time
# ======================================
# COST IMPACT ESTIMATION
# ======================================

def estimate_cost_impact(delay_probability,
                         order_value,
                         shipping_cost,
                         delay_hours,
                         is_express=False):

    # SLA penalty increases if delay is large
    if delay_hours > 60:
        sla_penalty = 0.15 * order_value
    elif delay_hours > 30:
        sla_penalty = 0.08 * order_value
    else:
        sla_penalty = 0

    # Refund risk proportional to probability
    refund_cost = 0.05 * order_value * delay_probability

    # Extra shipping adjustment
    extra_shipping_cost = 0.2 * shipping_cost if delay_probability > 0.5 else 0

    # Express penalty
    express_penalty = 0.05 * order_value if is_express else 0

    total_delay_cost = (
        sla_penalty +
        refund_cost +
        extra_shipping_cost +
        express_penalty
    )

    expected_loss = delay_probability * total_delay_cost

    return round(expected_loss, 2)
# ======================================
# FINANCIAL IMPACT PIPELINE
# ======================================

def calculate_financial_impact(delay_probability,
                               operational_base_time,
                               order_value,
                               shipping_cost,
                               is_express=False):

    delay_hours = estimate_delay_hours(
        delay_probability,
        operational_base_time
    )

    expected_loss = estimate_cost_impact(
        delay_probability=delay_probability,
        order_value=order_value,
        shipping_cost=shipping_cost,
        delay_hours=delay_hours,
        is_express=is_express
    )

    return delay_hours, expected_loss
