"""Traffic flow -> travel time for COS30019 Assignment 2B.

Primary source: lecturer's 'Traffic Flow to Travel Time Conversion v1.0.pdf'.
It specifies a quadratic under-capacity branch, speed capped at 60 km/h,
traffic flow from the *starting* SCATS site, and distance / speed + delay.
The A2B main brief instead mentions volume at the destination (B), so the
site convention must be confirmed with the tutor. This module makes it explicit.

Do not feed raw 15-minute traffic counts into vehicle/hour inputs.
"""

import math

QUADRATIC_A = 1.4648375  # q = -a * v^2 + b * v
QUADRATIC_B = 93.75
SPEED_LIMIT_KMH = 60.0
INTERSECTION_DELAY_SECONDS = 30.0
CAPACITY_VEH_PER_HOUR = 1500.0  # assignment's assumed road capacity


def _nonnegative_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def speed_from_hourly_flow(hourly_volume):
    """Under-capacity (green) root of the PDF quadratic; km/hour.

    Reject flows greater than the stipulated 1500 veh/hour capacity rather
    than silently assign made-up over-capacity speeds. The assignment asks
    us to assume every road is under capacity.
    """
    q = _nonnegative_number("hourly_volume", hourly_volume)
    if q > CAPACITY_VEH_PER_HOUR:
        raise ValueError("Predicted flow exceeds the assignment's 1500 veh/hour assumption")
    d = QUADRATIC_B ** 2 - 4 * QUADRATIC_A * q
    if d < 0:  # round-off tolerance near turning point
        if d < -1e-9:
            raise ValueError("Predicted flow outside quadratic's real-valued domain")
        d = 0.0
    upper_root = (QUADRATIC_B + math.sqrt(d)) / (2 * QUADRATIC_A)
    return min(SPEED_LIMIT_KMH, upper_root)


def edge_travel_time_seconds(distance_km, hourly_volume, intersection_delay_s=INTERSECTION_DELAY_SECONDS):
    """Travel-time weight, in seconds, for one verified directed road segment.

    Assumes one controlled-intersection delay per traversed edge (arrival).
    For the origin==destination route, the routing algorithm returns 0.
    """
    distance = _nonnegative_number("distance_km", distance_km)
    delay = _nonnegative_number("intersection_delay_s", intersection_delay_s)
    speed = speed_from_hourly_flow(hourly_volume)
    return distance / speed * 3600.0 + delay


def hourly_volume_from_four_quarters(quarter_hour_volumes):
    """Sum exactly four consecutive 15-minute vehicle counts (vehicles/hour)."""
    values = list(quarter_hour_volumes)
    if len(values) != 4:
        raise ValueError("Provide exactly four consecutive 15-minute counts")
    return sum(_nonnegative_number("quarter_hour_count", v) for v in values)


def weights_from_road_links(verified_road_links, hourly_predictions_by_site,
                            flow_site='origin', intersection_delay_s=INTERSECTION_DELAY_SECONDS):
    """Return (origin_id, dest_id, edge_seconds) for *verified* road connections.

    verified_road_links: [(origin_scats_id, destination_scats_id, distance_km), ...]
    hourly_predictions_by_site: {scats_site_id: hourly_vehicle_count, ...}
    flow_site: 'origin' (conversion PDF) or 'destination' (main A2B brief)
    This function never guesses whether two intersections are connected.
    """
    if flow_site not in ('origin', 'destination'):
        raise ValueError("flow_site must be 'origin' or 'destination'")
    result = []
    for origin, destination, distance_km in verified_road_links:
        scats_id = origin if flow_site == 'origin' else destination
        if scats_id not in hourly_predictions_by_site:
            raise KeyError(f"Missing hourly traffic prediction for SCATS site {scats_id}")
        result.append((origin, destination, edge_travel_time_seconds(
            distance_km, hourly_predictions_by_site[scats_id], intersection_delay_s)))
    return result
