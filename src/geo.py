"""Arena locations for travel and time-zone features.

Coordinates are approximate arena locations; time zones are standard-time
UTC offsets (DST shifts every team equally, so differences are unaffected).
Franchise moves inside the 2007-2020 window are handled by start year.
"""

from __future__ import annotations

import math

# team -> list of (first season start year, lat, lon, utc_offset)
_ARENAS: dict[str, list[tuple[int, float, float, int]]] = {
    "ATL": [(0, 33.757, -84.396, -5)],
    "BOS": [(0, 42.366, -71.062, -5)],
    "BKN": [(0, 40.812, -74.071, -5), (2010, 40.734, -74.171, -5), (2012, 40.683, -73.976, -5)],
    "CHA": [(0, 35.225, -80.839, -5)],
    "CHI": [(0, 41.881, -87.674, -6)],
    "CLE": [(0, 41.496, -81.688, -5)],
    "DAL": [(0, 32.790, -96.810, -6)],
    "DEN": [(0, 39.749, -105.008, -7)],
    "DET": [(0, 42.697, -83.245, -5), (2017, 42.341, -83.055, -5)],
    "GSW": [(0, 37.750, -122.203, -8), (2019, 37.768, -122.388, -8)],
    "HOU": [(0, 29.751, -95.362, -6)],
    "IND": [(0, 39.764, -86.155, -5)],
    "LAC": [(0, 34.043, -118.267, -8)],
    "LAL": [(0, 34.043, -118.267, -8)],
    "MEM": [(0, 35.138, -90.051, -6)],
    "MIA": [(0, 25.781, -80.188, -5)],
    "MIL": [(0, 43.044, -87.917, -6)],
    "MIN": [(0, 44.980, -93.276, -6)],
    "NOP": [(0, 29.949, -90.082, -6)],
    "NYK": [(0, 40.751, -73.993, -5)],
    "OKC": [(0, 47.622, -122.354, -8), (2008, 35.463, -97.515, -6)],  # Seattle until 2008
    "ORL": [(0, 28.539, -81.384, -5)],
    "PHI": [(0, 39.901, -75.172, -5)],
    "PHX": [(0, 33.446, -112.071, -7)],
    "POR": [(0, 45.532, -122.667, -8)],
    "SAC": [(0, 38.649, -121.518, -8), (2016, 38.580, -121.500, -8)],
    "SAS": [(0, 29.427, -98.438, -6)],
    "TOR": [(0, 43.643, -79.379, -5)],
    "UTA": [(0, 40.768, -111.901, -7)],
    "WAS": [(0, 38.898, -77.021, -5)],
}

HIGH_ALTITUDE = {"DEN", "UTA"}


def arena(team: str, season_start_year: int) -> tuple[float, float, int]:
    """(lat, lon, utc_offset) of a team's home arena in a given season."""
    if team not in _ARENAS:
        raise KeyError(f"no arena for team {team!r}")
    loc = None
    for first_year, lat, lon, tz in _ARENAS[team]:
        if season_start_year >= first_year:
            loc = (lat, lon, tz)
    return loc


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
