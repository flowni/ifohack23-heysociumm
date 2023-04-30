import pandas as pd
import numpy as np
from constants import *

def clean_rows(df: pd.DataFrame):
    """Removes infinite and NaN entries inplace.

    Args:
        df (pd.DataFrame): The dataframe that needs to be cleaned.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna()

def color_scale(val: float) -> list:
    """Maps a percentage value to a color.

    Args:
        val (float): Percentage value like 80%

    Returns:
        list: RGB values of the color
    """
    for i, b in enumerate(BREAKS):
        if val <= b:
            return COLOR_RANGE[i]
    return COLOR_RANGE[i]


def compute_view(polygons):
        """ Computes the optimal zoom and starting location for a folium map from a list of polygons."""
        bounds = np.array([list(p.bounds) for p in polygons])
        min_lon, min_lat = bounds[:, 0].astype(float).min(), bounds[:, 1].astype(float).min()
        max_lon, max_lat = bounds[:, 2].astype(float).max(), bounds[:, 3].astype(float).max()
        center_lon, center_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
        zoom = 10
        return {
            "min_lat": min_lat,
            "min_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "location": [center_lat, center_lon]
        }