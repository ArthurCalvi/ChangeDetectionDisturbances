
import os
from tqdm import tqdm
import rasterio 
import numpy as np 

def load_folder(folder, func=None, func_args=None):
    files = os.listdir(folder)
    files = [f for f in files if (f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png'))]
    files.sort()
    data = []
    for file in tqdm(files):
        with rasterio.open(os.path.join(folder, file)) as src:
            mask = src.read().squeeze()
            if func is not None:
                mask = func(mask, **func_args)
            data.append(mask)
    return np.array(data)


def normalize(array, gain=2):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return gain * ((array - array_min)/(array_max - array_min))


def datetime_to_ordinal(dates):
    """Convert datetime objects to days since the first date in the list."""
    base_date = dates[0]
    return np.array([(date - base_date).days for date in dates]) 


def calculate_slope_with_dates(tree_cover_timeseries, dates, n, K):
    """
    Calculate the slope of linear regression over tree cover data using actual dates for each time step.

    Parameters:
    - tree_cover_timeseries: 3D numpy array [time, height, width]
    - dates: list of datetime objects corresponding to each time step
    - n: int, central time index
    - K: int, window size around the central index

    Returns:
    - slope_map: 2D numpy array [height, width] representing the slope of the tree cover trend
    """
    # Define the time window
    start_index = max(n - K, 0)
    end_index = min(n + K + 1, tree_cover_timeseries.shape[0])

    # Convert dates to ordinal days
    date_nums = datetime_to_ordinal(dates[start_index:end_index])

    # Create the time indices matrix
    t = date_nums.reshape(-1, 1)  # Column vector
    X = np.hstack([t, np.ones((t.shape[0], 1))])  # Add a column of ones for the intercept

    # Reshape the data so each pixel's time series is a column
    Y = tree_cover_timeseries[start_index:end_index].reshape(t.shape[0], -1)

    # Perform matrix multiplication for the linear regression coefficients
    XT_X = X.T @ X
    XT_X_inv = np.linalg.inv(XT_X)
    XT_Y = X.T @ Y
    beta = XT_X_inv @ XT_Y

    # Extract the slope (first row of beta)
    slopes = beta[0, :].reshape(tree_cover_timeseries.shape[1], tree_cover_timeseries.shape[2])

    return slopes # %/day

def calculate_tree_cover_change(tree_cover_previous, tree_cover_current, date_previous, date_current):
    """
    Calculate the change in tree cover between two time steps, normalized by the number of days between measurements.

    Parameters:
    - tree_cover_previous: float, tree cover at the previous time step
    - tree_cover_current: float, tree cover at the current time step
    - date_previous: datetime, date of the previous measurement
    - date_current: datetime, date of the current measurement

    Returns:
    - change_per_day: float, the daily change rate in tree cover
    """
    # Calculate the raw change in tree cover
    change = tree_cover_current - tree_cover_previous
    
    # Calculate the number of days between the two measurements
    num_days = (date_current - date_previous).days
    if num_days == 0:
        raise ValueError("The two dates should not be the same to avoid division by zero.")

    # Normalize the change by the number of days to get a daily change rate
    change_per_day = change / num_days

    return change_per_day