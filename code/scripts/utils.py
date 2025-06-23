
import numpy as np
import scipy.interpolate


def get_outliers_indices(data_x, data_y, percentile=95):
    step_sizes = np.sqrt(np.diff(data_x)**2 + np.diff(data_y)**2)
    step_sizes_nan_removed = step_sizes[np.where(np.logical_not(np.isnan(step_sizes)))[0]]
    percentile_value = np.percentile(step_sizes_nan_removed, percentile)
    ouliers_indices = np.where(step_sizes>percentile_value)[0] + 1
    return ouliers_indices

def interploate_nan(data_x, data_y, times):
    not_nan_indices_x = set(np.where(np.logical_not(np.isnan(data_x)))[0])
    not_nan_indices_y = set(np.where(np.logical_not(np.isnan(data_y)))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_x.union(not_nan_indices_y)))
    data_x_no_nan = data_x[not_nan_indices]
    data_y_no_nan = data_y[not_nan_indices]
    times_no_nan = times[not_nan_indices]
    tck, u = scipy.interpolate.splprep([data_x_no_nan, data_y_no_nan], s=0,
                                       u=times_no_nan)
    data_x_interpolated, data_y_interpolated = scipy.interpolate.splev(times, tck)
    return data_x_interpolated, data_y_interpolated

