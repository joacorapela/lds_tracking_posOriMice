import sys
import os
import random
import pickle
import math
import argparse
import configparser
import numpy as np
import pandas as pd
import torch

import lds.inference
import lds.tracking.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_offset_secs", type=int, default=1450,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=194,
                        help="duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="sample rate (Hz)")
    parser.add_argument("--initial_state_section", type=str,
                        default="initial_state",
                        help=("section of ini file containing the initial state "
                              "params"))
    parser.add_argument("--state_cov_section", type=str,
                        default="state_cov",
                        help=("section of ini file containing the state cov "
                              "params"))
    parser.add_argument("--measurements_cov_section", type=str,
                        default="measurements_cov",
                        help=("section of ini file containing the measurement cov "
                              "params"))
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000002_filtering.ini",
                        help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument("--data_filename", type=str,
                        default="~/gatsby-swc/collaborations/aman/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv",
                        help="inputs positions filename")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}")
    args = parser.parse_args()

    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    initial_state_section = args.initial_state_section
    state_cov_section = args.state_cov_section
    measurements_cov_section = args.measurements_cov_section
    filtering_params_filename = args.filtering_params_filename
    data_filename = args.data_filename
    results_filename_pattern = args.results_filename_pattern

    # read data
    data = pd.read_csv(data_filename)
    start_sample = int(start_offset_secs * sample_rate)

    if duration_secs < 0:
        number_samples = data.shape[0] - start_sample
    else:
        number_samples = int(duration_secs * sample_rate)

    times = np.arange(start_sample,
                      start_sample+number_samples)/sample_rate
    data = data.iloc[
        start_sample:(start_sample+number_samples), :]
    thetas = np.arctan2(data["noseZ"]-data["implantZ"],
                        data["noseX"]-data["implantX"])
    measurements = np.vstack((data["implantX"], data["implantZ"], np.cos(thetas), np.sin(thetas)))

    # make sure that the first data point is not NaN
    first_not_nan_index = np.where(np.logical_and(
        np.logical_not(np.isnan(data["noseX"])),
        np.logical_not(np.isnan(data["noseZ"]))))[0][0]
    times = times[first_not_nan_index:]
    measurements = measurements[:, first_not_nan_index:]
    #

    # read filtering parameters
    filtering_params = configparser.ConfigParser()
    filtering_params.read(filtering_params_filename)

    pos_x_m0 = float(filtering_params[initial_state_section]["pos_x_mean"])
    vel_x_m0 = float(filtering_params[initial_state_section]["vel_x_mean"])
    acc_x_m0 = float(filtering_params[initial_state_section]["acc_x_mean"])
    pos_y_m0 = float(filtering_params[initial_state_section]["pos_y_mean"])
    vel_y_m0 = float(filtering_params[initial_state_section]["vel_y_mean"])
    acc_y_m0 = float(filtering_params[initial_state_section]["acc_y_mean"])
    cos_theta_m0 = float(filtering_params[initial_state_section]["cos_theta_mean"])
    sin_theta_m0 = float(filtering_params[initial_state_section]["sin_theta_mean"])
    omega_m0 = float(filtering_params[initial_state_section]["omega_mean"])

    pos_x_V0_std = float(filtering_params[initial_state_section]["pos_x_std"])
    vel_x_V0_std = float(filtering_params[initial_state_section]["vel_x_std"])
    acc_x_V0_std = float(filtering_params[initial_state_section]["acc_x_std"])
    pos_y_V0_std = float(filtering_params[initial_state_section]["pos_y_std"])
    vel_y_V0_std = float(filtering_params[initial_state_section]["vel_y_std"])
    acc_y_V0_std = float(filtering_params[initial_state_section]["acc_y_std"])
    cos_theta_V0_std = float(filtering_params[initial_state_section]["cos_theta_std"])
    sin_theta_V0_std = float(filtering_params[initial_state_section]["sin_theta_std"])
    omega_V0_std = float(filtering_params[initial_state_section]["omega_std"])

    sigma_a = float(filtering_params[state_cov_section]["sigma_a"])
    cos_theta_Q_std = float(filtering_params[state_cov_section]["cos_theta_std"])
    sin_theta_Q_std = float(filtering_params[state_cov_section]["sin_theta_std"])
    omega_Q_std = float(filtering_params[state_cov_section]["omega_std"])

    pos_x_R_std = float(filtering_params[measurements_cov_section]["pos_x_std"])
    pos_y_R_std = float(filtering_params[measurements_cov_section]["pos_y_std"])
    cos_theta_R_std = float(filtering_params[measurements_cov_section]["cos_theta_std"])
    sin_theta_R_std = float(filtering_params[measurements_cov_section]["sin_theta_std"])

    alpha = float(filtering_params["other"]["alpha"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = measurements[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = measurements[1, 0]
    if math.isnan(cos_theta_m0):
        cos_theta_m0 = measurements[2, 0]
    if math.isnan(sin_theta_m0):
        sin_theta_m0 = measurements[3, 0]

    m0 = np.array([pos_x_m0, vel_x_m0, acc_x_m0, pos_y_m0, vel_y_m0, acc_y_m0,
                   cos_theta_m0, sin_theta_m0, omega_m0], dtype=np.double)
    V0 = np.diag([pos_x_V0_std, vel_x_V0_std, acc_x_V0_std,
                  pos_y_V0_std, vel_y_V0_std, acc_y_V0_std,
                  cos_theta_V0_std, sin_theta_V0_std, omega_V0_std])

    B, Bdot, Z, Zdot, Q, R = lds.tracking.utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt=1.0/sample_rate, alpha=alpha, sigma_a=sigma_a,
        cos_theta_Q_std=cos_theta_Q_std,
        sin_theta_Q_std=sin_theta_Q_std,
        omega_Q_std=omega_Q_std,
        pos_x_R_std=pos_x_R_std,
        pos_y_R_std=pos_y_R_std,
        cos_theta_R_std=cos_theta_R_std,
        sin_theta_R_std=sin_theta_R_std)
    m0 = torch.from_numpy(m0)
    V0 = torch.from_numpy(V0)
    measurements = torch.from_numpy(measurements)
    filter_res = lds.inference.filterEKF_withMissingValues_torch(
        y=measurements, B=B, Bdot=Bdot, Q=Q, m0=m0, V0=V0, Z=Z, Zdot=Zdot, R=R)

    results = {"times": times,
               "measurements": measurements,
               "filter_res": filter_res}

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved EK filter results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "data_filename": data_filename,
        "start_offset_secs": start_offset_secs,
        "duration_secs": duration_secs,
        "filtering_params_filename": filtering_params_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    import plotly.graph_objects as go
    fig = go.Figure()
    trace = go.Scatter(x=measurements[0,:], y=measurements[1,:],
                       mode="lines+markers", name="measured")
    fig.add_trace(trace)
    trace = go.Scatter(x=filter_res["xnn"][0, 0, :], y=filter_res["xnn"][3, 0, :],
                       mode="lines+markers", name="filtered")
    fig.add_trace(trace)
    fig.update_xaxes(title="x")
    fig.update_yaxes(title="y")
    fig.show()

    fig = go.Figure()
    trace = go.Scatter(x=times, y=measurements[3,:],
                       mode="lines+markers", name="measured")
    fig.add_trace(trace)
    trace = go.Scatter(x=times, y=filter_res["xnn"][7, 0, :],
                       mode="lines+markers", name="filtered ")
    fig.add_trace(trace)
    fig.update_layout(title=f'LL: {filter_res["logLike"]}')
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$sin(\theta)$")
    fig.show()

    fig = go.Figure()
    trace = go.Scatter(x=measurements[2, :], y=measurements[3,:],
                       mode="lines+markers", name="measured")
    fig.add_trace(trace)
    trace = go.Scatter(x=filter_res["xnn"][6, 0, :], y=filter_res["xnn"][7, 0, :],
                       mode="lines+markers", name="filtered ")
    fig.add_trace(trace)
    fig.update_layout(title=f'LL: {filter_res["logLike"]}')
    fig.update_xaxes(title=r"$cos(\theta)$")
    fig.update_yaxes(title=r"$sin(\theta)$")
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
