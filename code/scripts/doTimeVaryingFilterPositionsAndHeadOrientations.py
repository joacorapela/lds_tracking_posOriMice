import sys
import os
import random
import pickle
import math
import argparse
import configparser
import numpy as np
import pandas as pd

import lds.inference


def createB(dt, v, omega, alpha):
    B = np.zeros(shape=(9, 9), dtype=np.double)
    B[0, 0], B[0, 1], B[0, 2] = 1, dt, .5*dt**2
    B[1, 1], B[1, 2] = 1, dt
    B[2, 2] = 1
    B[3, 3], B[3, 4], B[3, 5] = 1, dt, .5*dt**2
    B[4, 4], B[4, 5] = 1, dt
    B[5, 5] = 1
    B[6, 1], B[6, 6], B[6, 7] = alpha*dt, 1-alpha*v*dt, -omega*dt
    B[7, 4], B[7, 6], B[7, 7] = alpha*dt, omega*dt, 1-alpha*v*dt
    B[8, 8] = 1
    return B


def updateB(B, dt, v, omega, alpha):
    B[6, 1], B[6, 6], B[6, 7] = alpha*dt, 1-alpha*v*dt, -omega*dt
    B[7, 4], B[7, 6], B[7, 7] = alpha*dt, omega*dt, 1-alpha*v*dt


def createQ(dt, sigma_a,  varCosTheta, varSineTheta, varOmega):
    Qt = np.array([[dt**4/4, dt**3/2, dt**2/2],
                   [dt**3/2, dt**2, dt],
                   [dt**2/2, dt, 1]], dtype=np.double)
    Q = np.zeros(shape=(9, 9), dtype=np.double)
    Q[:3, :3] = sigma_a**2 * Qt
    Q[3:6, 3:6] = sigma_a**2 * Qt
    Q[6, 6] = varCosTheta
    Q[7, 7] = varCosTheta
    Q[8, 8] = varOmega
    return Q


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_offset_secs", type=int, default=290,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=20,
                        help="duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="sample rate (Hz)")
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000001_filtering.ini",
                        help="filtering parameters filename")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help=("section of ini file containing the filtering "
                              "params"))
    parser.add_argument(
        "--data_filename", type=str,
        default="/nfs/ghome/live/rapela/gatsby-swc/collaborations/amanSaleem/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv",
        help="inputs positions filename")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}")
    args = parser.parse_args()

    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    filtering_params_filename = args.filtering_params_filename
    filtering_params_section = args.filtering_params_section
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
    pos_x0 = float(filtering_params[filtering_params_section]["pos_x0"])
    pos_y0 = float(filtering_params[filtering_params_section]["pos_y0"])
    vel_x0 = float(filtering_params[filtering_params_section]["vel_x0"])
    vel_y0 = float(filtering_params[filtering_params_section]["vel_x0"])
    acc_x0 = float(filtering_params[filtering_params_section]["acc_x0"])
    acc_y0 = float(filtering_params[filtering_params_section]["acc_x0"])
    cos_theta0 = float(filtering_params[filtering_params_section]["cos_theta0"])
    sin_theta0 = float(filtering_params[filtering_params_section]["sin_theta0"])
    omega0 = float(filtering_params[filtering_params_section]["omega0"])
    sigma_a = float(filtering_params[filtering_params_section]["sigma_a"])
    sigma_x = float(filtering_params[filtering_params_section]["sigma_x"])
    sigma_y = float(filtering_params[filtering_params_section]["sigma_y"])
    sigma_cosTheta_state = \
        float(filtering_params[filtering_params_section]["sigma_cosTheta_state"])
    sigma_sinTheta_state = \
        float(filtering_params[filtering_params_section]["sigma_sinTheta_state"])
    sigma_cosTheta_measurement = \
        float(filtering_params[filtering_params_section]["sigma_cosTheta_measurement"])
    sigma_sinTheta_measurement = \
        float(filtering_params[filtering_params_section]["sigma_sinTheta_measurement"])
    diag_V0 = np.array(
        [float(sqrt_diag_v0_value_str)
         for sqrt_diag_v0_value_str in filtering_params["params"]["diag_V0"][1:-1].split(",")]
    )
    sigma_omega = float(filtering_params[filtering_params_section]["sigma_omega"])
    alpha = float(filtering_params[filtering_params_section]["alpha"])

    # set initial conditions
    if math.isnan(pos_x0):
        pos_x0 = measurements[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = measurements[1, 0]
    if math.isnan(cos_theta0):
        cos_theta0 = measurements[2, 0]
    if math.isnan(sin_theta0):
        sin_theta0 = measurements[3, 0]

    m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0,
                   cos_theta0, sin_theta0, omega0], dtype=np.double)
    V0 = np.diag(diag_V0)
    R = np.diag([sigma_x**2, sigma_y**2])

    # build KF matrices
    dt=1.0/sample_rate
    v = np.sqrt(vel_x0**2 + vel_y0**2)
    B = createB(dt=dt, v=v, omega=omega0, alpha=alpha)
    Q = createQ(dt=dt, sigma_a=sigma_a,
                varCosTheta=sigma_cosTheta_state**2,
                varSineTheta=sigma_sinTheta_state**2,
                varOmega=sigma_omega**2)
    Z = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2,
                 sigma_cosTheta_measurement**2,
                 sigma_sinTheta_measurement**2])

    # perform Kalman filtering
    xnn1 = np.empty(shape=(9, 1, measurements.shape[1]), dtype=np.double)
    Vnn1 = np.empty(shape=(9, 9, measurements.shape[1]), dtype=np.double)
    xnn = np.empty(shape=(9, 1, measurements.shape[1]), dtype=np.double)
    Vnn = np.empty(shape=(9, 9, measurements.shape[1]), dtype=np.double)

    kf = lds.inference.TimeVaryingOnlineKalmanFilter()
    x = m0
    P = V0
    for i in range(measurements.shape[1]):
        y = measurements[:, i]
        x, P = kf.predict(x=x, P=P, B=B, Q=Q)
        xnn1[:, 0, i] = x
        Vnn1[:, :, i] = P
        x, P = kf.update(y=y, x=x, P=P, Z=Z, R=R)
        xnn[:, 0, i] = x
        Vnn[:, :, i] = P
        vel_x = xnn[1, 0, i]
        vel_y = xnn[4, 0, i]
        omega = xnn[8, 0, i]
        v = np.sqrt(vel_x**2 + vel_y**2)
        updateB(B=B, dt=dt, v=v, omega=omega, alpha=alpha)

    filter_res = dict(xnn1=xnn1, Vnn1=Vnn1, xnn=xnn, Vnn=Vnn)
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
    print(f"Saved smoothing results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "data_filename": data_filename,
        "start_offset_secs": start_offset_secs,
        "duration_secs": duration_secs,
        "filtering_params_filename": filtering_params_filename,
        "filtering_params_section": filtering_params_section,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    import plotly.graph_objects as go
    fig = go.Figure()
    trace = go.Scatter(x=measurements[0,:], y=measurements[1,:],
                       mode="lines+markers", name="measured")
    fig.add_trace(trace)
    trace = go.Scatter(x=filter_res["xnn1"][0, 0, :], y=filter_res["xnn1"][3, 0, :],
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
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$cos(\theta)$")
    fig.show()

    fig = go.Figure()
    trace = go.Scatter(x=measurements[2, :], y=measurements[3,:],
                       mode="lines+markers", name="measured")
    fig.add_trace(trace)
    trace = go.Scatter(x=filter_res["xnn"][6, 0, :], y=filter_res["xnn"][7, 0, :],
                       mode="lines+markers", name="filtered ")
    fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$cos(\theta)$")
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
