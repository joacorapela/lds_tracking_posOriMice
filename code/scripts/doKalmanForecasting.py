import sys
import os
import pickle
import math
import random
import numpy as np
import pandas as pd
import argparse
import configparser
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
    parser.add_argument("filtering_results_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--h", type=int,
                        default=2,
                        help="forecast horizon in samples")
    parser.add_argument("--filtering_results_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtering_results filename pattern")
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
    parser.add_argument("--data_filename", type=str,
                        default="~/gatsby-swc/collaborations/aman/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv",
                        help="inputs positions filename")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        default="../../results/{:08d}_forecasting.{:s}")

    args = parser.parse_args()

    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    filtering_results_number = args.filtering_results_number
    h = args.h
    initial_state_section = args.initial_state_section
    state_cov_section = args.state_cov_section
    measurements_cov_section = args.measurements_cov_section
    filtering_results_filenames_pattern = \
        args.filtering_results_filenames_pattern
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
        start_sample:(start_sample+number_samples), :][["implantX", "implantZ"]]
    data = data.to_numpy()
    if duration_secs < 0:
        number_samples = data.shape[0] - start_sample
    else:
        number_samples = int(duration_secs * sample_rate)

    times = np.arange(start_sample,
                      start_sample+number_samples) / sample_rate

    filtering_results_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "pickle")
    with open(filtering_results_filename, "rb") as f:
        filtering_results = pickle.load(f)

    xnn = filtering_results["filter_res"]["xnn"]
    Pnn = filtering_results["filter_res"]["Pnn"]

    filtered_metadata_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    filtering_params_filename = filtered_metadata["params"]["filtering_params_filename"]

    filtering_params = configparser.ConfigParser()
    filtering_params.read(filtering_params_filename)
    pos_x_m0 = float(filtering_params[initial_state_section]["pos_x_mean"])
    vel_x_m0 = float(filtering_params[initial_state_section]["vel_x_mean"])
    acc_x_m0 = float(filtering_params[initial_state_section]["acc_x_mean"])
    pos_y_m0 = float(filtering_params[initial_state_section]["pos_y_mean"])
    vel_y_m0 = float(filtering_params[initial_state_section]["vel_y_mean"])
    acc_y_m0 = float(filtering_params[initial_state_section]["acc_y_mean"])

    pos_x_V0_std = float(filtering_params[initial_state_section]["pos_x_std"])
    vel_x_V0_std = float(filtering_params[initial_state_section]["vel_x_std"])
    acc_x_V0_std = float(filtering_params[initial_state_section]["acc_x_std"])
    pos_y_V0_std = float(filtering_params[initial_state_section]["pos_y_std"])
    vel_y_V0_std = float(filtering_params[initial_state_section]["vel_y_std"])
    acc_y_V0_std = float(filtering_params[initial_state_section]["acc_y_std"])

    sigma_a = float(filtering_params[state_cov_section]["sigma_a"])
    pos_x_R_std = float(filtering_params[measurements_cov_section]["pos_x_std"])
    pos_y_R_std = float(filtering_params[measurements_cov_section]["pos_y_std"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = data[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = data[0, 1]

    m0 = np.array([pos_x_m0, vel_x_m0, acc_x_m0, pos_y_m0, vel_y_m0, acc_y_m0],
                   dtype=np.double)
    V0 = np.diag([pos_x_V0_std, vel_x_V0_std, acc_x_V0_std,
                  pos_y_V0_std, vel_y_V0_std, acc_y_V0_std])

    B, Q, _, Z, R = lds.tracking.utils.getLDSmatricesForKinematics_torch(
        dt=1.0/sample_rate, sigma_a=sigma_a,
        pos_x_R_std=pos_x_R_std,
        pos_y_R_std=pos_y_R_std)

    x_pred, P_pred = lds.inference.lds_forecast_batch(xnn=xnn, Pnn=Pnn, B=B, Q=Q, m0=m0, V0=V0, h=h)
    log_like = lds.inference.log_like_observations_given_forecasts_lds(
        h=h, y=data.T, x_pred=x_pred, P_pred=P_pred, Z=Z.numpy(), R=R.numpy())
    print(f"log likelihood: {log_like}")

    forecast_res = dict(x=x_pred, P=P_pred, log_like=log_like)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(forecast_res, f)
    print(f"Saved forecasting results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "horizon": h,
        "filtering_results_number": filtering_results_number,
        "filtering_results_filenames_pattern": filtering_results_filenames_pattern,
        "data_filename": data_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
