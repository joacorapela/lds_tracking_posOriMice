import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import functools
import numpy as np
import pandas as pd
import torch

import lds.learning
import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--estMeta_number", help="estimation metadata number",
                        type=int, default=24)
    parser.add_argument("--start_offset_secs", type=int, default=1450,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=194,
                        help="duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="sample rate (Hz)")
    parser.add_argument("--estimate_sigma_a",
                        help="estimate sigma_a",
                        type=bool, default=True)
    parser.add_argument("--estimate_pos_x_R_std",
                        help="estimate pos_x_R_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_pos_y_R_std",
                        help="estimate pos_y_R_std",
                        type=bool, default=True)
    parser.add_argument("--estimate_m0",
                        help="estimate m0",
                        type=bool, default=True)
    parser.add_argument("--estimate_sqrt_diag_V0",
                        help="estimate sqrt_diag_V0",
                        type=bool, default=True)
    parser.add_argument("--data_filename", type=str,
                        default="~/gatsby-swc/collaborations/aman/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv",
                        help="inputs positions filename")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    estimate_sigma_a = args.estimate_sigma_a
    estimate_pos_x_R_std = args.estimate_pos_x_R_std
    estimate_pos_y_R_std = args.estimate_pos_y_R_std
    estimate_m0 = args.estimate_m0
    estimate_sqrt_diag_V0 = args.estimate_sqrt_diag_V0
    data_filename = args.data_filename
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

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

    # make sure that the first data point is not NaN
    is_not_NaN_arrays = [np.logical_not(np.isnan(data["noseX"])),
                         np.logical_not(np.isnan(data["noseZ"])),
                         np.logical_not(np.isnan(data["implantX"])),
                         np.logical_not(np.isnan(data["implantZ"]))]
    not_NaN_all = functools.reduce(np.logical_and, is_not_NaN_arrays)
    first_not_NaN_index = np.where(not_NaN_all)[0][0]
    times = times[first_not_NaN_index:]
    data = data.iloc[first_not_NaN_index:]
    #

    # set outliers to np.NaN
    nose_outliers_indices = utils.get_outliers_indices(data_x=data["noseX"],
                                                       data_y=data["noseZ"])
    impl_outliers_indices = utils.get_outliers_indices(data_x=data["implantX"],
                                                       data_y=data["implantZ"])
    data.iloc[nose_outliers_indices]["noseX"] = np.NaN
    data.iloc[nose_outliers_indices]["noseZ"] = np.NaN
    data.iloc[impl_outliers_indices]["implantX"] = np.NaN
    data.iloc[impl_outliers_indices]["implantZ"] = np.NaN
    #

    # interpolated NaN
    implX_interpolated, implZ_interpolated = utils.interploate_nan(
        data_x=data["implantX"].to_numpy(), data_y=data["implantZ"].to_numpy(),
        times=times)
    #

    y = np.vstack((implX_interpolated, implZ_interpolated))

    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    pos_x_m0 = float(estMeta["initial_state"]["pos_x_mean"])
    pos_y_m0 = float(estMeta["initial_state"]["pos_y_mean"])
    vel_x_m0 = float(estMeta["initial_state"]["vel_x_mean"])
    vel_y_m0 = float(estMeta["initial_state"]["vel_y_mean"])
    acc_x_m0 = float(estMeta["initial_state"]["acc_x_mean"])
    acc_y_m0 = float(estMeta["initial_state"]["acc_y_mean"])
    pos_x_V0_std = float(estMeta["initial_state"]["pos_x_std"])
    pos_y_V0_std = float(estMeta["initial_state"]["pos_y_std"])
    vel_x_V0_std = float(estMeta["initial_state"]["vel_x_std"])
    vel_y_V0_std = float(estMeta["initial_state"]["vel_y_std"])
    acc_x_V0_std = float(estMeta["initial_state"]["acc_x_std"])
    acc_y_V0_std = float(estMeta["initial_state"]["acc_y_std"])

    sigma_a = torch.DoubleTensor([float(estMeta["state_cov"]["sigma_a"])])
    pos_x_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_x_std"])])
    pos_y_R_std = \
        torch.DoubleTensor([float(estMeta["measurements_cov"]["pos_y_std"])])

    max_iter = int(estMeta["optim_params"]["max_iter"])
    lr = float(estMeta["optim_params"]["learning_rate"])
    n_epochs = int(estMeta["optim_params"]["n_epochs"])
    line_search_fn = estMeta["optim_params"]["line_search_fn"]
    tolerance_grad = float(estMeta["optim_params"]["tolerance_grad"])
    tolerance_change = float(estMeta["optim_params"]["tolerance_change"])

    if math.isnan(pos_x_m0):
        pos_x_m0 = y[0, 0]
    if math.isnan(pos_y_m0):
        pos_y_m0 = y[1, 0]

    m0_0 = torch.DoubleTensor([pos_x_m0, vel_x_m0, acc_x_m0,
                               pos_y_m0, vel_y_m0, acc_y_m0])
    sqrt_diag_V0_0 = torch.DoubleTensor([pos_x_V0_std,
                                         vel_x_V0_std,
                                         acc_x_V0_std,
                                         pos_y_V0_std,
                                         vel_y_V0_std,
                                         acc_y_V0_std])

    y = torch.from_numpy(y.astype(np.double))

    vars_to_estimate = {}
    if estimate_sigma_a:
        vars_to_estimate["sigma_a"] = True
    else:
        vars_to_estimate["sigma_a"] = False

    if estimate_pos_x_R_std:
        vars_to_estimate["pos_x_R_std"] = True
    else:
        vars_to_estimate["pos_x_R_std"] = False

    if estimate_pos_y_R_std:
        vars_to_estimate["pos_y_R_std"] = True
    else:
        vars_to_estimate["pos_y_R_std"] = False

    if estimate_m0:
        vars_to_estimate["m0"] = True
    else:
        vars_to_estimate["m0"] = False

    if estimate_sqrt_diag_V0:
        vars_to_estimate["sqrt_diag_V0"] = True
    else:
        vars_to_estimate["sqrt_diag_V0"] = False

    B, Q, Qe, Z, R_0 = lds.tracking.utils.getLDSmatricesForKinematics_torch(dt=1.0/sample_rate,
                                                                            sigma_a=sigma_a,
                                                                            pos_x_R_std=pos_x_R_std,
                                                                            pos_y_R_std=pos_y_R_std)

    optim_res = lds.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
        y=y, B=B, sigma_a0=sigma_a, Qe=Qe, Z=Z,
        pos_x_R_std0=pos_x_R_std, pos_y_R_std0=pos_y_R_std,
        m0_0=m0_0, sqrt_diag_V0_0=sqrt_diag_V0_0,
        max_iter=max_iter, lr=lr, tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change, line_search_fn=line_search_fn,
        n_epochs=n_epochs, vars_to_estimate=vars_to_estimate)

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = \
            estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["estimation_params"] = \
        {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print("Saved results to {:s}".format(estRes_data_filename))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
