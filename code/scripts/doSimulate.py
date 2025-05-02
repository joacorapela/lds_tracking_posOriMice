import sys
import os
import random
import argparse
import configparser
import pickle
import torch
import lds.simulation
import lds.tracking.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="sampling period")
    parser.add_argument("--V0_std", type=float, default=1e-2,
                        help="std of initial state")
    parser.add_argument("--sigma_a", type=float, default=1e-3,
                        help="acceleration noise std")
    parser.add_argument("--sigma_x", type=float, default=1e-3,
                        help="position x noise std")
    parser.add_argument("--sigma_y", type=float, default=1e-3,
                        help="position y noise std")
    parser.add_argument("--alpha", type=float, default=1e+2,
                        help="weight of HO-vel constraint")
    parser.add_argument("--sigma_cos_theta_state", type=float, default=1e-2,
                        help="std of noise for cos_theta in state")
    parser.add_argument("--sigma_sin_theta_state", type=float, default=1e-2,
                        help="std of noise for sin_theta in state")
    parser.add_argument("--sigma_omega", type=float, default=1e-2,
                        help="std of noise for omega in state")
    parser.add_argument("--sigma_cos_theta_measurement", type=float, default=1e-2,
                        help="std of noise for cos_theta in measurements")
    parser.add_argument("--sigma_sin_theta_measurement", type=float, default=1e-2,
                        help="std of noise for sin_theta in measurements")
    parser.add_argument("--T", type=int, default=1000,
                        help="length of simulation (sec)")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="results filename pattern")
    args = parser.parse_args()

    dt = args.dt
    V0_std = args.V0_std
    sigma_a = args.sigma_a
    sigma_x = args.sigma_x
    sigma_y = args.sigma_y
    alpha = args.alpha
    sigma_cos_theta_state = args.sigma_cos_theta_state
    sigma_sin_theta_state = args.sigma_sin_theta_state
    sigma_omega = args.sigma_omega
    sigma_cos_theta_measurement = args.sigma_cos_theta_measurement
    sigma_sin_theta_measurement = args.sigma_sin_theta_measurement
    T = args.T
    results_filename_pattern = args.results_filename_pattern

    M = 9
    m0 = torch.zeros(size=(M,), dtype=torch.double)
    V0 = torch.diag(torch.ones(size=(M,), dtype=torch.double)*V0_std**2)

    B, Z, Q, R = lds.tracking.utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt=dt, sigma_a=sigma_a, sigma_x=sigma_x, sigma_y=sigma_y, alpha=alpha,
        sigma_cos_theta_state=sigma_cos_theta_state,
        sigma_sin_theta_state=sigma_sin_theta_state,
        sigma_omega=sigma_omega,
        sigma_cos_theta_measurement=sigma_cos_theta_measurement,
        sigma_sin_theta_measurement=sigma_sin_theta_measurement)
    x0, x, y = lds.simulation.simulateNDSgaussianNoise(T=T, B=B, Q=Q, m0=m0,
                                                       V0=V0, Z=Z, R=R)

    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False

    results_filename = results_filename_pattern.format(res_number, "pickle")
    results = {"x0": x0, "x": x, "y": y,}
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved smoothing results to {results_filename}")

    metadata_filename = results_filename_pattern.format(res_number, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "dt": dt,
        "V0_std": V0_std,
        "sigma_a": sigma_a,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "alpha": alpha,
        "sigma_cos_theta_state": sigma_cos_theta_state,
        "sigma_sin_theta_state": sigma_sin_theta_state,
        "sigma_omega": sigma_omega,
        "sigma_cos_theta_measurement": sigma_cos_theta_measurement,
        "sigma_sin_theta_measurement": sigma_sin_theta_measurement,
        "T": T,
        "results_filename_pattern": results_filename_pattern,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    print(f"Done with simulation {res_number}")
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
