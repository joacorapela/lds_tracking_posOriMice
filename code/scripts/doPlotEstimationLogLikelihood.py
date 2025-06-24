import sys
import argparse
import pickle
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinematics_est_results_number", type=int,
                        default=32729044,
                        help="kinematcs estimation results number")
    parser.add_argument("--HO_est_results_number", type=int,
                        default=57348728,
                        help="head orientation estimation results number")
    parser.add_argument("--est_results_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results filename pattern")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/estimationLogLike_1.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    kinematics_est_results_number = args.kinematics_est_results_number
    HO_est_results_number = args.HO_est_results_number
    est_results_filename_pattern = args.est_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern 

    kinematics_est_results_filename = est_results_filename_pattern.format(kinematics_est_results_number)
    with open(kinematics_est_results_filename, "rb") as f:
        kinematics_est_results = pickle.load(f)

    HO_est_results_filename = est_results_filename_pattern.format(HO_est_results_number)
    with open(HO_est_results_filename, "rb") as f:
        HO_est_results = pickle.load(f)

    fig = go.Figure()
    trace = go.Scatter(x=kinematics_est_results["elapsed_time"], y=kinematics_est_results["log_like"], name="kinematics", mode="lines+markers")
    fig.add_trace(trace)
    trace = go.Scatter(x=HO_est_results["elapsed_time"], y=HO_est_results["log_like"], name="head orientation", mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Log Likelihood")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
