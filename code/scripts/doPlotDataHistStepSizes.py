
import sys
import argparse
import functools
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)", type=int,
                        default=0)
                        # default=26)
    parser.add_argument("--plot_duration_sec", help="plot duration (sec)", type=int,
                        default=-1)
                        # default=151)
    parser.add_argument("--sample_rate", help="sample rate", type=int,
                        default=30)
    parser.add_argument("--data_filename", help="data filename", type=str,
                        default="~/gatsby-swc/collaborations/aman/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/M24086_20250203_0_tracking_2025-02-06T10_39_59_stepSize.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    plot_duration_sec = args.plot_duration_sec
    sample_rate = args.sample_rate
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    data = pd.read_csv(data_filename)

    start_sample = int(start_time_sec * sample_rate)
    if plot_duration_sec > 0:
        n_samples = int(plot_duration_sec * sample_rate)
    else:
        n_samples = data.shape[0]
        plot_duration_sec = n_samples / sample_rate

    times = np.arange(start_time_sec, start_time_sec+plot_duration_sec, 1.0/sample_rate)
    data = data.iloc[start_sample:(start_sample+n_samples), :]

    step_sizes_nose = np.sqrt(np.diff(data["noseX"])**2 + np.diff(data["noseZ"])**2)
    step_sizes_implant = np.sqrt(np.diff(data["implantX"])**2 + np.diff(data["implantZ"])**2)
    step_sizes_nose_nan_removed = step_sizes_nose[np.where(np.logical_not(np.isnan(step_sizes_nose)))[0]]
    step_sizes_implant_nan_removed = step_sizes_implant[np.where(np.logical_not(np.isnan(step_sizes_implant)))[0]]
    percentile_95_nose = np.percentile(step_sizes_nose_nan_removed, 95)
    percentile_95_implant = np.percentile(step_sizes_implant_nan_removed, 95)
    step_sizes_nose_nan_removed2 = step_sizes_nose_nan_removed[step_sizes_nose_nan_removed<percentile_95_nose]
    step_sizes_implant_nan_removed2 = step_sizes_implant_nan_removed[step_sizes_implant_nan_removed<percentile_95_implant]
    fig = go.Figure()
    trace = go.Histogram(x=step_sizes_nose_nan_removed2, nbinsx=60, name="Nose")
    fig.add_trace(trace)
    trace = go.Histogram(x=step_sizes_implant_nan_removed2, nbinsx=60, name="Implant")
    fig.add_trace(trace)
    fig.update_xaxes(title="Step Size")


    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
