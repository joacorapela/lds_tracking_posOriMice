import sys
import numpy as np
import pickle
import pandas as pd
import argparse
import configparser
import plotly.graph_objects as go
import lds.tracking.plotting

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_result_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=91672310)
    parser.add_argument("--start_offset_secs", type=int, default=1450,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=194,
                        help="duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="sample rate (Hz)")
    parser.add_argument("--color_measured", type=str,
                        default="black",
                        help="color for measured trace")
    parser.add_argument("--cb_alpha", type=float,
                        default=0.3,
                        help="transparency alpha for confidence band")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color pattern for filtered trace")
    parser.add_argument("--filtered_result_filename_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filter result filename pattern")
    parser.add_argument("--fig_filename_pattern_pattern", type=str,
                        default="../../figures/{:08d}_measuredAndEstimatedCosTheta_start{:.02f}_dur{:.02f}.{{:s}}",
                        help="figure filename pattern")

    args = parser.parse_args()

    filtered_result_number = args.filtered_result_number
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    color_measured = args.color_measured
    color_pattern_filtered = args.color_pattern_filtered
    cb_alpha = args.cb_alpha
    filtered_result_filename_pattern = args.filtered_result_filename_pattern
    fig_filename_pattern_pattern = args.fig_filename_pattern_pattern

    filtered_result_filename = \
        filtered_result_filename_pattern.format(filtered_result_number,
                                                "pickle")
    metadata_filename = \
        filtered_result_filename_pattern.format(filtered_result_number, "ini")
    fig_filename_pattern = \
        fig_filename_pattern_pattern.format(filtered_result_number,
                                            start_offset_secs, duration_secs)

    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    data_filename = metadata["params"]["data_filename"]

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
    # measurements = np.vstack((data["implantX"], data["implantZ"], np.cos(thetas), np.sin(thetas))).T
    measurements = np.cos(thetas).T

    with open(filtered_result_filename, "rb") as f:
        filter_res = pickle.load(f)
    times = filter_res["times"]
    samples = np.where(np.logical_and(
        start_offset_secs <= times,
        times <= start_offset_secs + duration_secs))[0]
    times = times[samples]

    # measured = filter_res["measurements"][2, samples].numpy()
    filtered_mean = filter_res["filter_res"]["xnn"][6, 0, samples].numpy()
    filtered_stds = np.sqrt(filter_res["filter_res"]["Pnn"][6, 6, samples]).numpy()
    filtered_ci_upper = filtered_mean + 1.96*filtered_stds
    filtered_ci_lower = filtered_mean - 1.96*filtered_stds

    fig = go.Figure()
    trace = go.Scatter(
        x=times, y=measurements,
        mode="lines+markers",
        marker={"color": color_measured},
        name="measured",
        showlegend=True,
    )
    fig.add_trace(trace)

    trace = go.Scatter(
        x=times, y=filtered_mean,
        mode="lines+markers",
        marker={"color": color_pattern_filtered.format(1.0)},
        name="filtered",
        showlegend=True,
        legendgroup="filtered",
    )
    fig.add_trace(trace)

    trace = go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([filtered_ci_upper,
                          filtered_ci_lower[::-1]]),
        fill="toself",
        fillcolor=color_pattern_filtered.format(cb_alpha),
        line=dict(color=color_pattern_filtered.format(0.0)),
        showlegend=False,
        legendgroup="filtered",
    )
    fig.add_trace(trace)

    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$\cos(\theta)$")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
