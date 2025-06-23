
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_offset_secs", type=int, default=1450,
                        help="start offset in seconds")
    parser.add_argument("--duration_secs", type=int, default=194,
                        help="duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="sample rate (Hz)")
    parser.add_argument("forecasting_results_number", type=int,
                        help="number corresponding to filtered results filename")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
    parser.add_argument("--forecasting_results_filename_pattern",
                        help="forecasting results filename pattern",
                        default="../../results/{:08d}_forecasting.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_{:s}_forecasting.{:s}")

    args = parser.parse_args()

    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    forecasting_results_number = args.forecasting_results_number
    variable = args.variable
    forecasting_results_filename_pattern = args.forecasting_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    forecasting_metadata_filename = forecasting_results_filename_pattern.format(
        forecasting_results_number, "ini")
    forecasting_results_filename = forecasting_results_filename_pattern.format(
        forecasting_results_number, "pickle")

    forecasting_metadata = configparser.ConfigParser()
    forecasting_metadata.read(forecasting_metadata_filename)
    h = int(forecasting_metadata["params"]["horizon"])
    filtering_results_number = int(forecasting_metadata["params"]["filtering_results_number"])
    filtering_results_filenames_pattern = forecasting_metadata["params"]["filtering_results_filenames_pattern"]
    data_filename = forecasting_metadata["params"]["data_filename"]
    filtered_metadata_filename = \
        filtering_results_filenames_pattern.format(filtering_results_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    filtering_params_filename = filtered_metadata["params"]["filtering_params_filename"]

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
    measurements = np.vstack((data["implantX"], data["implantZ"], np.cos(thetas), np.sin(thetas))).T

    times = np.arange(start_sample,
                      start_sample+number_samples) / sample_rate

    with open(forecasting_results_filename, "rb") as f:
        forecasting_results = pickle.load(f)
    x_pred = forecasting_results["x"]
    P_pred = forecasting_results["P"]
    log_like = forecasting_results["log_like"]

    N = measurements.shape[0]
    times_pred = np.concatenate((times[(h-1):],
                                 times[-1]+np.arange(1, h+1) / sample_rate))

    if variable == "pos":
        x_forecast_mean = x_pred[0, 0, :]
        x_forecast_std = np.sqrt(P_pred[0, 0, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        y_forecast_mean = x_pred[3, 0, :]
        y_forecast_std = np.sqrt(P_pred[3, 3, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=times, y=measurements[:, 0], name="measurement x",
                           marker_color="black", mode="lines+markers")
        fig.add_trace(trace)

        trace = go.Scatter(x=times, y=measurements[:, 1], name="measurement y",
                           marker_color="black", mode="lines+markers")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=x_forecast_mean, name="forecast x",
                           marker_color="red", legendgroup="forecast_x",
                           mode="lines+markers")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=y_forecast_mean, name="forecast y",
                           marker_color="red", legendgroup="forecast_y",
                           mode="lines+markers")
        fig.add_trace(trace)

        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Position")

    elif variable == "vel":
        x_forecast_mean = x_pred[1, 0, :]
        x_forecast_std = np.sqrt(P_pred[1, 1, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        y_forecast_mean = x_pred[4, 0, :]
        y_forecast_std = np.sqrt(P_pred[4, 4, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=times_pred, y=x_forecast_mean, name="forecast x", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=y_forecast_mean, name="forecast y", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Velocity")

    elif variable == "acc":
        x_forecast_mean = x_pred[2, 0, :]
        x_forecast_std = np.sqrt(P_pred[2, 2, :])
        x_forecast_95ci_down = x_forecast_mean - 1.96 * x_forecast_std
        x_forecast_95ci_up = x_forecast_mean + 1.96 * x_forecast_std

        y_forecast_mean = x_pred[5, 0, :]
        y_forecast_std = np.sqrt(P_pred[5, 5, :])
        y_forecast_95ci_down = y_forecast_mean - 1.96 * y_forecast_std
        y_forecast_95ci_up = y_forecast_mean + 1.96 * y_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=times_pred, y=x_forecast_mean, name="forecast x", marker_color="red", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=y_forecast_mean, name="forecast y", marker_color="red", legendgroup="forecast_y")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((x_forecast_95ci_up,
                                             x_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((y_forecast_95ci_up,
                                             y_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)
        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Acceleration")

    elif variable == "omega":
        forecast_mean = x_pred[8, 0, :]
        forecast_std = np.sqrt(P_pred[8, 8, :])
        forecast_95ci_down = forecast_mean - 1.96 * forecast_std
        forecast_95ci_up = forecast_mean + 1.96 * forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=times_pred, y=forecast_mean, name="forecast", marker_color="red", legendgroup="forecast")
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((forecast_95ci_up,
                                             forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast"
                           )
        fig.add_trace(trace)

        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title=r"\omega")

    elif variable == "HO":
        cosTheta_forecast_mean = x_pred[6, 0, :]
        cosTheta_forecast_std = np.sqrt(P_pred[6, 6, :])
        cosTheta_forecast_95ci_down = cosTheta_forecast_mean - 1.96 * cosTheta_forecast_std
        cosTheta_forecast_95ci_up = cosTheta_forecast_mean + 1.96 * cosTheta_forecast_std

        sinTheta_forecast_mean = x_pred[7, 0, :]
        sinTheta_forecast_std = np.sqrt(P_pred[7, 7, :])
        sinTheta_forecast_95ci_down = sinTheta_forecast_mean - 1.96 * sinTheta_forecast_std
        sinTheta_forecast_95ci_up = sinTheta_forecast_mean + 1.96 * sinTheta_forecast_std

        color_pattern_filtered = "rgba(255,0,0,{:f})"
        cb_alpha = 0.3
        fig = go.Figure()

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((cosTheta_forecast_95ci_up,
                                             cosTheta_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_x"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=np.concatenate((times_pred, times_pred[::-1])),
                           y=np.concatenate((sinTheta_forecast_95ci_up,
                                             sinTheta_forecast_95ci_down[::-1])),
                           fill="toself",
                           fillcolor=color_pattern_filtered.format(cb_alpha),
                           line=dict(color=color_pattern_filtered.format(0.0)),
                           showlegend=False,
                           legendgroup="forecast_y"
                           )
        fig.add_trace(trace)

        trace = go.Scatter(x=times, y=measurements[:, 2],
                           name="measurement cos(theta)", marker_color="black",
                           mode="lines+markers")
        fig.add_trace(trace)

        trace = go.Scatter(x=times, y=measurements[:, 3],
                           name="measurement sin(theta)", marker_color="black",
                           mode="lines+markers")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=cosTheta_forecast_mean,
                           name="forecast cos(theta)", marker_color="red",
                           mode="lines+markers", legendgroup="forecast_x")
        fig.add_trace(trace)

        trace = go.Scatter(x=times_pred, y=sinTheta_forecast_mean,
                           name="forecast sin(theta)", marker_color="red",
                           mode="lines+markers", legendgroup="forecast_y")
        fig.add_trace(trace)

        fig.update_layout(title=f"Forecasting Horizon: {h} samples, Log-Likelihood: {log_like}")
        fig.update_xaxes(title="Time (sec)")
        # fig.update_yaxes(title="Position")

    fig.write_image(fig_filename_pattern.format(forecasting_results_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(forecasting_results_number, variable, "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
