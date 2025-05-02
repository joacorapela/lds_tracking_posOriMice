
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation_ID", type=int, help="simulation ID",
                        default=92754116)
    parser.add_argument("--skip_head_orientations",
                        help="use this options to plot head orientations",
                        action="store_true")
    parser.add_argument("--colorbar_speed",
                        help="use this options to plot speed in the colobar",
                        action="store_true")
    parser.add_argument("--start_time", help="start time to plot", type=int,
                        default=0)
    parser.add_argument("--plot_duration", help="plot duration (sec)", type=int,
                        default=1000)
    parser.add_argument("--arrow_scale_factor", help="scale factor for quiver arrows", type=float,
                        default=0.01)
    parser.add_argument("--plot_type", type=str,
                        default="state",
                        help="plot type: sate | measurement")
    parser.add_argument("--simulation_results_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="results filename pattern")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/{:08d}_simulation_{:s}_pos2DandHO.{:s}")
    args = parser.parse_args()

    simulation_ID = args.simulation_ID
    skip_head_orientations = args.skip_head_orientations
    colorbar_speed = args.colorbar_speed
    start_time = args.start_time
    plot_duration = args.plot_duration
    arrow_scale_factor = args.arrow_scale_factor
    plot_type = args.plot_type
    simulation_results_filename_pattern = args.simulation_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    simulation_results_filename = simulation_results_filename_pattern.format(simulation_ID, "pickle")
    with open(simulation_results_filename, "rb") as f:
        sim_res = pickle.load(f)

    if plot_type == "state":
        x = sim_res["x"]
        n_samples = x.shape[1]
        pos_x = x[0, :]
        pos_y = x[3, :]
        vel_x = x[1, :]
        vel_y = x[4, :]
        cos_theta = x[6, :]
        sin_theta = x[7, :]
    elif plot_type == "measurement":
        x = sim_res["x"]
        y = sim_res["y"]
        n_samples = y.shape[1]
        pos_x = y[0, :]
        pos_y = y[1, :]
        vel_x = x[1, :]
        vel_y = x[4, :]
        cos_theta = y[2, :]
        sin_theta = y[3, :]

    speed = np.sqrt(vel_x**2 + vel_y**2)

    metadata_filename = simulation_results_filename_pattern.format(simulation_ID, "ini")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    dt = float(metadata["params"]["dt"])

    time = np.arange(n_samples) * dt

    if skip_head_orientations:
        fig = go.Figure()
    else:
        fig = ff.create_quiver(x=pos_x, y=pos_y,
                               u=cos_theta*arrow_scale_factor,
                               v=sin_theta*arrow_scale_factor,
                               showlegend=False)
    if colorbar_speed:
        marker_dict=dict(color=speed, colorscale="Viridis",
                         colorbar=dict(title="Speed (pixels/sec)"), showscale=True)
    else:
        marker_dict=dict(color=time, colorscale="Viridis",
                         colorbar=dict(title="Time (sec)"), showscale=True)
    trace = go.Scatter(x=pos_x, y=pos_y,
                       marker=marker_dict,
                       line_color="gray",
                       mode="lines+markers",
                       text=[f"<b>time</b>: {t:.2f} (secs)" for t in time],
                       hovertemplate="<b>x</b>: %{x}<br>" +
                                     "<b>z</b>: %{y}<br>" +
                                     "%{text}",
                       showlegend=False)
    fig.add_trace(trace)
    fig.update_xaxes(title="x")
    fig.update_yaxes(title="y")

    fig.write_image(fig_filename_pattern.format(simulation_ID, plot_type, "png"))
    fig.write_html(fig_filename_pattern.format(simulation_ID, plot_type, "html"))

    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
