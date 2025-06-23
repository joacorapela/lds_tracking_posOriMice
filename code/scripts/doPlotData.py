
import sys
import argparse
import dateutil
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_head_orientations",
                        help="use this options to plot head orientations",
                        action="store_true")
    parser.add_argument("--start_time_sec", help="start time to plot (sec)", type=int,
                        default=1450)
                        # default=26)
    parser.add_argument("--plot_duration_sec", help="plot duration (sec)", type=int,
                        default=194)
                        # default=151)
    parser.add_argument("--sample_rate", help="sample rate", type=int,
                        default=30)
    parser.add_argument("--arrow_scale_factor", help="scale factor for quiver arrows", type=float,
                        default=3.0)
    parser.add_argument("--data_filename", help="data filename", type=str,
                        default="~/gatsby-swc/collaborations/aman/data/posAndHeadOrientationCSV/M24086_20250203_0_tracking_2025-02-06T10_39_59.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/M24086_20250203_0_tracking_2025-02-06T10_39_59.{:s}")
    args = parser.parse_args()

    skip_head_orientations = args.skip_head_orientations
    start_time_sec = args.start_time_sec
    plot_duration_sec = args.plot_duration_sec
    sample_rate = args.sample_rate
    arrow_scale_factor = args.arrow_scale_factor
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    data = pd.read_csv(data_filename)

    start_sample = int(start_time_sec * sample_rate)
    if plot_duration_sec > 0:
        n_samples = int(plot_duration_sec * sample_rate)
    else:
        n_samples = data.shape[0]
        plot_duration_sec = n_samples / sample_rate

    data = data.iloc[start_sample:(start_sample+n_samples), :]

    if n_samples < 0:
        n_samples = data.shape[0]
    times = np.arange(start_time_sec, start_time_sec+plot_duration_sec, 1.0/sample_rate)

    if skip_head_orientations:
        fig = go.Figure()
    else:
        fig = ff.create_quiver(x=data["implantX"], y=data["implantZ"],
                               u=(data["noseX"]-data["implantX"])*arrow_scale_factor,
                               v=(data["noseZ"]-data["implantZ"])*arrow_scale_factor,
                               showlegend=False)
    trace = go.Scatter(x=data["implantX"], y=data["implantZ"],
                       marker=dict(color=times, colorscale="Viridis",
                                   colorbar=dict(title="Time (sec)"), showscale=True),
                       line_color="gray",
                       mode="lines+markers",
                       text=[f"<b>time</b>: {time:.2f} (secs)" for time in times],
                       hovertemplate="<b>x</b>: %{x}<br>" +
                                     "<b>z</b>: %{y}<br>" +
                                     "%{text}",
                       showlegend=False)
    fig.add_trace(trace)
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-.35, y0=-.35, x1=.35, y1=.35,
                  line_color="Gray")
    fig.update_xaxes(title="x")
    fig.update_yaxes(title="z")
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    fig.update_layout(yaxis_scaleanchor="x")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
