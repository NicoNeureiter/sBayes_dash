from __future__ import annotations

import os; os.environ['USE_PYGEOS'] = '0'  # Fix for Pandas deprecation warning

from io import StringIO
import base64
from pathlib import Path

# from jupyter_dash import JupyterDash
# from dash import Input, Output, State
from dash import html
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sbayes_dash.app_state import AppState
from sbayes_dash import dash_components as components
from sbayes_dash.load_data import Confounder, Objects
from sbayes_dash.util import read_data_csv, reproject_locations
from sbayes_dash.plotting import plot_summary_map, plot_sample_map, create_results_figure

# data_projection: str = "+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs "
data_projection: str = "EPSG:4326"


def find_biggest_angle_gap(degrees: NDArray[float]) -> float:
    degrees = np.sort(degrees)
    degrees = np.append(degrees, degrees[0] + 360)
    i = np.argmax(np.diff(degrees))
    return (degrees[i + 1] + degrees[i]) / 2


def parse_content(content: str):
    content_type, content_bytestr = content.split(',')
    content_str = str(base64.b64decode(content_bytestr))[2:-1]
    return content_str.replace(r"\t", "\t").replace(r"\n", "\n")


def parse_clusters_samples(clusters_samples: str) -> NDArray[bool]:  # shape: (n_clusters, n_samples, n_sites)
    samples_list = [
        [list(c) for c in line.split('\t')]
        for line in clusters_samples.split("\n")
        if line.strip()
    ]
    return np.array(samples_list, dtype=int).astype(bool).transpose((1, 0, 2))


# Initialized app
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], suppress_callback_exceptions=True)
# app = JupyterDash(__name__, suppress_callback_exceptions=True)
server = app.server
state = AppState()

# Set up the layout
app.layout = components.get_base_layout()


@app.callback(
    #    Output('upload-data', 'disabled'),
    Output('upload-clusters', 'disabled'),
    Output('upload-data', 'children'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
)
def update_data(content, filename):
    if content is None:
        return

    # Load data
    data_str = parse_content(content)
    data_file = StringIO(data_str)
    state.data = data = read_data_csv(data_file)
    state.objects = objects = Objects.from_dataframe(data)
    state.families = families = Confounder.from_dataframe(data, confounder_name="family")
    state.locations = locations = reproject_locations(objects.locations, data_projection, "EPSG:4326")
    cut_longitude = find_biggest_angle_gap(locations[:, 0])
    locations[:, 0] = (locations[:, 0] - cut_longitude) % 360 + cut_longitude

    family_names = np.array(families.group_names + [""])
    family_ids = []
    for i, lang in enumerate(objects.names):
        i_fam = np.flatnonzero(families.group_assignment[:, i])
        i_fam = i_fam[0] if len(i_fam) > 0 else families.n_groups
        family_ids.append(i_fam)
    family_ids = np.array(family_ids)

    state.object_data = pd.DataFrame({
        "x": locations[:, 0],
        "y": locations[:, 1],
        "name": objects.names,
        "family": family_names[family_ids],
        "posterior_support": np.zeros(len(locations))
    })

    # return True, False, html.Div([filename])
    return False, html.Div([filename])


@app.callback(
    Output('uploaded-clusters', 'children'),
    Output('upload-clusters', 'children'),
    #    Output('upload-clusters', 'disabled'),
    Input('upload-clusters', 'contents'),
    Input('upload-clusters', 'filename'),
)
def update_clusters(content, filename):
    if content is None:
        return

    state.clusters_path = Path(filename)
    clusters_str = parse_content(content)
    state.clusters = clusters = parse_clusters_samples(clusters_str)
    n_clusters, n_samples, n_sites = clusters.shape

    fig = create_results_figure(state)

    state.fig = fig
    state.lines = fig.data[1:]
    state.scatter = fig.data[0]

    # Fix z-order so that lines are behind scatter:
    fig.data = fig.data[::-1]

    # for i in range(n_clusters):
    #     f = fig.add_trace(
    #         go.Scatter(x=[np.nan], y=[np.nan], legendgroup=f"Cluster {i}", marker_color=cluster_colors[i], name=f"Cluster {i}", visible="legendonly")
    #     )
    #
    # fig.update_layout(showlegend=True)

    results_components = components.build_results_components(state)
    return results_components, html.Div([filename])  # , True


@app.callback(
    Output("map", "figure"),
    Input("i_sample", "value"),
)
def update_map(i_sample: int):
    if state.clusters is None:
        return None

    state.i_sample = i_sample
    return plot_sample_map(i_sample, state)


@app.callback(
    Output("map", "figure"),
    Input("sample_range", "value"),
)
def update_map(sample_range: list[int]):
    if state.clusters is None:
        return None

    state.i_start, state.i_end = sample_range
    return plot_summary_map(state, sample_range)


@app.callback(
    Output("map", "figure"),
    Output("slider_div", "children"),
    Input("summarize_switch", "on"),
)
def switch_summarization(summarize: bool):
    if summarize:
        state.slider = components.get_summary_range_slider(state)
        slider_div = [
            html.P(children="Sample range", style={"font-size": 14, "text-indent": "10px"}),
            state.slider,
        ]
        map_figure = plot_summary_map(state, [0, state.n_samples])

    else:
        state.slider = components.get_sample_slider(state)
        slider_div = [
            html.P(children="Sample number", style={"font-size": 14, "text-indent": "10px"}),
            state.slider,
        ]
        map_figure = plot_sample_map(state.i_sample, state)

    return map_figure, slider_div


def main():
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
