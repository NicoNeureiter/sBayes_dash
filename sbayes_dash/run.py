from __future__ import annotations

import os; os.environ['USE_PYGEOS'] = '0'  # Fix for Pandas deprecation warning

from io import StringIO
import base64
from dataclasses import dataclass
from pathlib import Path

from jupyter_dash import JupyterDash
from dash import Input, Output, html, dcc, State
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
import dash_daq as daq
from plotly import express as px, graph_objects as go
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from numpy.typing import NDArray

from sbayes_dash.load_data import Confounder, Objects
from sbayes_dash.util import compute_delaunay, gabriel_graph_from_delaunay, parse_cluster_columns
from sbayes_dash.util import min_and_max_with_padding, read_data_csv, reproject_locations


# data_projection: str = "+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs "
data_projection: str = "EPSG:4326"


def find_biggest_angle_gap(degrees: NDArray[float]) -> float:
    degrees = np.sort(degrees)
    np.append(degrees, degrees[0] + 360)
    i = np.argmax(np.diff(degrees))
    return (degrees[i+1] + degrees[i]) / 2


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


def cluster_to_graph(locations):
    if len(locations) < 2:
        return [], []
    delaunay = compute_delaunay(locations)
    graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

    x, y = [], []
    for i1, i2 in graph_connections:
        x += [locations[i1, 0], locations[i2, 0], None]
        y += [locations[i1, 1], locations[i2, 1], None]
    return x, y


@dataclass
class AppState:

    clusters_path = None
    _clusters = None
    fig = None
    lines = None
    scatter = None
    cluster_colors = None
    locations = None
    object_data = None
    objects = None
    i_sample = 0
    burnin = 0

    slider = None



    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters
        self.cluster_colors = self.get_cluster_colors(self.n_clusters)

    @staticmethod
    def get_cluster_colors(K):
        # cm = plt.get_cmap('gist_rainbow')
        # cluster_colors = [colors.to_hex(c) for c in cm(np.linspace(0, 1, K, endpoint=False))]
        colors = []
        for i, x in enumerate(np.linspace(0, 1, K, endpoint=False)):
            b = i % 2
            h = x % 1
            s = 0.6 + 0.4 * b
            v = 0.5 + 0.3 * (1 - b)
            colors.append(
                mpl_colors.to_hex(mpl_colors.hsv_to_rgb((h, s, v)))
            )
        return colors

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]


# Initialized app
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()], suppress_callback_exceptions=True)
# app = JupyterDash(__name__, suppress_callback_exceptions=True)
server = app.server
state = AppState()


upload_box_style = {
    "width": "98%",
    "height": "40px",
    "lineHeight": "40px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px",
}

# Set up the layout
app.layout = html.Div(
    children=[
        html.Div([dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and drop or select the ', html.B('data file')
            ]),
            style=upload_box_style,
            disabled=False,
            style_disabled={"opacity": 0.3},
        )], style={"width": "50%", "display": "inline-block"}),
        html.Div([dcc.Upload(
            id='upload-clusters',
            children=html.Div([
                'Drag and drop or select the ', html.B('clusters file')
            ]),
            style=upload_box_style,
            disabled=True,
            style_disabled={"opacity": 0.3},
        )], style={"width": "50%", "display": "inline-block"},),
        html.Div(id='uploaded-clusters')
    ]
)


@app.callback(
    Output('upload-data', 'disabled'),
    Output('upload-clusters', 'disabled'),
    Input('upload-data', 'contents'),
)
def update_data(content):
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
    })

    return True, False


@app.callback(
    Output('uploaded-clusters', 'children'),
    Output('upload-clusters', 'disabled'),
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

    fig = px.scatter_geo(
        state.object_data,
        lat="y", lon="x",
        hover_data=["name", "family"],
        projection="natural earth",
    )

    for i in range(n_clusters):
        fig_lines = px.line_geo(lat=[None], lon=[None])
        fig = go.Figure(fig.data + fig_lines.data)

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        geo=dict(
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 0])],
                dtick=5,
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(state.locations[:, 1])],
                dtick=5,
            ),
        ),
    )

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

    state.slider = dcc.Slider(
        id="i_sample", value=0, step=1, min=0, max=state.n_samples-1,
        marks={i: str(i) for i in range(0, state.n_samples, max(1, state.n_samples//10))},
    )

    results_components = html.Div([
        html.Div(
    [
                html.P(id="sample", children="Sample number"),
                state.slider,
            ],
            style={"width": "90%", "display": "inline-block"},
        ),
        html.Div([daq.BooleanSwitch(id="summarize_switch", label="Summarize samples", labelPosition="top")],
                 style={"width": "9%", "display": "inline-block"}),
        dcc.Graph(id="map"),
    ])

    return results_components, True


def plot_sample_map(i_sample: int):
    colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)
    for i, c in enumerate(state.clusters[:, i_sample, :]):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]

    state.scatter.marker.color = list(colors)
    return state.fig


def plot_summary_map():
    colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)
    summary_clusters = np.mean(state.clusters[:, state.burnin:, :], axis=1) > 0.5
    for i, c in enumerate(summary_clusters):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]

    state.scatter.marker.color = list(colors)
    return state.fig


@app.callback(
    Output("map", "figure"),
    Input("i_sample", "value"),
)
def update_map(i_sample: int):
    if state.clusters is None:
        return None

    state.i_sample = i_sample
    return plot_sample_map(i_sample=i_sample)


@app.callback(
    Output("map", "figure"),
    Output("i_sample", "disabled"),
    Input("summarize_switch", "on"),
)
def switch_summarization(summarize: bool):
    if summarize:
        return plot_summary_map(), True
    else:
        return plot_sample_map(i_sample=state.i_sample), False


def main():
    app.run_server(debug=True)

    
if __name__ == '__main__':
    main()
