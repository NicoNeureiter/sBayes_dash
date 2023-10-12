from __future__ import annotations

import numpy as np
from plotly import express as px, graph_objects as go

from sbayes_dash.app_state import AppState
from sbayes_dash.util import min_and_max_with_padding, compute_delaunay, gabriel_graph_from_delaunay


def create_results_figure(state: AppState):
    """Initialize the figure to show sBayes results on the dashboard."""
    fig = px.scatter_geo(
        state.object_data,
        lat="y", lon="x",
        hover_data=["name", "family", "posterior_support"],
        projection="natural earth",
    )

    for i in range(state.n_clusters):
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
    return fig

def plot_summary_map(state: AppState, sample_range: list[int], posterior_threshold: float = 0.5):
    """Plot a summary map where an object is assigned to a cluster if its posterior
    frequency is above `posterior_threshold`."""
    i_start, i_end = sample_range
    colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)
    cluster_posterior = np.mean(state.clusters[:, i_start:i_end, :], axis=1)
    summary_clusters = (cluster_posterior > posterior_threshold)
    state.object_data.posterior_support = 1 - np.sum(cluster_posterior, axis=0)

    for i, c in enumerate(summary_clusters):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]
        state.scatter.customdata[c, 2] = cluster_posterior[i, c]
    state.scatter.hovertemplate = "y=%{lat}<br>x=%{lon}<br>name=%{customdata[0]}<br>family=%{customdata[1]}<br>posterior_support=%{customdata[2]:.2f}"

    state.scatter.marker.color = list(colors)
    return state.fig


def plot_sample_map(i_sample: int, state: AppState):
    """Plot a map of the clusters in a single posterior sample."""
    colors = np.full(state.objects.n_objects, "lightgrey", dtype=object)
    for i, c in enumerate(state.clusters[:, i_sample, :]):
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(state.locations[c])
        colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]
    state.scatter.hovertemplate = "y=%{lat}<br>x=%{lon}<br>name=%{customdata[0]}<br>family=%{customdata[1]}"
    state.scatter.marker.color = list(colors)
    return state.fig


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
