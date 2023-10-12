from dash import html, dcc
import dash_daq as daq

from sbayes_dash.app_state import AppState

upload_box_style = {
    "width": "98%",
    "height": "40px",
    "lineHeight": "40px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px",
    "font-variant": "small-caps",
}


def get_base_layout() -> html.Div:
    return html.Div(
        children=[
            # html.Img(src='assets/sbayes_logo.png', style={"width": "10%"}),
            html.Div([dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'drag and drop or select the ', html.B('data file'), ' (.csv)'
                ]),
                style=upload_box_style,
                disabled=False,
                style_disabled={"opacity": 0.3},
            )], style={"width": "50%", "display": "inline-block"}),
            html.Div([dcc.Upload(
                id='upload-clusters',
                children=html.Div([
                    'drag and drop or select the ', html.B('clusters file'), ' (clusters_*.txt)'
                ]),
                style=upload_box_style,
                disabled=True,
                style_disabled={"opacity": 0.3},
            )], style={"width": "50%", "display": "inline-block"},),
            html.Div(id='uploaded-clusters')
        ], style={"font-family": "sans-serif"}
    )


def get_sample_slider(state: AppState) -> dcc.Slider:
    return dcc.Slider(
            id="i_sample", value=0, step=1, min=0, max=state.n_samples-1,
            marks={i: str(i) for i in range(0, state.n_samples, max(1, state.n_samples//10))},
    )


def get_summary_range_slider(state: AppState) -> dcc.RangeSlider:
    return dcc.RangeSlider(
        id="sample_range", value=[0, state.n_samples], step=1, min=0, max=state.n_samples,
        marks={i: str(i) for i in range(0, state.n_samples, max(1, state.n_samples//10))},
    )


def build_results_components(state: AppState) -> html.Div:
    state.slider = get_sample_slider(state)
    return html.Div([
        html.Div([
                html.P(id="sample", children="Sample number", style={"font-size": 14, "text-indent": "10px"}),
                state.slider,
            ],
            style={"width": "90%", "display": "inline-block"},
            id="slider_div"
        ),
        html.Div([
            daq.BooleanSwitch(id="summarize_switch", label={"label": "Summarize samples", "style": {"font-size": 14}},
                              labelPosition="top")
        ], style={"width": "9%", "display": "inline-block"}),
        dcc.Graph(id="map"),
    ])
