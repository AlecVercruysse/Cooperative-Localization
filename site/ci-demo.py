from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pdb

def get_cov_ellipse_params(x, y, cov):
    """
    alpha: angle of ellipse
    major: width of ellipse
    minor: height of ellipse
    """
    w, v = np.linalg.eig(cov)
    lambda1, lambda2 = w
    v1, v2 = v.T
    alpha = np.arctan2(v1[1], v1[0])
    major = 2*np.sqrt(5.991*lambda1)
    minor = 2*np.sqrt(5.991*lambda2)
    return alpha, major, minor

def add_cov(fig, x, cov, c='blue'):
    alpha, major, minor = get_cov_ellipse_params(*x, cov)
    theta = np.linspace(0, 2*np.pi, 30)
    xy = np.array([major * np.cos(theta), minor * np.sin(theta)])
    # pdb.set_trace()
    rot_matrix = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    points = rot_matrix @ xy + x.reshape(2, 1)
    path = f"M {points[0,0]}, {points[1,0]}" + \
        " ".join([f"L{x}, {y}" for x, y in points[:, 1:].T]) + \
        " Z"
    fig.add_shape(type="path", path=path, fillcolor=c, opacity = 0.4,
                  row=1, col=1)


def do_CI(a, cova, b, covb, ret_traces=False):
    # pdb.set_trace()
    a = a.reshape(2, 1)
    b = b.reshape(2, 1)
    traces = []  # brute force optimize omega
    omegas = np.linspace(0, 1)
    for omega in omegas:
        try:
            covc = np.linalg.inv(omega * np.linalg.inv(cova) +
                                 (1 - omega ) * np.linalg.inv(covb))
        except np.linalg.LinAlgError:
            print(f"LinAlgError computing CI for {omega=}")
            covc = cova*2 # out of range
        traces += [np.trace(covc)]
    best_trace_idx = np.argmin(traces)
    omega = omegas[best_trace_idx]
    covc = np.linalg.inv(omega * np.linalg.inv(cova) +
                         (1 - omega ) * np.linalg.inv(covb))
    C = covc @ (omega * np.linalg.inv(cova) @ a +
                (1 - omega) * np.linalg.inv(covb) @ b)
    if not ret_traces:
        return C, covc
    else:
        return C, covc, traces

def do_kalman(a, cova, b, covb):
    a = a.reshape(2, 1)
    b = b.reshape(2, 1)

    try:
        K = cova @ np.linalg.inv(cova + covb)
    except np.linalg.LinAlgError:
        print("singular matrix in Kalman update!")
        return a * 0, b*0
    c = a + K @ (b - a)
    C = (np.eye(2) - K) @ cova
    return c, C

def getitem(name, rgb):
    r, g, b = rgb
    d = {
        "label": html.Div(name),
                          #style={"backgroundColor":f"rgba({r}, {g}, {b}, 0.4)"}),
                          #className="border rounded m-3 p-3"),
        "value": name
    }
    return d
    
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dbc.Row([
        html.H4('The Covariance Intersection Algorithm', style={"text-align":"center"}),
        dcc.Graph(id="graph", style={"height": "800px", "width":"800px"})
    ], justify="center"),
    dbc.Row([
        html.Div([
            dcc.Checklist(["CI result", "Kalman result", "A input", "B input"],
                          ["CI result", "Kalman result", "A input", "B input"],
                          id='display-options',
                          labelClassName="col border rounded m-1 pl-3",
                          labelStyle={"display":"flex"},
                          inputStyle={"margin-right":"3px"},
                          className="row displayer")
        ], className="container", style={"width": "600px"}),
    ], justify="center"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("x variance"),
                daq.Slider(id='x1-variance', updatemode='drag', value=.8, max=1, step=.01, marks={0:0, 1:1}),
                html.Br(),
                html.P("y variance"),
                daq.Slider(id='y1-variance', updatemode='drag', value=.05, max=1, step=.01, marks={0:0, 1:1}),
                html.Br(),
                html.P("x-y correlation"),
                daq.Slider(id='cov-1', updatemode='drag', min=-1, max=1, value=0.8, step=.01, marks={-1:-1, 0:0, 1:1}),
                html.Br()
            ], style={"float":"right", "backgroundColor":"rgba(255, 0, 0, 0.4)"}, className="border rounded p-3")
        ]),
        dbc.Col(
            html.Div([
                html.P("x variance"),
                daq.Slider(id='x2-variance', updatemode='drag', value=.1, max=1, step=.01, marks={0:0, 1:1}),
                html.Br(),
                html.P("y variance"),
                daq.Slider(id='y2-variance', updatemode='drag', value=.7, max=1, step=.01, marks={0:0, 1:1}),
                html.Br(),
                html.P("x-y correlation"),
                daq.Slider(id='cov-2', updatemode='drag', value=-.3,  min=-1, max=1, step=.01, marks={-1:-1, 0:0, 1:1}),
                html.Br(),
                html.P("x position"),
                daq.Slider(id='x2-pos', updatemode='drag', value=3,  min=-5, max=5, step=.1, marks={-5:-5, 0:0, 5:5}),
                html.Br(),
                html.P("y position"),
                daq.Slider(id='y2-pos', updatemode='drag', value=1,  min=-5, max=5, step=.1, marks={-5:-5, 0:0, 5:5}),
                html.Br()
            ], style={"float":"left","backgroundColor":"rgba(0, 0, 255, 0.4)"}, className="border rounded p-3")
        )],
    )], style={"height":"100vh"})

@app.callback(
    Output("graph", "figure"), 
    Input("x1-variance", "value"),
    Input("y1-variance", "value"),
    Input("cov-1", "value"),
    Input("x2-variance", "value"),
    Input("y2-variance", "value"),
    Input("cov-2", "value"),
    Input("x2-pos", "value"),
    Input("y2-pos", "value"),
    Input("display-options", "value"))
def update_fig(x1, y1, corr1, x2, y2, corr2, x2pos, y2pos, display_options):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["95% Confidence Covariance Ellipses",
                                        "Trace of Covariance Matrix vs. Omega"],
                        row_heights=[.75, .25],
                        vertical_spacing=0.1)
    
    center1 = np.array([0, 0])
    center2 = np.array([x2pos, y2pos])
    cov1 = corr1 * np.sqrt(x1) * np.sqrt(y1)
    cov2 = corr2 * np.sqrt(x2) * np.sqrt(y2)
    covm1 = np.array([[x1, cov1], [cov1, y1]])
    covm2 = np.array([[x2, cov2], [cov2, y2]])
    
    cires, cicov, traces = do_CI(center1, covm1,
                                 center2, covm2,
                                 ret_traces=True)
    Kres, Kcov = do_kalman(center1, covm1,
                           center2, covm2)

    if "A input" in display_options:
        add_cov(fig, center1, covm1, c='red')
    if "B input" in display_options:
        add_cov(fig, center2, covm2, c='blue')
    if "CI result" in display_options:
        add_cov(fig, cires, cicov, c='green')
    if "Kalman result" in display_options:
        add_cov(fig, Kres, Kcov, c='black')
    
    fig.update_yaxes(range=(-6, 6), row=1, col=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    

    fig.add_trace(go.Scatter(x=np.linspace(0, 1),
                             y=traces), row=2, col=1)
    amin = traces.index(min(traces))
    fig.add_trace(go.Scatter(x=[amin/50],
                             y=[traces[amin]],
                             marker={"color":"green"}),
                             row=2, col=1)
    fig.update_yaxes(range=(0, max(max(traces), 1)), row=2, col=1)
    fig.update_xaxes(range=(0, 1), row=2, col=1)
    fig.update_layout(showlegend=False)
    
    return fig

app.run_server(debug=True)
