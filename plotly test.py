import plotly.graph_objects as go
import social_media_simulation
from social_media_simulation import *

G = social_media_simulation.G
num_nodes = social_media_simulation.num_nodes
model = social_media_simulation.model
pos = nx.spring_layout(G)
general_pub_op = social_media_simulation.general_pub_op


def create_node_trace():
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(showscale=True, colorscale='curl', cmin=-1, cmax=1,
                                        reversescale=False, color=[], size=10,
                                        colorbar=dict(thickness=15, title='Opinion', xanchor='left', titleside='right'),
                                        line_width=2))

    # values for colour scale get stored here
    node_opinions = []
    node_text = []

    for node in G:
        node_opinions.append(round(G.nodes[node]['opinion'], 2))
        node_text.append('# of connections: ' + str(G.degree(node)) + '. opinion' + str(G.nodes[node]['opinion']))

    node_trace.marker.color = node_opinions
    node_trace.text = node_text

    return node_trace


def create_edge_trace():
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.3, color='#888'), hoverinfo='none', mode='lines')
    return edge_trace


def py_ploy_make_plot(num_steps):
    global pos

    for step in pos:
        x = list(pos[step])
        G.nodes[step]['pos'] = x

    node_trace = create_node_trace()
    edge_trace = create_edge_trace()

    fig2 = go.Figure(data=[node_trace],
                     layout=go.Layout(titlefont_size=16, showlegend=False, hovermode='closest',
                                      margin=dict(b=20, l=5, r=5, t=40),
                                      annotations=[dict(showarrow=False, xref="paper",
                                                        yref="paper", x=0.005, y=-0.002)],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                      )
                     )

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    frames = []

    # make each frame model step
    for step in range(num_steps):
        node_trace = create_node_trace()

        frame = go.Frame(data=[node_trace], name="step" + str(step))

        frames.append(frame)
        slider_step = {"args": [
            ["step" + str(step)],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": str(step),
            "method": "animate"}

        sliders_dict["steps"].append(slider_step)

        model.step()

    # add the frames to the animations
    fig2.frames = frames
    fig2.layout.sliders = [sliders_dict]

    fig2.update_layout(titlefont_size=16, showlegend=False, hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       annotations=[dict(text="", showarrow=False, xref="paper",
                                         yref="paper", x=0.005, y=-0.002)],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       updatemenus=[
                           dict(type='buttons', showactive=False, x=0.08, xanchor="right", y=-0.1, yanchor="top",
                                buttons=[{
                                    "args": [None, {"frame": {"duration": 500, "redraw": False}, "fromcurrent": True,
                                                    "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                                    "label": "Play",
                                    "method": "animate"
                                },

                                    {
                                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                          "mode": "immediate",
                                                          "transition": {"duration": 0}}],
                                        "label": "Pause",
                                        "method": "animate"
                                    }
                                ]
                                )]
                       )

    fig2.show()


py_ploy_make_plot(20)
print(general_pub_op)

