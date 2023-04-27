from typing import Any
import mesa
from statistics import mean
from mesa import model
from matplotlib import pyplot as plt, animation
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import random
import networkx as nx
import plotly.graph_objects as go

import social_media_simulation
from social_media_simulation import User
from social_media_simulation import *

if __name__ == '__main__':
    # create the graph

    G = social_media_simulation.G
    num_nodes = social_media_simulation.num_nodes
    model = social_media_simulation.model
    """
    stuff for plotting: come back to it after finalizing the model to observe behaviour.
    """
    pos = nx.spring_layout(G)
    # colors = ['blue' if G.nodes[i]['opinion'] == 0.00 else 'pink' if 0 > G.nodes[i]['opinion'] >= -1 else 'red' if 0 < G.nodes[i]['opinion'] <= 1 else 'black'
    #          for i in range(num_nodes)]

    colors = []

    for i in range(num_nodes):
        if -1.00 < G.nodes[i]['opinion'] <= -0.50:
            colors.append('red')
        elif -0.50 < G.nodes[i]['opinion'] <= 0.00:
            colors.append('pink')

        elif 0.00 < G.nodes[i]['opinion'] <= 0.50:
            colors.append('lightblue')

        elif 0.50 < G.nodes[i]['opinion'] <= 1.00:
            colors.append('blue')
        else:
            colors.append('green')

    fig, ax = plt.subplots()
    it_number = 0


    def animate(frame):
        global it_number
        global colors
        colors = []
        iteration = "user network " + str(it_number)
        fig.clear()
        model.step()

        it_number += 1
        plt.axis('off')
        plt.title(iteration)

        for i in range(num_nodes):
            if -1.0 < G.nodes[i]['opinion'] < -0.5:
                colors.append('red')
            elif -0.5 < G.nodes[i]['opinion'] < 0.0:
                colors.append('pink')

            elif 0.0 < G.nodes[i]['opinion'] < 0.5:
                colors.append('lightblue')

            elif 0.5 < G.nodes[i]['opinion'] < 1.0:
                colors.append('blue')
            else:
                colors.append('green')

        ax.set_title("User Network")

        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)


    def make_plot():
        plt.axis('off')
        global it_number

        plt.title("Time step: " + str(it_number))

        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)

        # nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.5)

        ani = animation.FuncAnimation(fig, animate, interval=1000, repeat=False, cache_frame_data=False)

        plt.show()


    # make_plot()

    def py_ploy_make_plot():
        global pos

        for i in pos:
            x = list(pos[i])
            G.nodes[i]['pos'] = x

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

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.9, color='#888'),
                                hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                marker=dict(showscale=True, colorscale='curl', cmin=-1.5, cmax=1.5,
                                            reversescale=True, color=[], size=10,
                                            colorbar=dict(thickness=15, title='Opinion', xanchor='left',
                                                          titleside='right'),
                                            line_width=2))

        # values for colour scale get stored here
        node_opinions = []
        node_text = []

        for node, adjacencies in enumerate(G.adjacency()):
            node_opinions.append(round(G.nodes[node]['opinion']))
            node_text.append(
                '# of connections: ' + str(len(adjacencies[1])) + '. opinion' + str(G.nodes[node]['opinion']))

        print(node_opinions)
        print(mean(node_opinions))
        node_trace.marker.color = node_opinions
        node_trace.text = node_text

        fig2 = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(titlefont_size=16, showlegend=False, hovermode='closest',
                                          margin=dict(b=20, l=5, r=5, t=40),
                                          annotations=[dict(showarrow=False, xref="paper",
                                                            yref="paper", x=0.005, y=-0.002)],
                                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)

                                          )
                         )
        frames = []

        for i in range(6):
            model.step()

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = G.nodes[node]['pos']
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                    marker=dict(showscale=True, colorscale='curl', cmin=-1.5, cmax=1.5,
                                                reversescale=True, color=[], size=10,
                                                colorbar=dict(thickness=15, title='Opinion', xanchor='left',
                                                              titleside='right'),
                                                line_width=2))

            # values for colour scale get stored here
            node_opinions = []

            for node, adjacencies in enumerate(G.adjacency()):
                node_opinions.append(round(G.nodes[node]['opinion']))
                node_text.append(
                    '# of connections: ' + str(len(adjacencies[1])) + '. opinion: ' + str(G.nodes[node]['opinion']))

            node_trace.marker.color = node_opinions
            node_trace.text = node_text

            frame = go.Frame(data=[edge_trace, node_trace])

            frames.append(frame)

        fig2.frames = frames

        fig2.update_layout(titlefont_size=16, showlegend=False, hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(text="step " + str(0), showarrow=False, xref="paper",
                                             yref="paper", x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           updatemenus=[dict(type='buttons', showactive=False, x=0.1, xanchor="right", y= 0, yanchor="top",
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

        """
        [{
        'buttons': 
        [{'args': [None, {'frame': {'duration': 500, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}], 'label': 'Play', 'method': 'animate'}, {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}]

        """


    py_ploy_make_plot()
