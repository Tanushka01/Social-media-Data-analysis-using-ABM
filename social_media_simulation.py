import random
from typing import Any
import mesa
from statistics import mean
# from matplotlib import pyplot as plt, animation
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import plotly.graph_objects as go
from graphs import *

general_pub_op = []
maximum = []
minimum = []


# from matplotlib.colors import ListedColormap


class User(mesa.Agent):
    """
    User Class defines 'user' agents present on a network, their attributes and behaviours.
    """

    # user class constructor
    def __init__(self, unique_id, network_model, opinion, influence, experiment):
        super().__init__(unique_id, network_model)

        self.experiment = experiment  # dictates which combinations of parameters to run

        # where the user lies on the polarization scale (-1, 1)
        self.opinion = opinion
        self.opinion_flexibility = 50  # flexibility of opinion is diverging to that of the timeline.

        # not everyone should have influence - only needed when testing the effect of influence
        if 13 <= self.experiment < 19:
            if self.unique_id == 1:
                self.influence = 100
            else:
                self.influence = influence

        else:
            self.influence = influence

        # self.tweeting_probability = [0, 25, 50, 75, 90, 100]
        self.tweeting_probability = False

        # the threshold of accepting further away opinions. based on absolute difference
        self.p_of_dropping_worst_neighbour = []

        # more polarized they are the less they are going to be influenced by their surroundings maybe?
        # if abs(opinion) == 1 or abs(opinion) == -1:
        #     self.opinion_flexibility = 0.2
        #     self.budge_value = 99.8
        #
        # elif 0.8 <= abs(opinion) < 1:
        #     self.opinion_flexibility = 1
        #     self.budge_value = 99
        #
        # elif 0.6 <= abs(opinion) < 0.8:
        #     self.opinion_flexibility = 10
        #     self.budge_value = 90
        #
        # else:
        #     self.opinion_flexibility = 20
        #     self.budge_value = 80

        self.neighbours = []
        self.timeline = []
        self.tweets = []
        self.tweet_id = 0

    # calculate opinion flexibility
    def calculate_o_flexibility(self):
        if 0.95 <= abs(self.opinion) <= 1:
            self.opinion_flexibility = 5

        elif 0.8 <= abs(self.opinion) < 0.95:
            self.opinion_flexibility = 10

        elif 0.3 <= abs(self.opinion) < 0.8:
            self.opinion_flexibility = 20

        else:
            self.opinion_flexibility = random.randint(5, 50)

    # get list of neighbouring users - if user endorses someone with a different opinion then a connection forms.
    def my_neighbours(self):
        self.neighbours = model.get_friends(self.unique_id)

    # function that lets users make a tweet
    def create_tweet(self):

        # weight of tweet - determines amount of influence when adjusting overall opinions.
        if self.unique_id == 1 or self.unique_id == 2:
            weight = 300
        else:
            weight = 1

        tweet_opinion = self.opinion

        # tweet sent to other users - they point back to the author to have one place where weight is updated
        tweet = [self.tweet_id, self.unique_id, tweet_opinion, self.influence]

        for_self = [self.tweet_id, weight]

        self.tweets.append(for_self)
        self.tweet_id += 1
        return tweet
    def send_tweet_to_friends(self, tweet):

        # add tweet to the screens of all neighbours
        for neighbour in self.neighbours:
            if len(self.model.schedule.agents[neighbour].timeline) < 20:
                if self.experiment > 13:
                    if self.model.schedule.agents[neighbour].unique_id != 1:
                        self.model.schedule.agents[neighbour].timeline.append(tweet)
                else:
                    self.model.schedule.agents[neighbour].timeline.append(tweet)

    # functions for endorsement
    def like_tweet(self, tweet_id, author):
        self.model.schedule.agents[author].tweets[tweet_id][1] += 1
    def retweet(self, tweet):
        for neighbour in self.neighbours:
            if self.model.schedule.agents[neighbour].unique_id != 1:
                if (self.model.schedule.agents[neighbour].unique_id != tweet[1]) and (
                        len(self.model.schedule.agents[neighbour].timeline) < 10):

                    if tweet[1] not in self.model.schedule.agents[neighbour].neighbours:
                        # print("retweet made to", self.model.schedule.agents[neighbour].unique_id)
                        self.model.schedule.agents[neighbour].timeline.append(tweet)

    def check_for_worst_opinion(self, tweet_opinions, drop_threshold):
        worst_opinion = self.opinion

        if self.opinion < 0:
            worst_opinion = max(tweet_opinions)
        elif self.opinion > 0:
            worst_opinion = min(tweet_opinions)

        # if worst input is over the threshold, remove
        #         if abs(worst_opinion - self.opinion) >= drop_threshold and (worst_opinion*self.opinion < 0):
        if abs(worst_opinion - self.opinion) >= drop_threshold:

            if abs(worst_opinion) != 0.0:
                return tweet_opinions.index(worst_opinion)

        else:
            return False

    # reads timeline and updates opinion based on timeline average
    def read_timeline(self, endorsement_index, drop_threshold):
        tweet_opinions = []
        input_opinions = []

        for tweet in self.timeline:
            tweet_id = tweet[0]
            tweet_author = tweet[1]
            tweet_opinion = tweet[2]
            author_influence = tweet[3]
            tweet_weight = self.model.schedule.agents[tweet_author].tweets[tweet_id][1]

            opinion_difference = tweet_opinion - self.opinion

            # depending on weight, adds to tweet_opinions list.
            for i in range(tweet_weight):
                tweet_opinions.append(tweet_opinion)

            input_opinions.append(tweet_opinion)

            # if abs(opinion_difference) < 0.005:
            #     model.G.add_edge(self.unique_id, tweet_author)
            #     self.my_neighbours()
            #     self.model.schedule.agents[tweet_author].my_neighbours()

            # make friends
            # if tweet_author not in self.neighbours:
            #     if abs(self.opinion) <= abs(tweet_opinion) and random.random() < 0.1:
            #         model.G.add_edge(self.unique_id, tweet_author)
            #         self.my_neighbours()
            #         self.model.schedule.agents[tweet_author].my_neighbours()

            # elif (-1 <= self.opinion <= 0.0) and (-1 <= tweet_opinion < -0.5):
            #     model.G.add_edge(self.unique_id, tweet_author)
            #     self.my_neighbours()
            #     self.model.schedule.agents[tweet_author].my_neighbours()
            #
            # elif (0.0 <= self.opinion <= 1) and (0.5 < tweet_opinion <= 1):
            #     model.G.add_edge(self.unique_id, tweet_author)
            #     self.my_neighbours()
            #     self.model.schedule.agents[tweet_author].my_neighbours()
            #
            # elif abs(opinion_difference) < 0.005:
            #     model.G.add_edge(self.unique_id, tweet_author)
            #     self.my_neighbours()
            #     self.model.schedule.agents[tweet_author].my_neighbours()

            if abs(opinion_difference) <= endorsement_index:
                self.like_tweet(tweet_id, tweet_author)
                if random.uniform(0, 100) < 50:
                    self.retweet(tweet)

        # calculate timeline opinion mean
        if tweet_opinions:
            average_screen_opinion = round(mean(tweet_opinions), 2)

            # if (abs(average_screen_opinion) > abs(self.opinion)) and (abs(average_screen_opinion - self.opinion) < 0.1):
            #     self.opinion = round(mean([self.opinion, average_screen_opinion]), 2)
            #     model.set_opinion(self.opinion, self.unique_id)

            self.opinion = round((self.opinion * (
                    100 - self.opinion_flexibility) + average_screen_opinion * self.opinion_flexibility) / 100, 2)
            model.set_opinion(self.opinion, self.unique_id)

            if self.experiment < 7:
                worst_input = self.check_for_worst_opinion(input_opinions, drop_threshold)
                # print("worst input pos: ", worst_input)

                if type(worst_input) == int:

                    ex_friend = self.timeline[worst_input]

                    if model.G.has_edge(self.unique_id, ex_friend[1]):
                        model.G.remove_edge(self.unique_id, ex_friend[1])
                        self.my_neighbours()
                        self.model.schedule.agents[ex_friend[1]].my_neighbours()

            self.timeline.clear()


    # user tweet reaches opinion threshold - leading to echo chambers? have to incorporate sway
    def drop_neighbour(self, drop_threshold):
        if len(self.neighbours) > 1:
            for neighbour in self.neighbours:
                if abs(self.opinion - self.model.schedule.agents[neighbour].opinion) > drop_threshold and (
                        self.opinion * self.model.schedule.agents[neighbour].opinion < 0):

                    if len(self.model.schedule.agents[neighbour].neighbours) > 1:
                        model.G.remove_edge(self.unique_id, self.model.schedule.agents[neighbour].unique_id)
                    self.my_neighbours()
                    self.model.schedule.agents[neighbour].my_neighbours()

    # tester function for model behaviour behaviour
    def pick_agent(self):

        if len(self.neighbours) > 0:
            agent = self.model.schedule.agents[random.choice(self.neighbours)]

            return agent.unique_id

        else:
            return "no neighbours"

    """
    range of endorsement experiment 1:
    constants - tweeting rate distributed between 0 - 100.
                Opinion flexibility - 50/50
                Dropping worst input - exceeding absolute difference of 1 - dependant on graph
                Not looking at User Influence
    """

    def range_of_endorsement_ex1(self, range_of_endorsement, drop_threshold):

        self.opinion_flexibility = 50

        if random.randint(0, model.tweeting_probability_range) < self.tweeting_probability:
            self.send_tweet_to_friends(self.create_tweet())

        # read timeline and update opinion
        self.read_timeline(range_of_endorsement, drop_threshold)

    def dropping_worst_neighbour_ex2(self, drop_threshold):
        self.opinion_flexibility = 50

        if random.randint(0, model.tweeting_probability_range) < self.tweeting_probability:
            self.send_tweet_to_friends(self.create_tweet())

        self.read_timeline(0.5, 0.00)
        self.drop_neighbour(drop_threshold)

    def influence_experiment_ex3(self, e):

        # only high influence node can tweet - as a control
        if e == 1:
            self.opinion_flexibility = 50

            if self.unique_id == 1:
                self.influence = 100
                self.send_tweet_to_friends(self.create_tweet())

            self.read_timeline(0.5, 0.00)

        # opposing crowd
        if e == 2:
            self.opinion_flexibility = 10

            if self.unique_id == 1:
                self.influence = 100
                self.send_tweet_to_friends(self.create_tweet())

            self.read_timeline(0.5, 0.00)

        # varying sway
        if e == 3:
            if model.step_count == 1:
                self.calculate_o_flexibility()

            if self.unique_id == 1:
                self.influence = 100
                self.send_tweet_to_friends(self.create_tweet())

            self.read_timeline(0.5, 0.00)

        # others can tweet on mixed graph
        if e == 4:
            if model.step_count == 1:
                self.calculate_o_flexibility()

            if self.unique_id == 1:
                self.influence = 100
                self.send_tweet_to_friends(self.create_tweet())

            elif random.randint(0, model.tweeting_probability_range) < self.tweeting_probability:
                self.send_tweet_to_friends(self.create_tweet())
                self.read_timeline(0.5, 0.8)

    def step(self) -> None:
        """
        At each time step users can do one of the following:
        make a tweet based on their opinions and inclination to interact with the network,
        read their screens and choose to endorse and/or update their opinions
        drop/acquire worst/best neighbours

        always use unique_id - 1 when using model functions
        """

        # get a list of your neighbours
        self.my_neighbours()

        # set tweeting probability
        if not self.tweeting_probability:
            self.tweeting_probability = random.randint(0, model.tweeting_probability_range)

        # range of endorsement
        if self.experiment < 7:
            if self.experiment == 1:
                self.range_of_endorsement_ex1(1.5, 0.7)
            if self.experiment == 2:
                self.range_of_endorsement_ex1(1, 0.6)
            if self.experiment == 3:
                self.range_of_endorsement_ex1(0.5, 0.5)
            if self.experiment == 4:
                self.range_of_endorsement_ex1(0.25, 0.4)
            if self.experiment == 5:
                self.range_of_endorsement_ex1(0.152, 0.3)
            if self.experiment == 6:
                self.range_of_endorsement_ex1(0.05, 0.25)

        # dropping worst neighbour
        elif self.experiment < 13:
            if self.experiment == 7:
                self.dropping_worst_neighbour_ex2(1.5)
            if self.experiment == 8:
                self.dropping_worst_neighbour_ex2(1)
            if self.experiment == 9:
                self.dropping_worst_neighbour_ex2(0.75)
            if self.experiment == 10:
                self.dropping_worst_neighbour_ex2(0.5)
            if self.experiment == 11:
                self.dropping_worst_neighbour_ex2(0.25)
            if self.experiment == 12:
                self.dropping_worst_neighbour_ex2(0.05)

        # role of influence
        elif self.experiment < 19:
            if self.experiment == 13:
                self.influence_experiment_ex3(1)
            if self.experiment == 14:
                self.influence_experiment_ex3(2)
            if self.experiment == 15:
                self.influence_experiment_ex3(3)
            if self.experiment == 16:
                self.influence_experiment_ex3(4)

        # # get a list of your neighbours
        # self.my_neighbours()
        #
        # # create a tweet and send it to your friends --- add this as a maybe. not everyone creates a tweet.
        # if self.unique_id == 1 or self.unique_id == 2:
        #     self.send_tweet_to_friends(self.create_tweet())
        #
        # probability_of_tweet = random.uniform(0, 100)
        #
        # # self.calculate_sway()
        # # updates the opinion
        # self.read_timeline()
        #
        # #  different steps for different parameters


class Network(mesa.Model):
    """
    Network class defines the model and its parameters
    """
    global general_pub_op
    global minimum, maximum

    def __init__(self, n, graph, experiment, tweeting_probability_range=100, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_agents = n
        self.experiment = experiment

        self.G = graph
        self.grid = NetworkGrid(self.G)

        self.tweeting_probability = [0, 25, 50, 75, 90, 100]
        self.tweeting_probability_range = tweeting_probability_range

        self.schedule = RandomActivation(self)
        self.running = True
        self.step_count = 0

        # create the user agents based on input graph and add them to the model
        for i, node in enumerate(self.G.nodes()):
            op = self.G.nodes[i]['opinion']

            a = User(i, self, opinion=op, influence=0, experiment=self.experiment)

            self.schedule.add(a)
            self.grid.place_agent(a, node)

    # allows edit of opinion on the input graph
    def set_opinion(self, o, node):
        self.G.nodes[node]['opinion'] = o

    def get_opinion(self, node):
        return self.G.nodes[node]['opinion']

    def get_step_count(self):
        return self.step_count

    # getter for obtaining the neighbours/friends of a given user
    def get_friends(self, node):
        friends = [n for n in self.G.neighbors(node)]
        return friends

    def calculate_population_opinion(self):
        public_opinion = []
        plus = 0
        minus = 0
        for i, node in enumerate(self.G.nodes()):
            public_opinion.append(self.G.nodes[i]['opinion'])
            if 0 > self.G.nodes[i]['opinion']:
                minus += 1

            elif 0 < self.G.nodes[i]['opinion']:
                plus += 1

        return public_opinion, plus, minus

    def step(self):
        """Advance the model by one step."""

        self.step_count += 1
        if self.step_count == 1:
            print("Model Running (This may take a second)")

        self.schedule.step()

        a, p, m = self.calculate_population_opinion()

        minimum.append(min(a))
        maximum.append(max(a))

        general_pub_op.append(round(mean(a), 3))


num_nodes = 300
edges_dropped_1 = []
edges_dropped_2 = []

G = create_graph(num_nodes, connection_threshold=1, p=0.1)
pos1 = nx.spring_layout(G)
initial_edge_count = len(G.edges)
G1 = G.copy()
G2 = G.copy()
G3 = G.copy()
G4 = G.copy()
G5 = G.copy()
G6 = G.copy()
# graphs = [G, G1, G2, G3, G4, G5]

neutral_influence_graph = influence_tester_graph_builder(num_nodes, "neutral")
pos2 = nx.spring_layout(neutral_influence_graph)
polarized_influence_graph = influence_tester_graph_builder(num_nodes, "polarized")
even_influence_graph = influence_tester_graph_builder(num_nodes, "even")
IG1 = neutral_influence_graph.copy()
IG2 = polarized_influence_graph.copy()
IG3 = even_influence_graph.copy()
IG4 = even_influence_graph.copy()




def create_node_trace(graph):
    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(showscale=True, colorscale='curl', cmin=-1, cmax=1,
                                        reversescale=False, color=[], size=10
                                        ,
                                        colorbar=dict(thickness=15, title='Opinion', xanchor='left', titleside='right'),
                                        line_width=2))

    # values for colour scale get stored here
    node_opinions = []
    node_text = []

    for node in graph:
        node_opinions.append(round(graph.nodes[node]['opinion'], 2))
        node_text.append( ' node: ' + str(node) + ' # of connections: ' + str(graph.degree(node)) + '. opinion' + str(graph.nodes[node]['opinion']))

    node_trace.marker.color = node_opinions
    node_trace.text = node_text

    return node_trace


def create_edge_trace(graph):
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # if G.nodes[edge[0]]['opinion']*G.nodes[edge[1]]['opinion'] > 0:
        #     if G.nodes[edge[0]]['opinion'] > 0:
        #         colour.append('blue')
        #     elif G.nodes[edge[0]]['opinion'] < 0:
        #         colour.append('red')
        #     else:
        #         colour.append('#888')

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.3, color='#b7bbbd'), hoverinfo='none', mode='lines')
    return edge_trace


def make_plot(graph, num_steps, model, pos):
    for step in pos:
        x = list(pos[step])
        graph.nodes[step]['pos'] = x

    node_trace = create_node_trace(graph)
    edge_trace = create_edge_trace(graph)

    fig2 = go.Figure(data=[edge_trace, node_trace],
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

    # make each frame model step
    frames = []
    for step in range(num_steps + 1):
        node_trace = create_node_trace(graph)
        # edge_trace = create_edge_trace(graph)

        if step % 3 == 0:
            edge_trace = create_edge_trace(graph)

        frame = go.Frame(data=[edge_trace, node_trace], name="step" + str(step))

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


def line_plots(maximums, minimums, averages, n):
    # Define the x-axis and y-axis data
    x_data = list(range(1, n))
    y_data_1 = maximums
    y_data_2 = minimums
    y_data_3 = averages

    # Define the trace for each line
    trace_1 = go.Scatter(x=x_data, y=y_data_1, hovertemplate=y_data_1, mode='lines', name='Maximum')
    trace_2 = go.Scatter(x=x_data, y=y_data_2, hovertemplate=y_data_2, mode='lines', name='Minimum')
    trace_3 = go.Scatter(x=x_data, y=y_data_3, hovertemplate=y_data_3, mode='lines', name='Average Public Opinion')

    # Define the layout of the graph
    layout = go.Layout(title='Range of endorsement', xaxis=dict(title='Time Steps'), yaxis=dict(title='Opinion Scale'))

    # Define the data to be plotted
    data = [trace_1, trace_2, trace_3]

    # Create the figure and plot the data
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def influence_experiment_line_plots_single(minimums, averages, n):
    # Define the x-axis and y-axis data
    x_data = list(range(1, n))
    y_data_1 = maximums
    y_data_2 = minimums
    y_data_3 = averages

    # Define the trace for each line
    trace_1 = go.Scatter(x=x_data, y=y_data_1, hovertemplate=y_data_1, mode='lines', name='Maximum')
    trace_2 = go.Scatter(x=x_data, y=y_data_2, hovertemplate=y_data_2, mode='lines', name='Minimum')
    trace_3 = go.Scatter(x=x_data, y=y_data_3, hovertemplate=y_data_3, mode='lines', name='Average Public Opinion')

    # Define the layout of the graph
    layout = go.Layout(title='Range of endorsement', xaxis=dict(title='Time Steps'), yaxis=dict(title='Opinion Scale'))

    # Define the data to be plotted
    data = [trace_1, trace_2, trace_3]

    # Create the figure and plot the data
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def influence_experiment_line_plots(ic1, ic2, ic3, ic4, n):
    # Define the x-axis and y-axis data
    x_data = list(range(1, n))
    y_data_1 = ic1
    y_data_2 = ic2
    y_data_3 = ic3
    y_data_4 = ic4

    # Define the trace for each line
    trace_1 = go.Scatter(x=x_data, y=y_data_1, hovertemplate=y_data_1, mode='lines', name='Experiment 1')
    trace_2 = go.Scatter(x=x_data, y=y_data_2, hovertemplate=y_data_2, mode='lines', name='Experiment 2')
    trace_3 = go.Scatter(x=x_data, y=y_data_3, hovertemplate=y_data_3, mode='lines', name='Experiment 3')
    trace_4 = go.Scatter(x=x_data, y=y_data_4, hovertemplate=y_data_4, mode='lines', name='Experiment 4')


    # Define the layout of the graph
    layout = go.Layout(title='Role of Influence', xaxis=dict(title='Time Steps'), yaxis=dict(title='Opinion'))

    # Define the data to be plotted
    data = [trace_1, trace_2, trace_3, trace_4]

    # Create the figure and plot the data
    fig = go.Figure(data=data, layout=layout)
    fig.show()


model = Network(num_nodes, G, 1)


def experiment_func_filler(graph, experiment, pos, steps=10):
    global model
    global general_pub_op
    global minimum
    global maximum
    global G

    maximum = []
    general_pub_op = []
    minimum = []
    model = Network(num_nodes, graph, experiment)
    make_plot(graph, steps, model, pos)
    line_plots(maximum, minimum, general_pub_op, steps)
    print("Number of edges after model run: ", len(graph.edges))
    return len(graph.edges)



def range_of_endorsement_experiment(experiment, steps):
    global model
    global general_pub_op
    global minimum
    global maximum
    global initial_edge_count
    global edges_dropped_1
    global G
    global G1, G2, G3, G4, G5, G6

    x = len(G.edges)

    print("Number of edges before model run: ", initial_edge_count)

    if experiment == 1:
        edx = experiment_func_filler(G1, 1, pos1, steps)
        edges_dropped_1.append(x-edx)
        G1 = G.copy()

    elif experiment == 2:
        edx = experiment_func_filler(G2, 2, pos1, steps)
        edges_dropped_1.append(x-edx)
        G2 = G.copy()


    elif experiment == 3:
        edx = experiment_func_filler(G3, 3, pos1, steps)
        edges_dropped_1.append(x-edx)


        G3 = G.copy()

    elif experiment == 4:
        edx = experiment_func_filler(G4, 4, pos1, steps)
        edges_dropped_1.append(x-edx)

        G4 = G.copy()

    elif experiment == 5:
        edx = experiment_func_filler(G5, 5, pos1, steps)
        edges_dropped_1.append(x-edx)


        G5 = G.copy()

    elif experiment == 6:
        edx = experiment_func_filler(G6, 6, pos1, steps)
        edges_dropped_1.append(x-edx)


        G6 = G.copy()


def dropping_neighbour_experiment(experiment, steps):
    global model
    global general_pub_op
    global minimum
    global maximum
    global edges_dropped_2
    global G, G1, G2, G3, G4, G5, G6

    x = len(G.edges)

    print("Number of edges before model run: ", initial_edge_count)



    if experiment == 7:
        edx = experiment_func_filler(G1, 7, pos1, steps)
        edges_dropped_2.append(x-edx)

        G1 = G.copy()
    elif experiment == 8:
        edx = experiment_func_filler(G2, 8, pos1, steps)
        edges_dropped_2.append(x-edx)

        G2 = G.copy()
    elif experiment == 9:
        edx = experiment_func_filler(G3, 9, pos1, steps)
        edges_dropped_2.append(x-edx)

        G3 = G.copy()
    elif experiment == 10:
        edx = experiment_func_filler(G4, 10, pos1, steps)
        edges_dropped_2.append(x-edx)

        G4 = G.copy()
    elif experiment == 11:
        edx = experiment_func_filler(G5, 11, pos1, steps)
        edges_dropped_2.append(x-edx)

        G5 = G.copy()
    elif experiment == 12:
        edx = experiment_func_filler(G6, 12, pos1, steps)
        edges_dropped_2.append(x-edx)
        G6 = G.copy()


influence_convergence1 = []
influence_convergence2 = []
influence_convergence3 = []
influence_convergence4 = []


def influence_experiment(experiment, steps):
    global model, pos2, general_pub_op, minimum
    global IG1, IG2, IG3, IG4, neutral_influence_graph, polarized_influence_graph, even_influence_graph
    global influence_convergence1, influence_convergence2, influence_convergence3, influence_convergence4


    general_pub_op = []
    minimum = []

    if experiment == 13:
        model = Network(num_nodes, IG1, 13)
        make_plot(IG1, steps, model, pos2)
        influence_convergence1 = general_pub_op
        print(influence_convergence1)
        IG1 = neutral_influence_graph.copy
    elif experiment == 14:
        general_pub_op = []
        minimum = []
        model = Network(num_nodes, IG2, 14)
        make_plot(IG2, steps, model, pos2)
        influence_convergence2 = general_pub_op
        print(influence_convergence2)


        IG2 = polarized_influence_graph.copy
    elif experiment == 15:
        general_pub_op = []
        minimum = []
        model = Network(num_nodes, IG3, 15)
        make_plot(IG3, steps, model, pos2)
        influence_convergence3 = general_pub_op
        print(influence_convergence3)

        IG3 = even_influence_graph.copy
    elif experiment == 16:
        general_pub_op = []
        minimum = []
        model = Network(num_nodes, IG4, 16)
        make_plot(IG4, steps, model, pos2)
        influence_convergence4 = general_pub_op
        print(influence_convergence4)

        IG4 = even_influence_graph.copy


influence_experiment(13, 50)
influence_experiment(14, 50)
influence_experiment(15, 50)
influence_experiment(16, 50)
influence_experiment_line_plots(influence_convergence1, influence_convergence2, influence_convergence3, influence_convergence4, 50)
