from typing import Any
import mesa
from statistics import mean
# from matplotlib import pyplot as plt, animation
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import random
import networkx as nx

# from matplotlib.colors import ListedColormap

general_pub_op = []

"""
WHAT I'M TRYING TO ACCOMPLISH:
- How does information spread within an established echo chamber?

- What factors can get opposing communities to come to an overall mean that's equal within the communities.

- Does individual opinion change have any effect on the over all mean opinions of the group? NO it does not

- What are the factors dictate how information flows in online communities leading to echo chambers ? 
    such as - likeliness to interact, recommendations based on positive reinforcement.
    drop far opinions has to be implemented! get rid of worst neighbours

- How does the presence of echo chambers affect the degree of polarization in a society?
    what factors will have people with opposing opinions endorse each other online?

- Higher the wright the more influence it has on the opinion of user reading.

- does overall public opinion have an influence on smaller twitter communities, one with no influence, 
    do they even out to the overall opinion within the community? to wht degree ???

- affect of single node with high influence within a neutral community. the part influence has to play and this points out the importance of bias control!!!

- Does the model perform as it should? yes it absolutely does. it was built on an even distrabution graph and has pradictable outputs on various different types of graphs

- are the findings consistent as the number of users scales and how do they compare with the real twitter network? DOES IT SCALE AND REMAIN CONSISTANT

- Comparison between influential information and public opinion. which is stronger. most nodes swayed? is influence stronger than general public opinion

- closer to an edge, lower the sway. lmao sure

"""


class User(mesa.Agent):
    """
    User Class defines 'user' agents present on a network, their attributes and behaviours.
    """

    # user class constructor
    def __init__(self, unique_id, network_model, opinion, tweeting_rate, influence):
        super().__init__(unique_id, network_model)

        self.opinion = opinion

        # more polarized they are the less they are going to be influenced by their surroundings
        if abs(opinion) == 1:
            self.sway_power = 0.2
            self.budge_value = 99.8

        elif 0.8 <= abs(opinion) < 1:
            self.sway_power = 1
            self.budge_value = 99

        elif 0.5 <= abs(opinion) < 0.8:
            self.sway_power = 15
            self.budge_value = 75

        else:
            self.sway_power = 45
            self.budge_value = 55

        self.sway = 50
        self.budge_value = 50

        self.influence = influence
        self.tweeting_rate = tweeting_rate

        self.neighbours = []
        self.screen = []
        self.tweets = []
        self.tweet_id = 0

        # to track tweets that you have interacted with, so it doesn't reappear
        self.interaction_history = []

    def calculate_sway(self):
        if 0.95 <= abs(self.opinion) <= 1:
            self.sway_power = 5
            self.budge_value = 95

        elif 0.8 <= abs(self.opinion) < 0.95:
            self.sway_power = 10
            self.budge_value = 90

        elif 0.3 <= abs(self.opinion) < 0.8:
            self.sway_power = 20
            self.budge_value = 80

        else:
            self.sway_power = 45
            self.budge_value = 55

    def retweet(self, tweet):
        for neighbour in self.neighbours:

            if (self.model.schedule.agents[neighbour].unique_id != tweet[1]) and (
                    len(self.model.schedule.agents[neighbour].screen) < 10):

                if tweet[1] not in self.model.schedule.agents[neighbour].neighbours:
                    # print("retweet made to", self.model.schedule.agents[neighbour].unique_id)
                    self.model.schedule.agents[neighbour].screen.append(tweet)

    # get list of neighbouring users - if user endorses someone with a different opinion then a connection forms.
    def my_neighbours(self):
        self.neighbours = model.get_friends(self.unique_id)

    def endorse_test(self, id, author):
        self.model.schedule.agents[author].tweets[id][1] += 1

    def check_for_worst_opinion(self, tweet_opinions):
        worst_opinion = self.opinion

        if self.opinion < 0:
            worst_opinion = max(tweet_opinions)
        elif self.opinion > 0:
            worst_opinion = min(tweet_opinions)

        if (abs(worst_opinion - self.opinion) > 0.2) and (worst_opinion * self.opinion < 0):
            return tweet_opinions.index(worst_opinion)

        else:
            return False

    # reads screen and updates opinion based on screen average
    def read_screen(self):
        tweet_opinions = []
        input_opinions = []

        for tweet in self.screen:
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

            if abs(opinion_difference) < 0.005:
                model.G.add_edge(self.unique_id, tweet_author)
                self.my_neighbours()
                self.model.schedule.agents[tweet_author].my_neighbours()

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

            if abs(tweet_opinion - self.opinion) < 0.5:
                self.endorse_test(tweet_id, tweet_author)
                self.retweet(tweet)

        # calculate screen opinion mean
        if tweet_opinions:
            average_screen_opinion = round(mean(tweet_opinions), 2)

            """
            incorporate the sway factor? likeliness to stray

            """

            if (abs(average_screen_opinion) > abs(self.opinion)) and (abs(average_screen_opinion - self.opinion) < 0.5):
                self.opinion = round(mean([self.opinion, average_screen_opinion]), 2)
                model.set_opinion(self.opinion, self.unique_id)

            else:
                self.opinion = round((self.opinion * self.budge_value + average_screen_opinion * self.sway_power) / 100,
                                     2)
                model.set_opinion(self.opinion, self.unique_id)

            worst_input = self.check_for_worst_opinion(input_opinions)
            # print("worst input pos: ", worst_input)

            if type(worst_input) == int:

                ex_friend = self.screen[worst_input]

                if model.G.has_edge(self.unique_id, ex_friend[1]):
                    model.G.remove_edge(self.unique_id, ex_friend[1])
                    self.my_neighbours()
                    self.model.schedule.agents[ex_friend[1]].my_neighbours()

        self.screen.clear()

    # function that lets users make a tweet
    def create_tweet(self):

        # weight of tweet - determines amount of influence when adjusting overall opinions.
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
            if len(self.model.schedule.agents[neighbour].screen) < 10:
                self.model.schedule.agents[neighbour].screen.append(tweet)
                # print(self.model.schedule.agents[neighbour].unique_id - 1, " ", self.model.schedule.agents[neighbour].screen)

    # user tweet reaches opinion threshold - leading to echo chambers?
    def drop_neighbour(self):
        if len(self.neighbours) > 1:
            for neighbour in self.neighbours:
                if abs(self.opinion) > 0.7 and (self.model.schedule.agents[neighbour].opinion * self.opinion < 0):
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

        # create a tweet and send it to your friends --- add this as a maybe. not everyone creates a tweet.
        self.send_tweet_to_friends(self.create_tweet())

        # self.calculate_sway()
        # updates the opinion
        self.read_screen()

        print("done with step")
        print("")


class Network(mesa.Model):
    """
    Network class defines the model and its parameters
    """
    global general_pub_op

    def __init__(self, n, graph, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_agents = n

        self.G = graph
        self.grid = NetworkGrid(self.G)

        self.schedule = RandomActivation(self)
        self.running = True
        self.step_count = 0

        # create the user agents based on input graph and add them to the model
        for i, node in enumerate(self.G.nodes()):
            op = self.G.nodes[i]['opinion']
            unique_id = self.G.nodes[i]['id']

            a = User(i, self, op, (round(random.uniform(1, 100))) / 100, influence=0)
            # a = User(i, self, op, (round(random.uniform(1, 100))) / 100, influence=0)

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

        return round(mean(public_opinion), 3), plus, minus

    def step(self):
        """Advance the model by one step."""

        self.step_count += 1
        print("Time step: ", self.step_count)
        self.schedule.step()

        a, p, m = self.calculate_population_opinion()

        general_pub_op.append(a)

        print("general public opinion: ", a, " plus: ", p, " minus: ", m)


# function to create base graph with even distribution to test and make model
def create_graph(n):
    construction_graph = nx.Graph()

    # Define the probability of connecting nodes with similar opinions
    # As the number of users increase, this decreases.
    p = 0.2

    # for a in range(n):
    #     if a < 10:
    #         # opinion = round(random.uniform(-1.00, -0.10), 2)
    #         opinion = -0.5
    #         construction_graph.add_node(a, opinion=opinion)
    #     elif a < 40:
    #         opinion = round(random.uniform(0.00, 1.00), 2)
    #         construction_graph.add_node(a, opinion=opinion)
    #     else:
    #         opinion = 1.00
    #         construction_graph.add_node(a, opinion=opinion)

    for a in range(n):
        opinion = round(random.uniform(-1.00, 1.00), 2)
        construction_graph.add_node(a, id=a, opinion=opinion)

    """
    make connection function for connection to different 
    """
    # Connect nodes with similar opinions - come back to this
    for a in range(n):
        for b in range(a + 1, n):
            n1 = construction_graph.nodes[a]['opinion']
            n2 = construction_graph.nodes[b]['opinion']
            difference = round(n1 - n2, 2)

            if n1 * n2 >= 0:
                # print('same side', n1, n2)

                if random.random() < p:
                    # print('edge created')
                    construction_graph.add_edge(a, b)


            elif abs(difference) < 0.2:
                # print('different sides:', n1, n2)
                connect_question = round(random.random(), 2)
                # print(l)
                if connect_question < 0.1:
                    # print('edge created')
                    construction_graph.add_edge(a, b)
            # else:
            #     if random.random() < 0.001:
            #         construction_graph.add_edge(a, b)

    # remove isolates from graph
    if list(nx.isolates(construction_graph)) != 0:
        for isolated_node in list(nx.isolates(construction_graph)):
            for j in range(isolated_node + 1, num_nodes):
                if (construction_graph.nodes[isolated_node]['opinion'] - construction_graph.nodes[j][
                    'opinion'] < 0.5) or construction_graph.nodes[j]['opinion'] == 0:
                    if random.random() < 0.2:
                        construction_graph.add_edge(isolated_node, j)

    return construction_graph


num_nodes = 100

G = create_graph(num_nodes)
print(G)

# init model
model = Network(num_nodes, G)
# model.step()
# print(" ")
# print(" ")
# print(" ")
# print(" ")
#
# model.step()

# print(" ")
# model.step()

print(G)

