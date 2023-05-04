import random
import networkx as nx

"""
Connection threshold determines the connectivity of the graph. Further nodes are likely to connect with a high threshold
"""


# even distribution graph
def create_graph(n, connection_threshold, p):
    construction_graph = nx.Graph()

    # Define the probability of connecting nodes with similar opinions
    # As the number of users increase, this decreases.

    # p = 0.15

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


            if n1 * n2 > 0:
                if random.random() < p:
                    construction_graph.add_edge(a, b)
            elif abs(difference) < connection_threshold:
                if random.random() < p/2:
                    construction_graph.add_edge(a, b)


            # elif abs(difference) < 0.4:
            #     # print('different sides:', n1, n2)
            #     connect_question = round(random.random(), 2)
            #     # print(l)
            #     if random.random() < 0.05:
            #         # print('edge created')
            #         construction_graph.add_edge(a, b)
            # else:
            #     if n1*n2 >= 0:
            #         # print('same side', n1, n2)
            #
            #         if random.random() < p:
            #             # print('edge created')
            #             construction_graph.add_edge(a, b)
            #     if random.random() < 0.01:
            #         construction_graph.add_edge(a, b)

    # remove isolates from graph
    if list(nx.isolates(construction_graph)) != 0:
        for isolated_node in list(nx.isolates(construction_graph)):
            for j in range(isolated_node + 1, n):
                if (construction_graph.nodes[isolated_node]['opinion'] - construction_graph.nodes[j]['opinion'] < 0.5) \
                        or construction_graph.nodes[j]['opinion'] == 0:
                    if random.random() < 0.2:
                        construction_graph.add_edge(isolated_node, j)

    return construction_graph


# community - neutral, polarized, even
def influence_tester_graph_builder(n, community):
    construction_graph = nx.Graph()

    for a in range(n):
        if a == 1:
            opinion = 1
            construction_graph.add_node(a, opinion=opinion)

        # elif a == 2:
        #     opinion = - 1
        #     construction_graph.add_node(a, opinion=opinion)

        else:
            if community == "neutral":
                opinion = round(random.uniform(-0.3, 0.0), 2)
                construction_graph.add_node(a, opinion=opinion)
            elif community == "polarized":
                opinion = round(random.uniform(-1, 0.0), 2)
                construction_graph.add_node(a, opinion=opinion)
            elif community == "even":
                opinion = round(random.uniform(-1, 1), 2)
                construction_graph.add_node(a, opinion=opinion)

    for a in range(n):
        if a == 1:
            for b in range(a + 1, n):
                if random.random() < 0.5:
                    construction_graph.add_edge(a, b)
        else:
            for b in range(a + 1, n):
                if random.random() < 0.1:
                    construction_graph.add_edge(a, b)


    # remove isolates from graph
    if list(nx.isolates(construction_graph)) != 0:
        for isolated_node in list(nx.isolates(construction_graph)):
            for j in range(isolated_node + 1, n):
                if (construction_graph.nodes[isolated_node]['opinion'] - construction_graph.nodes[j]['opinion'] < 0.5)\
                        or construction_graph.nodes[j]['opinion'] == 0:
                    if random.random() < 0.2:
                        construction_graph.add_edge(isolated_node, j)
        if list(nx.isolates(construction_graph)) != 0:
            for isolated_node in list(nx.isolates(construction_graph)):
                new_connection = construction_graph.nodes[random.randint(isolated_node, n)]
                construction_graph.add_edge(isolated_node, new_connection)

    return construction_graph

