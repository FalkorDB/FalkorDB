import csv
import os
from falkordb import Graph, Node, Edge

graph_name = "social"


def populate_graph(con, graph):
    if con.exists(graph_name):
        return

    persons = {}   # dictionary person name to its node entity
    countries = {} # dictionary country name to its node entity
    node_count = 0

    # Create country entities
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/countries.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name = row[0]
            node = Node(alias=f"n_{node_count}", labels="country", properties={"name": name})
            countries[name] = node
            node_count += 1

    # Create person entities
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/person.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name = row[0]
            age = int(row[1])
            gender = row[2]
            status = row[3]
            node = Node(alias=f"n_{node_count}", labels="person", properties={"name": name,
                                                    "age": age,
                                                    "gender": gender,
                                                    "status": status})

            persons[name] = node
            node_count += 1

    # Connect people to places they've visited.
    edges = []
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/visits.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            person = row[0]
            country = row[1]
            purpose = row[2]
            edge = Edge(persons[person],
                        "visited",
                        countries[country],
                        properties={'purpose': purpose})
            edges.append(edge)

    # Connect friends
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/friends.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            person = persons[row[0]]
            friend = persons[row[1]]
            edge = Edge(person, "friend", friend)
            edges.append(edge)

    nodes_str = [str(node) for node in persons.values()] + [str(node) for node in countries.values()]
    edges_str = [str(edge) for edge in edges]
    graph.query(f"CREATE {','.join(nodes_str + edges_str)}")
