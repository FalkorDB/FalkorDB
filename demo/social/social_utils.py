import csv
import os
from falkordb import Graph, Node, Edge

graph_name = "social"


def populate_graph(redis_con, redis_graph):
    if redis_con.exists(graph_name):
        return

    # dictionary person name to its node entity
    persons = {}
    # dictionary country name to its node entity
    countries = {}

    # Create country entities
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/countries.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name = row[0]
            node = Node(labels="country", properties={"name": name})
            countries[name] = node
            redis_graph.add_node(node)

    # Create person entities
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/person.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name = row[0]
            age = int(row[1])
            gender = row[2]
            status = row[3]
            node = Node(labels="person", properties={"name": name,
                                                    "age": age,
                                                    "gender": gender,
                                                    "status": status})

            persons[name] = node
            redis_graph.add_node(node)

    # Connect people to places they've visited.
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
            redis_graph.add_edge(edge)

    # Connect friends
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/friends.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            person = persons[row[0]]
            friend = persons[row[1]]
            edge = Edge(person, "friend", friend)
            redis_graph.add_edge(edge)

    redis_graph.commit()
