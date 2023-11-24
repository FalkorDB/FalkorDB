import csv
import os
from datetime import date
from falkordb import Graph, Node, Edge

graph_name = "imdb"

def populate_graph(db, graph):
    # check if graph already exists
    if graph.key in db.list_graphs():
        return

    # Load movies entities
    movies = {}

    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/movies.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            title  = row[0]
            genre  = row[1]
            votes  = int(row[2])
            rating = float(row[3])
            year   = int(row[4])

            node = Node(labels="movie", properties={'title': title,
                                                    'genre': genre,
                                                    'votes': votes,
                                                    'rating': rating,
                                                    'year': year})
            movies[title] = node
            graph.add_node(node)

    # Load actors entities
    actors = {}

    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/actors.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name        = row[0]
            yearOfBirth = int(row[1])
            movie       = row[2]
            # All age calculations are done where 2019 is the the current year. 
            age         = 2019 - yearOfBirth

            if name not in actors:
                node = Node(labels="actor", properties={'name': name, 'age': age})
                actors[name] = node
                graph.add_node(node)

            if movie in movies:
                edge = Edge(actors[name], "act", movies[movie])
                graph.add_edge(edge)

    graph.commit()
    graph.create_node_fulltext_index("actor", "name")

    return (actors, movies)
