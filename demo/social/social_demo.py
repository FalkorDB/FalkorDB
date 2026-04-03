import os
import sys
import redis
import argparse
from falkordb import Graph

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from social_queries import queries_info
import social_utils
from utils import execute_query, _redis

con = None
graph = None


def run_queries():
    print("Querying...\n")

    for query_info in queries_info:
        execute_query(graph,
                      query_info.description,
                      query_info.query)

def debug(host, port):
    global con
    global graph
    db = FalkorDB(host=host, port=port)
    con = db.connection
    graph = db.select_graph(social_utils.graph_name)

    print("populate_graph")
    social_utils.populate_graph(con, graph)

    print("run_queries")
    run_queries()

def main(argv):
    global con
    global graph

    parser = argparse.ArgumentParser(description='Social demo.', add_help=False)
    parser.add_argument('-h', '--host', dest='host', help='redis host')
    parser.add_argument('-p', '--port', dest='port', type=int, help='redis port')
    parser.add_argument("--debug", action='store_const', const=True)
    args = parser.parse_args()

    if args.debug is not None:
        debug('127.0.0.1', 6379)
    elif args.host is not None and args.port is not None:
        debug(args.host, args.port)
    else:
        db = FalkorDB()
        con = db.connection
        graph = db.select_graph(social_utils.graph_name)
        social_utils.populate_graph(con, graph)
        run_queries()

if __name__ == '__main__':
    main(sys.argv[1:])
