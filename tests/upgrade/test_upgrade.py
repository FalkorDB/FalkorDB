import time
import sys
import os
import threading
import docker
import docker.models
import docker.models.containers

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../flow/")

from common import *
from graph_utils import *
from random_graph import *
from falkordb import FalkorDB


def display_logs(container: docker.models.containers.Container):
    line_text = ""
    for line in container.logs(stream=True):
        if line == b"\n":
            print(line_text)
            print('\n')
            line_text = ""
        else:
            line_text += line.decode("utf-8", errors="ignore").strip()


# starts db using docker
def run_db(image):
    from random import randint

    # Initialize the Docker client
    client = docker.from_env()

    random_port = randint(49152, 65535)

    # Run the FalkorDB container
    container = client.containers.run(
        image,  # Image
        detach=True,  # Run container in the background
        ports={"6379/tcp": random_port},  # Map port 6379
    )

    return container, random_port


# stop and remove docker container
def stop_db(container):
    container.stop()
    container.remove()


class test_upgrade:
    def __init__(self):
        self.env, self.replica_db = Env()
        self.replica_conn = self.env.getConnection()

    def upgrade(self, image):
        # start FalkorDB previous version
        container, master_port = run_db(image)

        try:
            # wait for DB to accept connections
            time.sleep(10)

            master_db = FalkorDB(host="127.0.0.1", port=master_port)

            master_graph = master_db.select_graph("upgrade")

            # Generate random graph
            nodes, edges = create_random_schema()
            res = create_random_graph(master_graph, nodes, edges)
            res = run_random_graph_ops(master_graph, nodes, edges, ALL_OPS)

            # Connect FlakorDB latest version as a replica and wait for it to sync
            self.replica_conn.replicaof("localhost", master_port)

            # Wait for replica to connect
            time.sleep(3)

            # Wait for replica to sync
            master_db.connection.wait(1, 0)

            replica_graph = self.replica_db.select_graph("upgrade")

            # Validate that both DBs are the same

            master_db.config_set("RESULTSET_SIZE", -1)  # unlimited result-set size

            if not graph_eq(replica_graph, master_graph):
                self.env.assertTrue(graph_eq(replica_graph, master_graph))

            # Terminate docker container
            stop_db(container)

            self.replica_conn.replicaof("NO", "ONE")
        except Exception as e:
            if container is not None:
                stop_db(container)
            raise e

    def test_v14_upgrade(self):
        image = "falkordb/falkordb:v4.0.7"
        self.upgrade(image)

    def test_v15_upgrade(self):
        image = "falkordb/falkordb:v4.2.2"
        self.upgrade(image)
