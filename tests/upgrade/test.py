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
        network=os.getenv("DOCKER_NETWORK", None),  # Use host network
    )

    print("started container " + container.id)

    # output container logs in a separate thread
    threading.Thread(target=display_logs, args=(container,)).start()

    return container, random_port


# stop and remove docker container
def stop_db(container):
    container.stop()
    container.remove()


if __name__ == "__main__":
    image = "falkordb/falkordb:v4.0.7"
    container, master_port = run_db(image)
    time.sleep(2)
    master_db = FalkorDB(port=master_port)

    master_graph = master_db.select_graph("upgrade")
    print(master_graph.query("MATCH (n) RETURN n").result_set)

    # stop and remove the container
    stop_db(container)
