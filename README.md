<div align="center">
  <img width="100" height="100" alt="FalkorDB Logo Square B" src="https://github.com/user-attachments/assets/ec67e5e3-a060-405d-b229-1fe2badcf8ad" />
</div>
<div align="center">
  <h1>FalkorDB</h1>
  <h2>Ultra-fast, Multi-tenant Graph Database</h2>
  <h3>Powering Generative AI, Agent Memory, Cloud Security, and Fraud Detection</h3>
</div>
<div align="center">
  <a href="https://app.falkordb.cloud/signup">
    <img src="https://img.shields.io/badge/Try%20Free-FalkorDB%20Cloud-FF8101?labelColor=FDE900&style=for-the-badge" alt="Try Free" />
  </a><br>
  <a href="https://discord.com/invite/TJ4SaPTptf" rel="nofollow">
    <img src="https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white" alt="Discord" />
  </a>
  <a href="https://hub.docker.com/r/falkordb/falkordb/">
    <img src="https://img.shields.io/docker/pulls/falkordb/falkordb?label=Docker" alt="Dockerhub" />
  </a>
  <a href="https://discord.gg/6M4QwDXn2w">
    <img src="https://img.shields.io/discord/1146782921294884966?style=flat-square" alt="Discord" />
  </a>
  <a href="https://codecov.io/gh/falkordb/falkordb">
    <img src="https://codecov.io/gh/falkordb/falkordb/graph/badge.svg?token=0G4HBEJMW0" alt="codecov" />
  </a>
  <a href="https://github.com/FalkorDB/FalkorDB/actions/workflows/build.yml">
    <img src="https://github.com/FalkorDB/FalkorDB/actions/workflows/build.yml/badge.svg?branch=master" alt="Workflow" />
  </a>
</div>

<div align="center">

![FalkorDB GitHub Repo - Video - 640x365](https://github.com/user-attachments/assets/131867d4-fa2b-41f8-ac72-5c5f6eaf4e12)

</div>

## UNIQUE FEATURES

Our goal is to build a high-performance Knowledge Graph tailored for Large Language Models (LLMs), prioritizing exceptionally low latency to ensure fast and efficient information delivery through our Graph Database.

🆕 [FalkorDB](https://www.falkordb.com/) is the first **queryable [Property Graph](https://github.com/opencypher/openCypher/blob/master/docs/property-graph-model.adoc) database to leverage sparse matrices** for representing the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) in graphs and [linear algebra](https://en.wikipedia.org/wiki/Adjacency_matrix) for querying.

### Key Features

* **Sparse Matrix Representation**: Utilizes sparse matrices to represent adjacency matrices, optimizing storage and performance.


* **Linear Algebra Querying**: Employs linear algebra for query execution, enhancing computational efficiency.


* **Property Graph Model Compliance**: Supports nodes and relationships with attributes, adhering to the Property Graph Model.


* **OpenCypher Support:** Compatible with [OpenCypher](ttps://github.com/opencypher/openCypher/blob/master/docs/property-graph-model.adoc) query language, including proprietary extensions for advanced querying capabilities.

>Explore FalkorDB in action by visiting the [Demos](https://github.com/FalkorDB/FalkorDB/tree/master/demo).

## DOCUMENTATION

[Official Docs](https://docs.falkordb.com/) | [Clients](https://docs.falkordb.com/clients.html) | [Commands](https://docs.falkordb.com/commands/) | 📊 [Latest Performance Benchmarks](https://benchmark.falkordb.com/)

### Community and Support

* **Discussions**: Join our community discussions on [GitHub Discussions](https://github.com/FalkorDB/FalkorDB/discussions) to ask questions, share ideas, and connect with other users.

* **Contributing**: We welcome contributions! Please see our [Contributing Guide](https://github.com/FalkorDB/FalkorDB/blob/master/CONTRIBUTING.md) for more details.

* **License**: This project is licensed under the Server Side Public License v1 (SSPLv1). See the [LICENSE](https://github.com/FalkorDB/FalkorDB/blob/master/LICENSE.txt) file for details.

## GET STARTED

### Step 1

To quickly try out FalkorDB, launch an instance using docker:

```
docker run -p 6379:6379 -p 3000:3000 -it --rm -v ./data:/var/lib/falkordb/data falkordb/falkordb
```

### Step 2

Then, open your browser and navigate to `http://localhost:3000`.


You can also interact with FalkorDB using any of the supported [Client Libraries](https://docs.falkordb.com/clients.html)

### MotoGP League Example

In this example, we'll use the [FalkorDB Python client](https://pypi.org/project/FalkorDB/) to create a small graph representing a subset of motorcycle riders and teams participating in the MotoGP league. After creating the graph, we'll query the data to explore its structure and relationships.

```python
from falkordb import FalkorDB

# Connect to FalkorDB
db = FalkorDB(host='localhost', port=6379)

# Create the 'MotoGP' graph
g = db.select_graph('MotoGP')
g.query("""CREATE (:Rider {name:'Valentino Rossi'})-[:rides]->(:Team {name:'Yamaha'}),
                  (:Rider {name:'Dani Pedrosa'})-[:rides]->(:Team {name:'Honda'}),
                  (:Rider {name:'Andrea Dovizioso'})-[:rides]->(:Team {name:'Ducati'})""")

# Query which riders represents Yamaha?
res = g.query("""MATCH (r:Rider)-[:rides]->(t:Team)
                 WHERE t.name = 'Yamaha'
                 RETURN r.name""")

for row in res.result_set:
	print(row[0])

# Prints: "Valentino Rossi"

# Query how many riders represent team Ducati ?
res = g.query("""MATCH (r:Rider)-[:rides]->(t:Team {name:'Ducati'})
                 RETURN count(r)""")

print(res.result_set[0][0])
# Prints: 1
```

## START BUILDING

### Compiling

Make sure to complete these requirements:

1️⃣ The FalkorDB repository: `git clone --recurse-submodules -j8 https://github.com/FalkorDB/FalkorDB.git`

>On Ubuntu Linux, run: `apt-get install build-essential cmake m4 automake peg libtool autoconf python3 python3-pip`
	
>On OS X, verify that `homebrew` is installed and run: `brew install cmake m4 automake peg libtool autoconf`.
>>The version of Clang that ships with the OS X toolchain does not support OpenMP, which is a requirement for FalkorDB. One way to resolve this is to run `brew install gcc g++` and follow the on-screen instructions to update the symbolic links. Note that this is a system-wide change - setting the environment variables for `CC` and `CXX` will work if that is not an option.

2️⃣ Build by running `make` in the project's directory.

Congratulations! You can find the compiled binary at `bin/<arch>/src/falkordb.so`.

### Running tests

Start by installing the required Python packages by running ```pip install -r requirements.txt``` from the ```tests``` directory.

> Note: If you've got ```redis-server``` in PATH, just invoke ```make test```. Otherwise, invoke ```REDIS_SERVER=<redis-server-location> make test```. For a more verbose output, run ```make test V=1```.

### Building in a docker

The FalkorDB build system runs within docker. For detailed instructions on building, please [see here](https://docs.falkordb.com/docker-examples/README.html).

## LOADING FALKORDB INTO REDIS

FalkorDB is hosted by [Redis](https://redis.io), so you'll first have to load it as a Module to a Redis server. 
> Note: [Redis 7.4](https://redis.io/download) is required for the latest FalkorDB version.

💡 We recommend having Redis load FalkorDB during startup by adding the following to your redis.conf file:

```
loadmodule /path/to/module/src/falkordb.so
```

In the line above, replace `/path/to/module/src/falkordb.so` with the actual path to FalkorDB's library.
If Redis is running as a service, you must ensure that the `redis` user (default) has the necessary file/folder permissions
to access `falkordb.so`.

Alternatively, you can have Redis load FalkorDB using the following command line argument syntax:

```sh
~/$ redis-server --loadmodule /path/to/module/src/falkordb.so
```

Lastly, you can also use the [`MODULE LOAD`](http://redis.io/commands/module-load) command. Note, however, that `MODULE LOAD` is a dangerous command and may be blocked/deprecated in the future due to security considerations.

Once you've successfully loaded FalkorDB your Redis log should see lines similar to:

```
...
30707:M 20 Jun 02:08:12.314 * Module 'graph' loaded from <redacted>/src/falkordb.so
...
```

If the server fails to launch with output similar to:

```
# Module /usr/lib/redis/modules/falkordb.so failed to load: libgomp.so.1: cannot open shared object file: No such file or directory
# Can't load module from /usr/lib/redis/modules/falkordb.so: server aborting
```

The system is missing the run-time dependency OpenMP. This can be installed on Ubuntu with `apt-get install libgomp1`, on RHEL/CentOS with `yum install libgomp`, and on OSX with `brew install libomp`.

## USING FALKORDB

You can call FalkorDB's commands from any Redis client. Here are several methods:

### With `redis-cli`

```sh
$ redis-cli
127.0.0.1:6379> GRAPH.QUERY social "CREATE (:person {name: 'roi', age: 33, gender: 'male', status: 'married'})"
```

### With any other client

You can interact with FalkorDB using your client's ability to send raw Redis commands.

>Note: Depending on your client of choice, the exact method for doing that may vary.

#### Example: Using FalkorDB with a Python client

This code snippet shows how to use FalkorDB with from Python using [falkordb-py](https://github.com/FalkorDB/falkordb-py):

```Python
from falkordb import FalkorDB

# Connect to FalkorDB
db = FalkorDB(host='localhost', port=6379)

# Select the social graph
g = db.select_graph('social')

reply = g.query("CREATE (:person {name:'roi', age:33, gender:'male', status:'married'})")
```

## CLIENT LIBRARIES

>Note: Some languages have client libraries that provide support for FalkorDB's commands:

### Official Clients


| Project                                                   | Language   | License | Author                                      | Stars                                                             | Package | Comment    |
| --------------------------------------------------------- | ---------- | ------- | ------------------------------------------- | ----------------------------------------------------------------- | ------- | ---------- |
| [jfalkordb][jfalkordb-url] | Java | BSD | [FalkorDB][falkordb-url] | [![jfalkordb-stars]][jfalkordb-url] | [Maven][jfalkordb-package]||
| [falkordb-py][falkordb-py-url] | Python | MIT | [FalkorDB][falkordb-url] | [![falkordb-py-stars]][falkordb-py-url] | [pypi][falkordb-py-package]||
| [falkordb-ts][falkordb-ts-url] | Node.JS | MIT | [FalkorDB][falkordb-url] | [![falkordb-ts-stars]][falkordb-ts-url] | [npm][falkordb-ts-package]||
| [falkordb-rs][falkordb-rs-url] | Rust | MIT | [FalkorDB][falkordb-url] | [![falkordb-rs-stars]][falkordb-rs-url] | [Crate][falkordb-rs-package]||
| [falkordb-go][falkordb-go-url] | Go | BSD | [FalkorDB][falkordb-url] | [![falkordb-go-stars]][falkordb-go-url] | [GitHub][falkordb-go-url]||
| [NFalkorDB][nfalkordb-url]  |  C# | Apache-2.0 | [FalkorDB][falkordb-url] | [![nfalkordb-stars]][nfalkordb-url] | [nuget][nfalkordb-package] | |


### Additional Clients

| Project                                                   | Language   | License | Author                                      | Stars                                                             | Package | Comment    |
| --------------------------------------------------------- | ---------- | ------- | ------------------------------------------- | ----------------------------------------------------------------- | ------- | ---------- |
| [nredisstack][nredisstack-url] | .NET | MIT | [Redis][redis-url] | [![nredisstack-stars]][nredisstack-url] | [nuget][nredisstack-package]||
| [redisgraph-rb][redisgraph-rb-url]                        | Ruby       | BSD     | [Redis][redisgraph-rb-author]          | [![redisgraph-rb-stars]][redisgraph-rb-url]                       | [GitHub][redisgraph-rb-url] ||
| [redgraph][redgraph-url]                                  | Ruby       | MIT     | [pzac][redgraph-author]                | [![redgraph-stars]][redgraph-url]                                 | [GitHub][redgraph-url] ||
| [redisgraph-go][redisgraph-go-url]                        | Go         | BSD     | [Redis][redisgraph-go-author]          | [![redisgraph-go-stars]][redisgraph-go-url]                       | [GitHub][redisgraph-go-url]||
| [rueidis][rueidis-url]                                    | Go         | Apache 2.0 | [Rueian][rueidis-author]          | [![rueidis-stars]][rueidis-url]                       | [GitHub][rueidis-url] ||
| [ioredisgraph][ioredisgraph-url]                          | JavaScript | ISC     | [Jonah][ioredisgraph-author]                | [![ioredisgraph-stars]][ioredisgraph-url]                         | [GitHub][ioredisgraph-url] ||
| [@hydre/rgraph][rgraph-url]                               | JavaScript | MIT     | [Sceat][rgraph-author]                      | [![rgraph-stars]][rgraph-url]                                     | [GitHub][rgraph-url] ||
| [php-redis-graph][php-redis-graph-url]                    | PHP        | MIT     | [KJDev][php-redis-graph-author]             | [![php-redis-graph-stars]][php-redis-graph-url]                   | [GitHub][php-redis-graph-url] ||
| [redisgraph_php][redisgraph_php-url]                      | PHP        | MIT     | [jpbourbon][redisgraph_php-author]          | [![redisgraph_php-stars]][redisgraph_php-url]                     | [GitHub][redisgraph_php-url] ||
| [redisgraph-ex][redisgraph-ex-url]                        | Elixir     | MIT     | [crflynn][redisgraph-ex-author]             | [![redisgraph-ex-stars]][redisgraph-ex-url]                       | [GitHub][redisgraph-ex-url] ||
| [redisgraph-rs][redisgraph-rs-url]                        | Rust       | MIT     | [malte-v][redisgraph-rs-author]             | [![redisgraph-rs-stars]][redisgraph-rs-url]                       | [GitHub][redisgraph-rs-url] ||
| [redis_graph][redis_graph-url]                            | Rust       | BSD     | [tompro][redis_graph-author]                | [![redis_graph-stars]][redis_graph-url]                           | [GitHub][redis_graph-url] ||
| [rustis][rustis-url]                                     | Rust       | MIT          | [Dahomey Technologies][rustis-author]       | [![rustis-stars]][rustis-url]    | [Crate](https://crates.io/crates/rustis) | [Documentation](https://docs.rs/rustis/latest/rustis/commands/trait.GraphCommands.html) |
| [NRedisGraph][NRedisGraph-url]                            | C#         | BSD     | [tombatron][NRedisGraph-author]             | [![NRedisGraph-stars]][NRedisGraph-url]                           | [GitHub][NRedisGraph-url] ||
| [RedisGraph.jl][RedisGraph.jl-url]                        | Julia      | MIT     | [xyxel][RedisGraph.jl-author]               | [![RedisGraph.jl-stars]][RedisGraph.jl-url]                       | [GitHub][RedisGraph.jl-url] ||

[redis-url]: https://redis.com
[falkordb-url]: https://www.falkordb.com

[falkordb-py-url]: http://github.com/falkorDB/falkordb-py
[falkordb-py-stars]: https://img.shields.io/github/stars/falkorDB/falkordb-py.svg?style=social&amp;label=Star&amp;maxAge=2592000
[falkordb-py-package]: https://pypi.org/project/FalkorDB

[jfalkordb-url]: https://github.com/falkordb/jfalkordb
[jfalkordb-stars]: https://img.shields.io/github/stars/falkordb/jfalkordb.svg?style=social&amp;label=Star&amp;maxAge=2592000
[jfalkordb-package]: https://search.maven.org/artifact/com.falkordb/jfalkordb

[nfalkordb-url]: https://github.com/falkordb/NFalkorDB
[nfalkordb-stars]: https://img.shields.io/github/stars/falkordb/nfalkordb.svg?style=social&label=Star&maxAge=2592000
[nfalkordb-package]: https://www.nuget.org/packages/NFalkorDB

[nredisstack-url]: https://github.com/redis/nredisstack
[nredisstack-stars]: https://img.shields.io/github/stars/redis/nredisstack.svg?style=social&amp;label=Star&amp;maxAge=2592000
[nredisstack-package]: https://www.nuget.org/packages/nredisstack/

[falkordb-ts-url]: https://github.com/falkordb/falkordb-ts
[falkordb-ts-stars]: https://img.shields.io/github/stars/falkordb/falkordb-ts.svg?style=social&amp;label=Star&amp;maxAge=2592000
[falkordb-ts-package]: https://www.npmjs.com/package/falkordb

[falkordb-rs-url]: https://github.com/falkordb/falkordb-rs
[falkordb-rs-stars]: https://img.shields.io/github/stars/falkordb/falkordb-rs.svg?style=social&amp;label=Star&amp;maxAge=2592000
[falkordb-rs-package]: https://crates.io/crates/falkordb

[falkordb-go-url]: https://github.com/falkordb/falkordb-go
[falkordb-go-stars]: https://img.shields.io/github/stars/falkordb/falkordb-go.svg?style=social&amp;label=Star&amp;maxAge=2592000
[falkordb-go-package]: https://github.com/falkordb/falkordb-go

[redisgraph-rb-author]: https://redislabs.com
[redisgraph-rb-url]: https://github.com/RedisGraph/redisgraph-rb
[redisgraph-rb-stars]: https://img.shields.io/github/stars/RedisGraph/redisgraph-rb.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redgraph-author]: https://github.com/pzac
[redgraph-url]: https://github.com/pzac/redgraph
[redgraph-stars]: https://img.shields.io/github/stars/pzac/redgraph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redisgraph-go-author]: https://redislabs.com
[redisgraph-go-url]: https://github.com/RedisGraph/redisgraph-go
[redisgraph-go-stars]: https://img.shields.io/github/stars/RedisGraph/redisgraph-go.svg?style=social&amp;label=Star&amp;maxAge=2592000

[rueidis-url]: https://github.com/rueian/rueidis
[rueidis-author]: https://github.com/rueian
[rueidis-stars]: https://img.shields.io/github/stars/rueian/rueidis.svg?style=social&amp;label=Star&amp;maxAge=2592000


[rgraph-author]: https://github.com/Sceat
[rgraph-url]: https://github.com/HydreIO/rgraph
[rgraph-stars]: https://img.shields.io/github/stars/HydreIO/rgraph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[ioredisgraph-author]: https://github.com/Jonahss
[ioredisgraph-url]: https://github.com/Jonahss/ioredisgraph
[ioredisgraph-stars]: https://img.shields.io/github/stars/Jonahss/ioredisgraph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[php-redis-graph-author]: https://github.com/kjdev
[php-redis-graph-url]: https://github.com/kjdev/php-redis-graph
[php-redis-graph-stars]: https://img.shields.io/github/stars/kjdev/php-redis-graph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redisgraph_php-author]: https://github.com/jpbourbon
[redisgraph_php-url]: https://github.com/jpbourbon/redisgraph_php
[redisgraph_php-stars]: https://img.shields.io/github/stars/jpbourbon/redisgraph_php.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redislabs-redisgraph-php-author]: https://github.com/mkorkmaz
[redislabs-redisgraph-php-url]: https://github.com/mkorkmaz/redislabs-redisgraph-php
[redislabs-redisgraph-php-stars]: https://img.shields.io/github/stars/mkorkmaz/redislabs-redisgraph-php.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redisgraph-ex-author]: https://github.com/crflynn
[redisgraph-ex-url]: https://github.com/crflynn/redisgraph-ex
[redisgraph-ex-stars]: https://img.shields.io/github/stars/crflynn/redisgraph-ex.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redisgraph-rs-author]: https://github.com/malte-v
[redisgraph-rs-url]: https://github.com/malte-v/redisgraph-rs
[redisgraph-rs-stars]: https://img.shields.io/github/stars/malte-v/redisgraph-rs.svg?style=social&amp;label=Star&amp;maxAge=2592000

[redis_graph-author]: https://github.com/tompro
[redis_graph-url]: https://github.com/tompro/redis_graph
[redis_graph-stars]: https://img.shields.io/github/stars/tompro/redis_graph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[NRedisGraph-author]: https://github.com/tombatron
[NRedisGraph-url]: https://github.com/tombatron/NRedisGraph
[NRedisGraph-stars]: https://img.shields.io/github/stars/tombatron/NRedisGraph.svg?style=social&amp;label=Star&amp;maxAge=2592000

[RedisGraphDotNet.Client-author]: https://github.com/Sgawrys
[RedisGraphDotNet.Client-url]: https://github.com/Sgawrys/RedisGraphDotNet.Client
[RedisGraphDotNet.Client-stars]: https://img.shields.io/github/stars/Sgawrys/RedisGraphDotNet.Client.svg?style=social&amp;label=Star&amp;maxAge=2592000

[RedisGraph.jl-author]: https://github.com/xyxel
[RedisGraph.jl-url]: https://github.com/xyxel/RedisGraph.jl
[RedisGraph.jl-stars]: https://img.shields.io/github/stars/xyxel/RedisGraph.jl.svg?style=social&amp;label=Star&amp;maxAge=2592000

[rustis-url]: https://github.com/dahomey-technologies/rustis
[rustis-author]: https://github.com/dahomey-technologies
[rustis-stars]: https://img.shields.io/github/stars/dahomey-technologies/rustis.svg?style=social&amp;label=Star&amp;maxAge=2592000

## LICENSE

Licensed under the Server Side Public License v1 (SSPLv1). See [LICENSE](LICENSE.txt).

### Support our work

⭐️ If you find this repository helpful, please consider giving it a star!

↗️ Graph, graph database, RAG, graphrag, Retrieval-Augmented Generation,Information Retrieval, Natural Language Processing, LLM, Embeddings, Semantic Search
