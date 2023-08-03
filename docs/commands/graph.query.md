Executes the given query against a specified graph.

Arguments: `Graph name, Query, Timeout [optional]`

Returns: [Result set](/redisgraph/design/result_structure)

### Queries and Parameterized Queries

The execution plans of queries, both regular and parameterized, are cached (up to [CACHE_SIZE](https://redis.io/docs/stack/graph/configuration/#cache_size) unique queries are cached). Therefore, it is recommended to use parametrized queries when executing many queries with the same pattern but different constants.

Query-level timeouts can be set as described in [the configuration section](/redisgraph/configuration#timeout).

#### Query structure: 

`GRAPH.QUERY graph_name "query"`

example:

```sh
GRAPH.QUERY us_government "MATCH (p:president)-[:born]->(:state {name:'Hawaii'}) RETURN p"
```

#### Parametrized query structure: 

`GRAPH.QUERY graph_name "CYPHER param=val [param=val ...] query"`

example:

```sh
GRAPH.QUERY us_government "CYPHER state_name='Hawaii' MATCH (p:president)-[:born]->(:state {name:$state_name}) RETURN p"
```

### Query language

The syntax is based on [Cypher](http://www.opencypher.org/). [Most](https://redis.io/docs/stack/graph/cypher_support/) of the language is supported. RedisGraph-specific extensions are also described below.

1. [Clauses](#query-structure)
2. [Functions](#functions)

### Query structure

* [MATCH](#match)
* [OPTIONAL MATCH](#optional-match)
* [WHERE](#where)
* [RETURN](#return)
* [ORDER BY](#order-by)
* [SKIP](#skip)
* [LIMIT](#limit)
* [CREATE](#create)
* [MERGE](#merge)
* [DELETE](#delete)
* [SET](#set)
* [WITH](#with)
* [UNION](#union)
* [UNWIND](#unwind)
* [FOREACH](#foreach)
* [CALL {}](/docs/commands/graph.query.md#call-)

#### MATCH

Match describes the relationship between queried entities, using ascii art to represent pattern(s) to match against.

Nodes are represented by parentheses `()`,
and Relationships are represented by brackets `[]`.

Each graph entity node/relationship can contain an alias and a label/relationship type, but both can be left empty if necessary.

Entity structure: `alias:label {filters}`.

Alias, label/relationship type, and filters are all optional.

Example:

```sh
(a:Actor)-[:ACT]->(m:Movie {title:"straight outta compton"})
```

`a` is an alias for the source node, which we'll be able to refer to at different places within our query.

`Actor` is the label under which this node is marked.

`ACT` is the relationship type.

`m` is an alias for the destination node.

`Movie` destination node is of "type" movie.

`{title:"straight outta compton"}` requires the node's title attribute to equal "straight outta compton".

In this example, we're interested in actor entities which have the relation "act" with **the** entity representing the
"straight outta compton" movie.

It is possible to describe broader relationships by composing a multi-hop query such as:

```sh
(me {name:'swilly'})-[:FRIENDS_WITH]->()-[:FRIENDS_WITH]->(foaf)
```

Here we're interested in finding out who my friends' friends are.

Nodes can have more than one relationship coming in or out of them, for instance:

```sh
(me {name:'swilly'})-[:VISITED]->(c:Country)<-[:VISITED]-(friend)<-[:FRIENDS_WITH]-(me)
```

Here we're interested in knowing which of my friends have visited at least one country I've been to.

##### Variable length relationships

Nodes that are a variable number of relationship→node hops away can be found using the following syntax:

```sh
-[:TYPE*minHops..maxHops]->
```

`TYPE`, `minHops` and `maxHops` are all optional and default to type agnostic, 1 and infinity, respectively.

When no bounds are given the dots may be omitted. The dots may also be omitted when setting only one bound and this implies a fixed length pattern.

Example:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (charlie:Actor { name: 'Charlie Sheen' })-[:PLAYED_WITH*1..3]->(colleague:Actor)
RETURN colleague"
```

Returns all actors related to 'Charlie Sheen' by 1 to 3 hops.

##### Bidirectional path traversal

If a relationship pattern does not specify a direction, it will match regardless of which node is the source and which is the destination:
```sh
-[:TYPE]-
```

Example:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (person_a:Person)-[:KNOWS]-(person_b:Person)
RETURN person_a, person_b"
```

Returns all pairs of people connected by a `KNOWS` relationship. Note that each pair will be returned twice, once with each node in the `person_a` field and once in the `person_b` field.

The syntactic sugar `(person_a)<-[:KNOWS]->(person_b)` will return the same results.

The bracketed edge description can be omitted if all relations should be considered: `(person_a)--(person_b)`.

##### Named paths

Named path variables are created by assigning a path in a MATCH clause to a single alias with the syntax:
`MATCH named_path = (path)-[to]->(capture)`

The named path includes all entities in the path, regardless of whether they have been explicitly aliased. Named paths can be accessed using [designated built-in functions](#path-functions) or returned directly if using a language-specific client.

Example:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH p=(charlie:Actor { name: 'Charlie Sheen' })-[:PLAYED_WITH*1..3]->(:Actor)
RETURN nodes(p) as actors"
```

This query will produce all the paths matching the pattern contained in the named path `p`. All of these paths will share the same starting point, the actor node representing Charlie Sheen, but will otherwise vary in length and contents. Though the variable-length traversal and `(:Actor)` endpoint are not explicitly aliased, all nodes and edges traversed along the path will be included in `p`. In this case, we are only interested in the nodes of each path, which we'll collect using the built-in function `nodes()`. The returned value will contain, in order, Charlie Sheen, between 0 and 2 intermediate nodes, and the unaliased endpoint.

##### All shortest paths

The `allShortestPaths` function returns all the shortest paths between a pair of entities.

`allShortestPaths()` is a MATCH mode in which only the shortest paths matching all criteria are captured. Both the source and the target nodes must be bound in an earlier WITH-demarcated scope to invoke `allShortestPaths()`.

A minimal length (must be 1) and maximal length (must be at least 1) for the search may be specified. Zero or more relationship types may be specified (e.g. [:R|Q*1..3]). No property filters may be introduced in the pattern.

`allShortestPaths()` can have any number of hops for its minimum and maximum, including zero. This number represents how many edges can be traversed in fulfilling the pattern, with a value of 0 entailing that the source node will be included in the returned path.

Filters on properties are supported, and any number of labels may be specified.

Example:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (charlie:Actor {name: 'Charlie Sheen'}), (kevin:Actor {name: 'Kevin Bacon'})
WITH charlie, kevin
MATCH p=allShortestPaths((charlie)-[:PLAYED_WITH*]->(kevin))
RETURN nodes(p) as actors"
```

This query will produce all paths of the minimum length connecting the actor node representing Charlie Sheen to the one representing Kevin Bacon. There are several 2-hop paths between the two actors, and all of these will be returned. The computation of paths then terminates, as we are not interested in any paths of length greater than 2.

##### Single-Pair minimal-weight bounded-cost bounded-length paths

(Since RedisGraph v2.10)

The `algo.SPpaths` procedure returns one, _n_, or all minimal-weight, [optionally] bounded-cost, [optionally] bounded-length distinct paths between a pair of entities. Each path is a sequence of distinct nodes connected by distinct edges.

`algo.SPpaths()` is a MATCH mode in which only the paths matching all criteria are captured. Both the source and the target nodes must be bound in an earlier WITH-demarcated scope to invoke `algo.SPpaths()`.

Input arguments:

- A map containing:
  - `sourceNode`: Mandatory. Must be of type node
  - `targetNode`: Mandatory. Must be of type node
  - `relTypes`: Optional. Array of zero or more relationship types. A relationship must have one of these types to be part of the path. If not specified or empty: the path may contain any relationship.
  - `relDirection`: Optional. string. one of `'incoming'`, `'outgoing'`, `'both'`. If not specified: `'outgoing'`.
  - `pathCount`: Optional. Number of minimal-weight paths to retrieve. Non-negative integer. If not specified: 1

    - `0`: retrieve all minimal-weight paths (all reported paths have the same weight)

      Order: 1st : minimal cost, 2nd: minimal length.

    - `1`: retrieve a single minimal-weight path

      When multiple equal-weight paths exist: (preferences: 1st : minimal cost, 2nd: minimal length)

    - _n_ > 1: retrieve up to _n_ minimal-weight paths (reported paths may have different weights)

      When multiple equal-weight paths exist: (preferences: 1st : minimal cost, 2nd: minimal length)

  - `weightProp`: Optional. If not specified: use the default weight: 1 for each relationship.

    The name of the property that represents the weight of each relationship (integer / float)

    If such property doesn’t exist, of if its value is not a positive numeric - use the default weight: 1

    Note: when all weights are equal: minimal-weight ≡ shortest-path.

  - `costProp`: Optional. If not specified: use the default cost: 1 for each relationship.

    The name of the property that represents the cost of each relationship (integer / float)

    If such property doesn't exist, or if its value is not a positive numeric - use the default cost: 1

  - `maxLen`: Optional. Maximal path length (number of relationships along the path). Positive integer. 

    If not specified: no maximal length constraint.

  - `maxCost`: Optional. Positive numeric. If not specified: no maximal cost constraint.

    The maximal cumulative cost for the relationships along the path.
    
Result:

  - Paths conforming to the input arguments. For each reported path:

    - `path` - the path

    - `pathWeight` - the path’s weight

    - `pathCost` - the path’s cost

    To retrieve additional information:

    - The path’s length can be retrieved with `length(path)`

    - An array of the nodes along the path can be retrieved with `nodes(path)`

    - The path’s first node can be retrieved with `nodes(path)[0]`

    - The path’s last node can be retrieved with `nodes(path)[-1]`

    - An array of the relationship's costs along the path can be retrieved with `[r in relationships(path) | r.cost]` where cost is the name of the cost property

    - An array of the relationship's weights along the path can be retrieved with `[r in relationships(path) | r.weight]` where weight is the name of the weight property

Behavior in presence on multiple-edges:

  - multi-edges are two or more edges connecting the same pair of vertices (possibly with different weights and costs). 

  - All matching edges are considered. Paths with identical vertices and different edges are different paths. The following are 3 different paths ('n1', 'n2', and 'n3' are nodes; 'e1', 'e2', 'e3', and 'e4' are edges): (n1)-[e1]-(n2)-[e2]-(n3),  (n1)-[e1]-(n2)-[e3]-(n3),  (n1)-[e4]-(n2)-[e3]-(n3)

Example:

```sh
GRAPH.QUERY DEMO_GRAPH 
"MATCH (s:Actor {name: 'Charlie Sheen'}), (t:Actor {name: 'Kevin Bacon'}) 
CALL algo.SPpaths( {sourceNode: s, targetNode: t, relTypes: ['r1', 'r2', 'r3'], relDirection: 'outgoing', pathCount: 1, weightProp: 'weight', costProp: 'cost', maxLen: 3, maxCost: 100} ) 
YIELD path, pathCost, pathWeight
RETURN path ORDER BY pathCost"
```

##### Single-Source minimal-weight bounded-cost bounded-length paths

(Since RedisGraph v2.10)

The `algo.SSpaths` procedure returns one, _n_, or all minimal-weight, [optionally] bounded-cost, [optionally] bounded-length distinct paths from a given entity. Each path is a sequence of distinct nodes connected by distinct edges.

`algo.SSpaths()` is a MATCH mode in which only the paths matching all criteria are captured. The source node must be bound in an earlier WITH-demarcated scope to invoke `algo.SSpaths()`.

Input arguments:

- A map containing:
  - `sourceNode`: Mandatory. Must be of type node
  - `relTypes`: Optional. Array of zero or more relationship types. A relationship must have one of these types to be part of the path. If not specified or empty: the path may contain any relationship.
  - `relDirection`: Optional. string. one of `'incoming'`, `'outgoing'`, `'both'`. If not specified: `'outgoing'`.
  - `pathCount`: Optional. Number of minimal-weight paths to retrieve. Non-negative integer. If not specified: 1

    This number is global (not per source-target pair); all returned paths may be with the same target.

    - `0`: retrieve all minimal-weight paths (all reported paths have the same weight)

      Order: 1st : minimal cost, 2nd: minimal length.

    - `1`: retrieve a single minimal-weight path

      When multiple equal-weight paths exist: (preferences: 1st : minimal cost, 2nd: minimal length)

    - _n_ > 1: retrieve up to _n_ minimal-weight paths (reported paths may have different weights)

      When multiple equal-weight paths exist: (preferences: 1st : minimal cost, 2nd: minimal length)

  - `weightProp`: Optional. If not specified: use the default weight: 1 for each relationship.

    The name of the property that represents the weight of each relationship (integer / float)

    If such property doesn’t exist, of if its value is not a positive numeric - use the default weight: 1

    Note: when all weights are equal: minimal-weight ≡ shortest-path.

  - `costProp`: Optional. If not specified: use the default cost: 1 for each relationship.

    The name of the property that represents the cost of each relationship (integer / float)

    If such property doesn't exist, or if its value is not a positive numeric - use the default cost: 1

  - `maxLen`: Optional. Maximal path length (number of relationships along the path). Positive integer. 

    If not specified: no maximal length constraint.

  - `maxCost`: Optional. Positive numeric. If not specified: no maximal cost constraint.

    The maximal cumulative cost for the relationships along the path.
    
Result:

  - Paths conforming to the input arguments. For each reported path:

    - `path` - the path

    - `pathWeight` - the path’s weight

    - `pathCost` - the path’s cost

    To retrieve additional information:

    - The path’s length can be retrieved with `length(path)`

    - An array of the nodes along the path can be retrieved with `nodes(path)`

    - The path’s first node can be retrieved with `nodes(path)[0]`

    - The path’s last node can be retrieved with `nodes(path)[-1]`

    - An array of the relationship's costs along the path can be retrieved with `[r in relationships(path) | r.cost]` where cost is the name of the cost property

    - An array of the relationship's weights along the path can be retrieved with `[r in relationships(path) | r.weight]` where weight is the name of the weight property

Behavior in presence on multiple-edges:

  - multi-edges are two or more edges connecting the same pair of vertices (possibly with different weights and costs). 

  - All matching edges are considered. Paths with identical vertices and different edges are different paths. The following are 3 different paths ('n1', 'n2', and 'n3' are nodes; 'e1', 'e2', 'e3', and 'e4' are edges): (n1)-[e1]-(n2)-[e2]-(n3), (n1)-[e1]-(n2)-[e3]-(n3), (n1)-[e4]-(n2)-[e3]-(n3)

Example:

```sh
GRAPH.QUERY DEMO_GRAPH 
"MATCH (s:Actor {name: 'Charlie Sheen'})
CALL algo.SSpaths( {sourceNode: s, relTypes: ['r1', 'r2', 'r3'], relDirection: 'outgoing', pathCount: 1, weightProp: 'weight', costProp: 'cost', maxLen: 3, maxCost: 100} ) 
YIELD path, pathCost, pathWeight
RETURN path ORDER BY pathCost"
```


#### OPTIONAL MATCH

The OPTIONAL MATCH clause is a MATCH variant that produces null values for elements that do not match successfully, rather than the all-or-nothing logic for patterns in MATCH clauses.

It can be considered to fill the same role as LEFT/RIGHT JOIN does in SQL, as MATCH entities must be resolved but nodes and edges introduced in OPTIONAL MATCH will be returned as nulls if they cannot be found.

OPTIONAL MATCH clauses accept the same patterns as standard MATCH clauses, and may similarly be modified by WHERE clauses.

Multiple MATCH and OPTIONAL MATCH clauses can be chained together, though a mandatory MATCH cannot follow an optional one.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (p:Person) OPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)
WHERE w.start_date > 2016
RETURN p, w, c"
```

All `Person` nodes are returned, as well as any `WORKS_AT` relations and `Company` nodes that can be resolved and satisfy the `start_date` constraint. For each `Person` that does not resolve the optional pattern, the person will be returned as normal and the non-matching elements will be returned as null.

Cypher is lenient in its handling of null values, so actions like property accesses and function calls on null values will return null values rather than emit errors.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (p:Person) OPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)
RETURN p, w.department, ID(c) as ID"
```

In this case, `w.department` and `ID` will be returned if the OPTIONAL MATCH was successful, and will be null otherwise.

Clauses like SET, CREATE, MERGE, and DELETE will ignore null inputs and perform the expected updates on real inputs. One exception to this is that attempting to create a relation with a null endpoint will cause an error:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (p:Person) OPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)
CREATE (c)-[:NEW_RELATION]->(:NEW_NODE)"
```

If `c` is null for any record, this query will emit an error. In this case, no changes to the graph are committed, even if some values for `c` were resolved.

#### WHERE

This clause is not mandatory, but if you want to filter results, you can specify your predicates here.

Supported operations:

- `=`
- `<>`
- `<`
- `<=`
- `>`
- `>=`
- `CONTAINS`
- `ENDS WITH`
- `IN`
- `STARTS WITH`

Predicates can be combined using AND / OR / NOT.

Be sure to wrap predicates within parentheses to control precedence.

Examples:

```
WHERE (actor.name = "john doe" OR movie.rating > 8.8) AND movie.votes <= 250)
```

```
WHERE actor.age >= director.age AND actor.age > 32
```

It is also possible to specify equality predicates within nodes using the curly braces as such:

```
(:President {name:"Jed Bartlett"})-[:WON]->(:State)
```

Here we've required that the president node's name will have the value "Jed Bartlett".

There's no difference between inline predicates and predicates specified within the WHERE clause.

It is also possible to filter on graph patterns. The following queries, which return all presidents and the states they won in, produce the same results:

```sh
MATCH (p:President), (s:State) WHERE (p)-[:WON]->(s) RETURN p, s
```

and

```sh
MATCH (p:President)-[:WON]->(s:State) RETURN p, s
```

Pattern predicates can be also negated and combined with the logical operators AND, OR, and NOT. The following query returns all the presidents that did not win in the states where they were governors:

```sh
MATCH (p:President), (s:State) WHERE NOT (p)-[:WON]->(s) AND (p)->[:governor]->(s) RETURN p, s
```

Nodes can also be filtered by label:

```sh
MATCH (n)-[:R]->() WHERE n:L1 OR n:L2 RETURN n 
```

When possible, it is preferable to specify the label in the node pattern of the MATCH clause.

#### RETURN

In its simple form, Return defines which properties the returned result-set will contain.

Its structure is a list of `alias.property` separated by commas.

For convenience, it's possible to specify the alias only when you're interested in every attribute an entity possesses,
and don't want to specify each attribute individually. For example:

```sh
RETURN movie.title, actor
```

Use the DISTINCT keyword to remove duplications within the result-set:

```sh
RETURN DISTINCT friend_of_friend.name
```

In the above example, suppose we have two friends, Joe and Miesha,
and both know Dominick.

DISTINCT will make sure Dominick will only appear once
in the final result set.


Return can also be used to aggregate data, similar to group by in SQL.

Once an aggregation function is added to the return
list, all other "none" aggregated values are considered as group keys, for example:

```sh
RETURN movie.title, MAX(actor.age), MIN(actor.age)
```

Here we group data by movie title and for each movie, and we find its youngest and oldest actor age.

#### Aggregations

Supported aggregation functions include:

- `avg`
- `collect`
- `count`
- `max`
- `min`
- `percentileCont`
- `percentileDisc`
- `stDev`
- `sum`

#### ORDER BY

Order by specifies that the output be sorted and how.

You can order by multiple properties by stating each variable in the ORDER BY clause.

Each property may specify its sort order with `ASC`/`ASCENDING` or `DESC`/`DESCENDING`. If no order is specified, it defaults to ascending.

The result will be sorted by the first variable listed.

For equal values, it will go to the next property in the ORDER BY clause, and so on.

```sh
ORDER BY <alias.property [ASC/DESC] list>
```

Below we sort our friends by height. For equal heights, weight is used to break ties.

```sh
ORDER BY friend.height, friend.weight DESC
```

#### SKIP

The optional skip clause allows a specified number of records to be omitted from the result set.

```sh
SKIP <number of records to skip>
```

This can be useful when processing results in batches. A query that would examine the second 100-element batch of nodes with the label `Person`, for example, would be:

```sh
GRAPH.QUERY DEMO_GRAPH "MATCH (p:Person) RETURN p ORDER BY p.name SKIP 100 LIMIT 100"
```

#### LIMIT

Although not mandatory, you can use the limit clause
to limit the number of records returned by a query:

```
LIMIT <max records to return>
```

If not specified, there's no limit to the number of records returned by a query.

#### CREATE

CREATE is used to introduce new nodes and relationships.

The simplest example of CREATE would be a single node creation:

```sh
CREATE (n)
```

It's possible to create multiple entities by separating them with a comma.

```sh
CREATE (n),(m)
```

```sh
CREATE (:Person {name: 'Kurt', age: 27})
```

To add relations between nodes, in the following example we first find an existing source node. After it's found, we create a new relationship and destination node.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (a:Person)
WHERE a.name = 'Kurt'
CREATE (a)-[:MEMBER]->(:Band {name:'Nirvana'})"
```

Here the source node is a bounded node, while the destination node is unbounded.

As a result, a new node is created representing the band Nirvana and a new relation connects Kurt to the band.

Lastly we create a complete pattern.

All entities within the pattern which are not bounded will be created.

```sh
GRAPH.QUERY DEMO_GRAPH
"CREATE (jim:Person{name:'Jim', age:29})-[:FRIENDS]->(pam:Person {name:'Pam', age:27})-[:WORKS]->(:Employer {name:'Dunder Mifflin'})"
```

This query will create three nodes and two relationships.

#### DELETE

DELETE is used to remove both nodes and relationships.

Note that deleting a node also deletes all of its incoming and outgoing relationships.

To delete a node and all of its relationships:

```sh
GRAPH.QUERY DEMO_GRAPH "MATCH (p:Person {name:'Jim'}) DELETE p"
```

To delete relationship:

```sh
GRAPH.QUERY DEMO_GRAPH "MATCH (:Person {name:'Jim'})-[r:FRIENDS]->() DELETE r"
```

This query will delete all `friend` outgoing relationships from the node with the name 'Jim'.

#### SET

SET is used to create or update properties on nodes and relationships.

To set a property on a node, use `SET`.

```sh
GRAPH.QUERY DEMO_GRAPH "MATCH (n { name: 'Jim' }) SET n.name = 'Bob'"
```

If you want to set multiple properties in one go, simply separate them with a comma to set multiple properties using a single SET clause.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (n { name: 'Jim', age:32 })
SET n.age = 33, n.name = 'Bob'"
```

The same can be accomplished by setting the graph entity variable to a map:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (n { name: 'Jim', age:32 })
SET n = {age: 33, name: 'Bob'}"
```

Using `=` in this way replaces all of the entity's previous properties, while `+=` will only set the properties it explicitly mentions.

In the same way, the full property set of a graph entity can be assigned or merged:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (jim {name: 'Jim'}), (pam {name: 'Pam'})
SET jim = pam"
```

After executing this query, the `jim` node will have the same property set as the `pam` node.

To remove a node's property, simply set property value to NULL.

```sh
GRAPH.QUERY DEMO_GRAPH "MATCH (n { name: 'Jim' }) SET n.name = NULL"
```

#### MERGE

The MERGE clause ensures that a path exists in the graph (either the path already exists, or it needs to be created).

MERGE either matches existing nodes and binds them, or it creates new data and binds that.

It’s like a combination of MATCH and CREATE that also allows you to specify what happens if the data was matched or created.

For example, you can specify that the graph must contain a node for a user with a certain name.

If there isn’t a node with the correct name, a new node will be created and its name property set.

Any aliases in the MERGE path that were introduced by earlier clauses can only be matched; MERGE will not create them.

When the MERGE path doesn't rely on earlier clauses, the whole path will always either be matched or created.

If all path elements are introduced by MERGE, a match failure will cause all elements to be created, even if part of the match succeeded.

The MERGE path can be followed by ON MATCH SET and ON CREATE SET directives to conditionally set properties depending on whether or not the match succeeded.

**Merging nodes**

To merge a single node with a label:

```sh
GRAPH.QUERY DEMO_GRAPH "MERGE (robert:Critic)"
```

To merge a single node with properties:

```sh
GRAPH.QUERY DEMO_GRAPH "MERGE (charlie { name: 'Charlie Sheen', age: 10 })"
```

To merge a single node, specifying both label and property:

```sh
GRAPH.QUERY DEMO_GRAPH "MERGE (michael:Person { name: 'Michael Douglas' })"
```

**Merging paths**

Because MERGE either matches or creates a full path, it is easy to accidentally create duplicate nodes.

For example, if we run the following query on our sample graph:

```sh
GRAPH.QUERY DEMO_GRAPH
"MERGE (charlie { name: 'Charlie Sheen '})-[r:ACTED_IN]->(wallStreet:Movie { name: 'Wall Street' })"
```

Even though a node with the name 'Charlie Sheen' already exists, the full pattern does not match, so 1 relation and 2 nodes - including a duplicate 'Charlie Sheen' node - will be created.

We should use multiple MERGE clauses to merge a relation and only create non-existent endpoints:

```sh
GRAPH.QUERY DEMO_GRAPH
"MERGE (charlie { name: 'Charlie Sheen' })
 MERGE (wallStreet:Movie { name: 'Wall Street' })
 MERGE (charlie)-[r:ACTED_IN]->(wallStreet)"
```

If we don't want to create anything if pattern elements don't exist, we can combine MATCH and MERGE clauses. The following query merges a relation only if both of its endpoints already exist:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (charlie { name: 'Charlie Sheen' })
 MATCH (wallStreet:Movie { name: 'Wall Street' })
 MERGE (charlie)-[r:ACTED_IN]->(wallStreet)"
```

**On Match and On Create directives**

Using ON MATCH and ON CREATE, MERGE can set properties differently depending on whether a pattern is matched or created.

In this query, we'll merge paths based on a list of properties and conditionally set a property when creating new entities:

```sh
GRAPH.QUERY DEMO_GRAPH
"UNWIND ['Charlie Sheen', 'Michael Douglas', 'Tamara Tunie'] AS actor_name
 MATCH (movie:Movie { name: 'Wall Street' })
 MERGE (person {name: actor_name})-[:ACTED_IN]->(movie)
 ON CREATE SET person.first_role = movie.name"
```

#### WITH
The WITH clause allows parts of queries to be independently executed and have their results handled uniquely.

This allows for more flexible query composition as well as data manipulations that would otherwise not be possible in a single query.

If, for example, we wanted to find all children in our graph who are above the average age of all people:
```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (p:Person) WITH AVG(p.age) AS average_age MATCH (:Person)-[:PARENT_OF]->(child:Person) WHERE child.age > average_age return child
```

This also allows us to use modifiers like `DISTINCT`, `SKIP`, `LIMIT`, and `ORDER` that otherwise require `RETURN` clauses.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (u:User)  WITH u AS nonrecent ORDER BY u.lastVisit LIMIT 3 SET nonrecent.should_contact = true"
```

#### UNION
The UNION clause is used to combine the result of multiple queries.

UNION combines the results of two or more queries into a single result set that includes all the rows that belong to all queries in the union.

The number and the names of the columns must be identical in all queries combined by using UNION.

To keep all the result rows, use UNION ALL.

Using just UNION will combine and remove duplicates from the result set.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (n:Actor) RETURN n.name AS name
UNION ALL
MATCH (n:Movie) RETURN n.title AS name"
```

#### UNWIND
The UNWIND clause breaks down a given list into a sequence of records; each contains a single element in the list.

The order of the records preserves the original list order.

```sh
GRAPH.QUERY DEMO_GRAPH
"CREATE (p {array:[1,2,3]})"
```

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (p) UNWIND p.array AS y RETURN y"
```

#### FOREACH

(Since RedisGraph v2.12)

The `FOREACH` clause feeds the components of a list to a sub-query comprised of **updating clauses only** (`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE` and `FOREACH`), while passing on the records it receives without change.

The clauses within the sub-query recognize the bound variables defined prior to the `FOREACH` clause, but are local in the sense that later clauses are not aware of the variables defined inside them. In other words, `FOREACH` uses the current context, and does not affect it.

The `FOREACH` clause can be used for numerous purposes, such as: Updating and creating graph entities in a concise manner, marking nodes\edges that satisfy some condition or are part of a path of interest and performing conditional queries.

We show examples of queries performing the above 3 use-cases.

The following query will create 5 nodes, each with property `v` with the values from 0 to 4 corresponding to the appropriate index in the list.

```sh
GRAPH.QUERY DEMO_GRAPH
"FOREACH(i in [1, 2, 3, 4] | CREATE (n:N {v: i}))"
```

The following query marks the nodes of all paths of length up to 15 km from a hotel in Toronto to a steakhouse with at least 2 Michelin stars.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH p = (hotel:HOTEL {City: 'Toronto'})-[r:ROAD*..5]->(rest:RESTAURANT {type: 'Steakhouse'}) WHERE sum(r.length) <= 15 AND hotel.stars >= 4 AND rest.Michelin_stars >= 2
FOREACH(n in nodes(p) | SET n.part_of_path = true)"
```

The following query searches for all the hotels, checks whether they buy directly from a bakery, and if not - makes sure they are marked as buying from a supplier that supplies bread, and that they do not buy directly from a bakery.

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (h:HOTEL) OPTIONAL MATCH (h)-[b:BUYS_FROM]->(bakery:BAKERY)
FOREACH(do_perform IN CASE WHEN b = NULL THEN [1] ELSE [] END | MERGE (h)-[b2:BUYS_FROM]->(s:SUPPLIER {supplies_bread: true}) SET b2.direct = false)"
```

#### CALL \{\}

(Since RedisGraph v2.12)

The CALL {} (subquery) clause allows local execution of subqueries, which opens the door for many comfortable and efficient actions on a graph.

The subquery is executed once for each record in the input stream.

The subquery may be a returning or non-returning subquery. A returning subquery may change the amount of records, while a non-returning subquery will not.

The variables in the scope before the CALL {} clause are available after the clause, together with the variables returned by the subquery (in the case of a returning subquery).

Variables may be imported from the outer scope **only** in an opening `WITH` clause, via simple projections (e.g. `WITH n, m`), or via `WITH *` (which imports all bound variables). The variables returned from a subquery may not override existing variables in the outer scope.

The CALL {} clause may be used for numerous purposes, such as: Post-`UNION` processing, local environment for aggregations and actions on every input row, efficient operations using a limited namespace (via imports) and performing side-effects using non-returning subqueries. Let's see some examples.

* Post-`UNION` processing.

We can easily get the cheapest and most expensive items in a store and set their `of_interest` property to `true` (to keep monitoring the 'interesting' items) using post-`UNION` processing:
  
  ```sh
  GRAPH.QUERY DEMO_GRAPH
  CALL {
    MATCH (s:Store {name: 'Walmart'})-[:SELLS]->(i:Item)
    RETURN i AS item
    ORDER BY price ASC
    LIMIT 1
    UNION
    MATCH (s:Store {name: 'Walmart'})-[:SELLS]->(i:Item)
    RETURN i AS item
    ORDER BY price DESC
    LIMIT 1
  }
  SET item.of_interest = true
  RETURN item.name AS name, item.price AS price
  ```

We can utilize post-`UNION` processing to perform aggregations over differently-matched entities. For example, we can count the number of customers and vendors that a store interacts with:

  ```sh
  GRAPH.QUERY DEMO_GRAPH
  CALL {
    MATCH (s:Store {name: 'Walmart'})-[:SELLS_TO]->(c:Customer)
    RETURN c AS interface
    UNION
    MATCH (s:Store {name: 'Walmart'})-[:BUYS_FROM]->(v:Vendor)
    RETURN v AS interface
  }
  RETURN count(interface) AS interfaces
  ```

* Local environment for aggregations and actions on every input row.

Another key feature of the CALL {} clause is the ability to perform isolated aggregations on every input row. For example, let's check if there is any correlation between the amount of sales per-product and the advertisement-intensity implemented for it in a particular month.

  ```sh
  GRAPH.QUERY DEMO_GRAPH
  MATCH (item:Item)
  CALL {
    WITH item
    MATCH (item)-[s:SOLD_TO {advertisement_intensity: 10}]->(c:Customer)
    WHERE s.date > '01-01-2023' AND s.date < '01-02-2023'
    RETURN count(s) AS item_sales_ads_high
  }
  CALL {
    WITH item
    MATCH (item)-[s:SOLD_TO {advertisement_intensity: 5}]->(c:Customer)
    WHERE s.date > '01-01-2023' AND s.date < '01-02-2023'
    RETURN count(s) AS item_sales_ads_low
  }
  RETURN item.name AS name, item_sales_ads_high as high_ads_sales, item_sales_ads_low as low_ads_sales
  ```

<!-- * Observe changes from previous executions (on previous records).

We can form useful structures and connections like linked-lists via the CALL {} clause. Let's form a linked-list of all items in a store, from the cheapest to the priciest:

```sh
MATCH (i:Item)
WITH i order BY i.price ASC LIMIT 1
SET i:HEAD
WITH i
MATCH (next_item:Item) WHERE NOT next_item:HEAD
WITH next_item ORDER BY next_item.price ASC
CALL {
  WITH next_item
  MATCH (curr_head:HEAD)
  REMOVE curr_head:HEAD
  SET next_item:HEAD
  CREATE (curr_head)-[:IS_CHEAPER_THAN]->(next_item)
}
``` -->

<!-- This will be added to a "performance-enhancement" section in the near future.

* Efficient operations using a limited namespace (via imports).

Given a query holding a respectively large namespace (a lot of bound variables), we can execute a subquery on a sub-namespace, and by thus enhance performance significantly. Let's look at an example.

Without a CALL {} clause:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (n:N), (m:M), (x:X), (y:Y), (z:Z), (e:E), (q:Q)
MATCH (temp:TEMP)
SET temp.v = n.v
RETURN n, m, x, y, z, e, q"
```
Runtime: 256 ms.

With a CALL {} clause:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (n:N), (m:M), (x:X), (y:Y), (z:Z), (e:E), (q:Q)
CALL {
  WITH n
  MATCH (temp:TEMP)
  SET temp.v = n.v
}
RETURN n, m, x, y, z, e, q"
```
Runtime: 99 ms. -->


* Side-effects.

We can comfortably perform side-effects using non-returning subqueries. For example, we can mark a sub-group of nodes in the graph withholding some shared property. Let's mark all the items in a Walmart store that were sold more than 100 times as popular items, and return **all** items in the store:

  ```sh
  GRAPH.QUERY DEMO_GRAPH
  MATCH (item:Item)
  CALL {
    WITH item
    MATCH (item)-[s:SOLD_TO]->(c:Customer)
    WITH item, count(s) AS item_sales
    WHERE item_sales > 100
    SET item.popular = true
  }
  RETURN item
  ```

### Functions

This section contains information on all supported functions from the Cypher query language.

* [Predicate functions](#predicate-functions)
* [Scalar functions](#scalar-functions)
* [Aggregating functions](#aggregating-functions)
* [List functions](#list-functions)
* [Mathematical operators](#mathematical-operators)
* [Mathematical functions](#mathematical-functions)
* [Trigonometric functions](#trigonometric-functions)
* [String functions](#string-functions)
* [Point functions](#point-functions)
* [Type conversion functions](#type-conversion-functions)
* [Node functions](#node-functions)
* [Path functions](#path-functions)

## Predicate functions

| Function                                                                          | Description|
| --------------------------------------------------------------------------------- | :----------|
| [all(_var_ IN _list_ WHERE _predicate_)](#existential-comprehension-functions)    | Returns true when _predicate_ holds true for all elements in _list_         |
| [any(_var_ IN _list_ WHERE _predicate_)](#existential-comprehension-functions)    | Returns true when _predicate_ holds true for at least one element in _list_ |
| exists(_pattern_)                                                                 | Returns true when at least one match for _pattern_ exists                   |
| isEmpty(_list_&#124;_map_&#124;_string_)                                          | Returns true if the input list or map contains no elements or if the input string contains no characters <br> Returns null when the input evaluates to null |
| [none(_var_ IN _list_ WHERE _predicate_)](#existential-comprehension-functions)   | Returns true when _predicate_ holds false for all elements in _list_        |
| [single(_var_ IN _list_ WHERE _predicate_)](#existential-comprehension-functions) | Returns true when _predicate_ holds true for exactly one element in _list_  |

## Scalar functions

| Function                          | Description|
| --------------------------------- | :----------|
| coalesce(_expr_[, expr...])       | Returns the evaluation of the first argument that evaluates to a non-null value <br> Returns null when all arguments evaluate to null       |
| endNode(_relationship_)           | Returns the destination node of a relationship <br> Returns null when _relationship_ evaluates to null                                      |
| hasLabels(_node_, _labelsList_) * | Returns true when _node_ contains all labels in _labelsList_, otherwise false <br> Return true when _labelsList_ evaluates to an empty list |
| id(_node_&#124;_relationship_)    | Returns the internal ID of a node or relationship (which is not immutable)                                                                  |
| labels(_node_)                    | Returns a list of strings: all labels of _node_ <br> Returns null when _node_ evaluates to null                                             |
| properties(_expr_)                | When _expr_ is a node or relationship: Returns a map containing all the properties of the given node or relationship <br> When _expr_ evaluates to a map: Returns _expr_ unchanged <br> Returns null when _expr_ evaluates to null |
| randomUUID()                      | Returns a random UUID (Universal Unique IDentifier)                                                                                         |
| startNode(_relationship_)         | Returns the source node of a relationship <br> Returns null when _relationship_ evaluates to null                                           |
| timestamp()                       | Returns the current system timestamp (milliseconds since epoch)                                                                             |
| type(_relationship_)              | Returns a string: the type of _relationship_ <br> Returns null when _relationship_ evaluates to null                                        |
| typeOf(_expr_) *                  | (Since RedisGraph v2.12) <br> Returns a string: the type of a literal, an expression's evaluation, an alias, a node's property, or a relationship's property <br> Return value is one of `Map`, `String`, `Integer`, `Boolean`, `Float`, `Node`, `Edge`, `List`, `Path`, `Point`, or `Null` |

&#42; RedisGraph-specific extensions to Cypher

## Aggregating functions

|Function                             | Description|
| ----------------------------------- |:-----------|
|avg(_expr_)                          | Returns the average of a set of numeric values. null values are ignored <br> Returns null when _expr_ has no evaluations                                                   |
|collect(_expr_)                      | Returns a list containing all non-null elements which evaluated from a given expression                                                                                   |
|count(_expr_&#124;&#42;)             | When argument is _expr_: returns the number of non-null evaluations of _expr_ <br> When argument is `*`: returns the total number of evaluations (including nulls)     |
|max(_expr_)                          | Returns the maximum value in a set of values (taking into account type ordering). null values are ignored <br> Returns null when _expr_ has no evaluations                |
|min(_expr_)                          | Returns the minimum value in a set of values (taking into account type ordering). null values are ignored <br> Returns null when _expr_ has no evaluations                |
|percentileCont(_expr_, _percentile_) | Returns a linear-interpolated percentile (between 0.0 and 1.0) over a set of numeric values. null values are ignored <br> Returns null when _expr_ has no evaluations    |
|percentileDisc(_expr_, _percentile_) | Returns a nearest-value percentile (between 0.0 and 1.0) over a set of numeric values. null values are ignored <br> Returns null when _expr_ has no evaluations         |
|stDev(_expr_)                        | Returns the sample standard deviation over a set of numeric values. null values are ignored <br> Returns null when _expr_ has no evaluations                       |
|stDevP(_expr_)                       | Returns the population standard deviation over a set of numeric values. null values are ignored <br> Returns null when _expr_ has no evaluations                       |
|sum(_expr_)                          | Returns the sum of a set of numeric values. null values are ignored <br> Returns 0 when _expr_ has no evaluations                                                         |

## List functions

| Function                             | Description|
| ------------------------------------ | :----------|
| head(_expr_)                         | Returns the first element of a list <br> Returns null when _expr_ evaluates to null or an empty list                                                                                                    |
| keys(_expr_)                         | Returns a list of strings: all key names for given map or all property names for a given node or edge <br> Returns null when _expr_ evaluates to null                                                |
| last(_expr_)                         | Returns the last element of a list <br> Returns null when _expr_ evaluates to null or an empty list                                     
| list.dedup(_list_) *                                               | (Since RedisGraph v2.12) <br> Given a list, returns a similar list after removing duplicate elements <br> Order is preserved, duplicates are removed from the end of the list <br> Returns null when _list_ evaluates to null <br> Emit an error when _list_ does not evaluate to a list or to null |
| list.insert(_list_, _idx_, _val_[, _dups_ = TRUE]) *               | (Since RedisGraph v2.12) <br> Given a list, returns a list after inserting a given value at a given index <br> _idx_ is 0-based when non-negative, or from the end of the list when negative <br> Returns null when _list_ evaluates to null <br> Returns _list_ when _val_ evaluates to null <br> Returns _list_ when _idx_ evaluates to an integer not in [-NumItems-1 .. NumItems] <br> When _dups_ evaluats to FALSE: returns _list_ when _val_ evaluates to a value that is already an element of _list_  <br> Emit an error when _list_ does not evaluate to a list or to null <br> Emit an error when _idx_ does not evaluate to an integer <br> Emit an error when _dups_, if specified, does not evaluate to a Boolean |
| list.insertListElements(_list_, _list2_, _idx_[, _dups_ = TRUE]) * | (Since RedisGraph v2.12) <br> Given a list, returns a list after inserting the elements of a second list at a given index <br> _idx_ is 0-based when non-negative, or from the end of the list when negative <br> Returns null when _list_ evaluates to null <br> Returns _list_ when _list2_ evaluates to null <br> Returns _list_ when _idx_ evaluates to an integer not in [-NumItems-1 .. NumItems] <br> When _dups_ evaluates to FALSE: If an element of _list2_ evaluates to an element of _list_ it would be skipped; If multiple elements of _list2_ evaluate to the same value - this value would be inserted at most once to _list_ <br> Emit an error when _list_ does not evaluate to a list or to null <br> Emit an error when _list2_ does not evaluate to a list or to null <br> Emit an error when _idx_ does not evaluate to an integer <br> Emit an error when _dups_, if specified, does not evaluate to a Boolean |
| list.remove(_list_, _idx_[, _count_ = 1]) *                        | (Since RedisGraph v2.12) <br> Given a list, returns a list after removing a given number of consecutive elements (or less, if the end of the list has been reached). starting at a given index. <br> _idx_ is 0-based when non-negative, or from the end of the list when negative <br> Returns _null_ when _list_ evaluates to null <br> Returns _list_ when _idx_ evaluates to an integer not in [-NumItems .. NumItems-1] <br> Returns _list_ when _count_ evaluates to a non-positive integer <br> Emit an error when _list_ does not evaluate to a list or to null <br> Emit an error when _idx_ does not evaluate to an integer <br> Emit an error when _count_, if specified, does not evaluate to an integer |
| list.sort(_list_[, _ascending_ = TRUE]) *                          | (Since RedisGraph v2.12) <br> Given a list, returns a list with similar elements, but sorted (inversely-sorted if _ascending_ is evaluated to FALSE) <br> Returns null when _list_ evaluates to null <br> Emit an error when _list_ does not evaluate to a list or to null <br> Emit an error when _ascending_, if specified, does not evaluate to a Boolean |
| range(_first_, _last_[, _step_ = 1]) | Returns a list of integers in the range of [start, end]. _step_, an optional integer argument, is the increment between consequtive elements                                                         |
| size(_expr_)                         | Returns the number of elements in a list <br> Returns null with _expr_ evaluates to null                                                                                                                |
| tail(_expr_)                         | Returns a sublist of a list, which contains all its elements except the first <br> Returns an empty list when _expr_ containst less than 2 elements. <br> Returns null when _expr_ evaluates to null |
| [reduce(...)](#reduce)               | Returns a scalar produced by evaluating an expression against each list member |

&#42; RedisGraph-specific extensions to Cypher

## Mathematical operators

|Function     | Description|
| ----------- |:-----------|
| +           | Add two values                                           |
| -           | Subtract second value from first                         |
| *           | Multiply two values                                      |
| /           | Divide first value by the second                         |
| ^           | Raise the first value to the power of the second         |
| %           | Perform modulo division of the first value by the second |

## Mathematical functions

|Function                   | Description|
| ------------------------- |:-----------|
| abs(_expr_)               | Returns the absolute value of a numeric value <br> Returns null when _expr_ evaluates to null |
| ceil(_expr_) **           | When _expr_ evaluates to an integer: returns its evaluation <br> When _expr_ evaluates to floating point: returns a floating point equals to the smallest integer greater than or equal to _expr_ <br> Returns null when _expr_ evaluates to null |
| e()                       | Returns the constant _e_, the base of the natural logarithm |
| exp(_expr_)               | Returns _e_^_expr_, where _e_ is the base of the natural logarithm <br> Returns null when _expr_ evaluates to null |
| floor(_expr_) **          | When _expr_ evaluates to an integer: returns its evaluation <br> When _expr_ evaluates to a floating point: returns a floating point equals to the greatest integer less than or equal to _expr_ <br> Returns null when _expr_ evaluates to null |
| log(_expr_)               | Returns the natural logarithm of a numeric value <br> Returns nan when _expr_ evaluates to a negative numeric value, -inf when _expr_ evaluates to 0, and null when _expr_ evaluates to null |
| log10(_expr_)             | Returns the base-10 logarithm of a numeric value <br> Returns nan when _expr_ evaluates to a negative numeric value, -inf when _expr_ evaluates to 0, and null when _expr_ evaluates to null |
| pow(_base_, _exponent_) * | Returns _base_ raised to the power of _exponent_ (equivalent to _base_^_exponent_) <br> Returns null when either evaluates to null |
| rand()                    | Returns a random floating point in the range [0,1] |
| round(_expr_) ** ***      | When _expr_ evaluates to an integer: returns its evaluation <br> When _expr_ evaluates to a floating point: returns a floating point equals to the integer closest to _expr_ <br> Returns null when _expr_ evaluates to null |
| sign(_expr_)              | Returns the signum of a numeric value: 0 when _expr_ evaluates to 0, -1 when _expr_ evaluates to a negative numeric value, and 1 when _expr_ evaluates to a positive numeric value <br> Returns null when _expr_ evaluates to null |
| sqrt(_expr_)              | Returns the square root of a numeric value <br> Returns nan when _expr_ evaluates to a negative value and null when _expr_ evaluates to null |

&#42; RedisGraph-specific extensions to Cypher

&#42;&#42; RedisGraph-specific behavior: to avoid possible loss of precision, when _expr_ evaluates to an integer - the result is an integer as well

&#42;&#42;&#42; RedisGraph-specific behavior: tie-breaking method is "half away from zero"

## Trigonometric functions

|Function               | Description|
| --------------------- |:-----------|
| acos(_expr_)          | Returns the arccosine, in radians, of a numeric value <br> Returns nan when _expr_ evaluates to a numeric value not in [-1, 1] and null when _expr_ evaluates to null |
| asin(_expr_)          | Returns the arcsine, in radians, of a numeric value <br> Returns nan when _expr_ evaluates to a numeric value not in [-1, 1] and null when _expr_ evaluates to null |
| atan(_expr_)          | Returns the arctangent, in radians, of a numeric value <br> Returns null when _expr_ evaluates to null                          |
| atan2(_expr_, _expr_) | Returns the 2-argument arctangent, in radians, of a pair of numeric values (Cartesian coordinates) <br> Returns 0 when both expressions evaluate to 0 <br> Returns null when either expression evaluates to null |
| cos(_expr_)           | Returns the cosine of a numeric value that represents an angle in radians <br> Returns null when _expr_ evaluates to null       |
| cot(_expr_)           | Returns the cotangent of a numeric value that represents an angle in radians <br> Returns inf when _expr_ evaluates to 0 and null when _expr_ evaluates to null |
| degrees(_expr_)       | Converts a numeric value from radians to degrees <br> Returns null when _expr_ evaluates to null                                |
| haversin(_expr_)      | Returns half the versine of a numeric value that represents an angle in radians <br> Returns null when _expr_ evaluates to null |
| pi()                  | Returns the mathematical constant _pi_                                                                                          |
| radians(_expr_)       | Converts a numeric value from degrees to radians <br> Returns null when _expr_ evaluates to null                                |
| sin(_expr_)           | Returns the sine of a numeric value that represents an angle in radians <br> Returns null when _expr_ evaluates to null         |
| tan(_expr_)           | Returns the tangent of a numeric value that represents an angle in radians <br> Returns null when _expr_ evaluates to null      |

## String functions

| Function                            | Description|
| ----------------------------------- | :----------|
| left(_str_, _len_)                  | Returns a string containing the _len_ leftmost characters of _str_ <br> Returns null when _str_ evaluates to null, otherwise emit an error if _len_ evaluates to null |
| lTrim(_str_)                        | Returns _str_ with leading whitespace removed <br> Returns null when _str_ evaluates to null                                                                     |
| replace(_str_, _search_, _replace_) | Returns _str_ with all occurrences of _search_ replaced with _replace_ <br> Returns null when any argument evaluates to null                                     |
| reverse(_str_)                      | Returns a string in which the order of all characters in _str_ are reversed <br> Returns null when _str_ evaluates to null                                       |
| right(_str_, _len_)                 | Returns a string containing the _len_ rightmost characters of _str_ <br> Returns null when _str_ evaluates to null, otherwise emit an error if _len_ evaluates to null |
| rTrim(_str_)                        | Returns _str_ with trailing whitespace removed <br> Returns null when _str_ evaluates to null                                                                   |
| split(_str_, _delimiter_)           | Returns a list of strings from splitting _str_ by _delimiter_ <br> Returns null when any argument evaluates to null                                             |
| string.join(_strList_[, _delimiter_ = '']) *         | (Since RedisGraph v2.12) <br> Returns a concatenation of a list of strings using a given delimiter <br> Returns null when _strList_ evaluates to null <br> Returns null when _delimiter_, if specified, evaluates to null <br> Emit an error when _strList_ does not evaluate to a list or to null <br> Emit an error when an element of _strList_ does not evaluate to a string <br> Emit an error when _delimiter_, if specified, does not evaluate to a string or to null |
| string.matchRegEx(_str_, _regex_) *                  | (Since RedisGraph v2.12) <br> Given a string and a regular expression, returns a list of all matches and matching regions <br> Returns an empty list when _str_ evaluates to null <br> Returns an empty list when _regex_ evaluates to null <br> Emit an error when _str_ does not evaluate to a string or to null <br> Emit an error when _regex_ does not evaluate to a valid regex string or to null |
| string.replaceRegEx(_str_, _regex_, _replacement_) * | (Since RedisGraph v2.12) <br> Given a string and a regular expression, returns a string after replacing each regex match with a given replacement <br> Returns null when _str_ evaluates to null <br> Returns null when _regex_ evaluates to null <br> Returns null when _replacement_ evaluates to null <br> Emit an error when _str_ does not evaluate to a string or to null <br> Emit an error when _regex_ does not evaluate to a valid regex string or to null <br> Emit an error when _replacement_ does not evaluate to a string or to null |
| substring(_str_, _start_[, _len_])  | When _len_ is specified: returns a substring of _str_ beginning with a 0-based index _start_ and with length _len_ <br> When _len_ is not specified: returns a substring of _str_ beginning with a 0-based index _start_ and extending to the end of _str_ <br> Returns null when _str_ evaluates to null <br> Emit an error when _start_ or _len_ evaluate to null |
| toLower(_str_)                      | Returns _str_ in lowercase <br> Returns null when _str_ evaluates to null                                 |
| toJSON(_expr_) *                    | Returns a [JSON representation](#json-format) of a value <br> Returns null when _expr_ evaluates to null  |
| toUpper(_str_)                      | Returns _str_ in uppercase <br> Returns null when _str_ evaluates to null                                 |
| trim(_str_)                         | Returns _str_ with leading and trailing whitespace removed <br> Returns null when _str_ evaluates to null |
| size(_str_)                         | Returns the number of characters in _str_ <br> Returns null when _str_ evaluates to null                  |

&#42; RedisGraph-specific extensions to Cypher

## Point functions

| Function                     | Description|
| ---------------------------- | :----------|
| [point(_map_)](#point)       | Returns a Point representing a lat/lon coordinates                                                          |
| distance(_point1_, _point2_) | Returns the distance in meters between the two given points <br> Returns null when either evaluates to null |

## Type conversion functions

|Function                     | Description|
| --------------------------- |:-----------|
| toBoolean(_expr_)           | Returns a Boolean when _expr_ evaluates to a Boolean <br> Converts a string to Boolean (`"true"` (case insensitive) to true, `"false"` (case insensitive) to false, any other value to null) <br> Converts an integer to Boolean (0 to `false`, any other values to `true`) <br> Returns null when _expr_ evaluates to null <br> Emit an error on other types |
| toBooleanList(_exprList_)   | Converts a list to a list of Booleans. Each element in the list is converted using toBooleanOrNull() |
| toBooleanOrNull(_expr_)     | Returns a Boolean when _expr_ evaluates to a Boolean <br> Converts a string to Boolean (`"true"` (case insensitive) to true, `"false"` (case insensitive) to false, any other value to null) <br> Converts an integer to Boolean (0 to `false`, any other values to `true`) <br> Returns null when _expr_ evaluates to null <br> Returns null for other types |
| toFloat(_expr_)             | Returns a floating point when _expr_ evaluates to a floating point <br> Converts an integer to a floating point <br> Converts a string to a floating point or null <br> Returns null when _expr_ evaluates to null <br> Emit an error on other types |
| toFloatList(_exprList_)     | Converts a list to a list of floating points. Each element in the list is converted using toFloatOrNull() |
| toFloatOrNull(_expr_)       | Returns a floating point when _expr_ evaluates to a floating point <br> Converts an integer to a floating point <br> Converts a string to a floating point or null <br> Returns null when _expr_ evaluates to null <br> Returns null for other types |
| toInteger(_expr_) *         | Returns an integer when _expr_ evaluates to an integer <br> Converts a floating point to integer <br> Converts a string to an integer or null <br> Converts a Boolean to an integer (false to 0, true to 1) (Since RedisGraph v2.10.8) <br> Returns null when _expr_ evaluates to null <br> Emit an error on other types |
| toIntegerList(_exprList_) * | Converts a list to a list of integer values. Each element in the list is converted using toIntegerOrNull() |
| toIntegerOrNull(_expr_) *   | Returns an integer when _expr_ evaluates to an integer <br> Converts a floating point to integer <br> Converts a string to an integer or null <br> Converts a Boolean to an integer (false to 0, true to 1) (Since RedisGraph v2.10.8) <br> Returns null when _expr_ evaluates to null <br> Returns null for other types |
| toString(_expr_)            | Returns a string when _expr_ evaluates to a string <br> Converts an integer, float, Boolean, string, or point to a string representation <br> Returns null when _expr_ evaluates to null <br> Emit an error on other types |
| toStringList(_exprList_)    | Converts a list to a list of strings. Each element in the list is converted using toStringOrNull() | 
| toStringOrNull(_expr_)      | Returns a string when _expr_ evaluates to a string <br> Converts an integer, float, Boolean, string, or point to a string representation <br> Returns null when _expr_ evaluates to null <br> Returns null for other types |

&#42; RedisGraph-specific behavior: rounding method when converting a floating point to an integer is "toward negative infinity (floor)"

## Node functions

|Function      | Description|
| ------------ |:-----------|
|indegree(_node_ [, _reltype_ ...]) * <br> indegree(_node_ [, _reltypeList_]) *   | When no relationship types are specified: Returns the number of _node_'s incoming edges <br> When one or more relationship types are specified: Returns the number of _node's_ incoming edges with one of the given relationship types <br> Return null when _node_ evaluates to null <br> the _reltypeList_ syntax is supported since RedisGraph v2.10.8 |
|outdegree(_node_ [, _reltype_ ...]) * <br> outdegree(_node_ [, _reltypeList_]) * | When no relationship types are specified: Returns the number of _node_'s outgoing edges <br> When one or more relationship types are specified: Returns the number of _node's_ outgoing edges with one of the given relationship types <br> Return null when _node_ evaluates to null <br> the _reltypeList_ syntax is supported since RedisGraph v2.10.8 |

&#42; RedisGraph-specific extensions to Cypher

## Path functions

| Function                             | Description|
| ------------------------------------ | :----------|
| nodes(_path_)                        | Returns a list containing all the nodes in _path_ <br> Returns null if _path_ evaluates to null         |
| relationships(_path_)                | Returns a list containing all the relationships in _path_ <br> Returns null if _path_ evaluates to null |
| length(_path_)                       | Return the length (number of edges) of _path_ <br> Returns null if _path_ evaluates to null             |
| [shortestPath(...)](#shortestPath) * | Return the shortest path that resolves the given pattern                                                |

&#42; RedisGraph-specific extensions to Cypher

### List comprehensions
List comprehensions are a syntactical construct that accepts an array and produces another based on the provided map and filter directives.

They are a common construct in functional languages and modern high-level languages. In Cypher, they use the syntax:

```sh
[element IN array WHERE condition | output elem]
```

- `array` can be any expression that produces an array: a literal, a property reference, or a function call.
- `WHERE condition` is an optional argument to only project elements that pass a certain criteria. If omitted, all elements in the array will be represented in the output.
- `| output elem` is an optional argument that allows elements to be transformed in the output array. If omitted, the output elements will be the same as their corresponding inputs.


The following query collects all paths of any length, then for each produces an array containing the `name` property of every node with a `rank` property greater than 10:

```sh
MATCH p=()-[*]->() RETURN [node IN nodes(p) WHERE node.rank > 10 | node.name]
```

#### Existential comprehension functions
The functions `any()`, `all()`, `single()` and `none()` use a simplified form of the list comprehension syntax and return a boolean value.

```sh
any(element IN array WHERE condition)
```

They can operate on any form of input array, but are particularly useful for path filtering. The following query collects all paths of any length in which all traversed edges have a weight less than 3:

```sh
MATCH p=()-[*]->() WHERE all(edge IN relationships(p) WHERE edge.weight < 3) RETURN p
```

### Pattern comprehensions

Pattern comprehensions are a method of producing a list composed of values found by performing the traversal of a given graph pattern.

The following query returns the name of a `Person` node and a list of all their friends' ages:

```sh
MATCH (n:Person)
RETURN
n.name,
[(n)-[:FRIEND_OF]->(f:Person) | f.age]
```

Optionally, a `WHERE` clause may be embedded in the pattern comprehension to filter results. In this query, all friends' ages will be gathered for friendships that started before 2010:

```sh
MATCH (n:Person)
RETURN
n.name,
[(n)-[e:FRIEND_OF]->(f:Person) WHERE e.since < 2010 | f.age]
```

### CASE WHEN

The case statement comes in two variants. Both accept an input argument and evaluates it against one or more expressions. The first `WHEN` argument that specifies a value matching the result will be accepted, and the value specified by the corresponding `THEN` keyword will be returned.

Optionally, an `ELSE` argument may also be specified to indicate what to do if none of the `WHEN` arguments match successfully.

In its simple form, there is only one expression to evaluate and it immediately follows the `CASE` keyword:

```sh
MATCH (n)
RETURN
CASE n.title
WHEN 'Engineer' THEN 100
WHEN 'Scientist' THEN 80
ELSE n.privileges
END
```

In its generic form, no expression follows the `CASE` keyword. Instead, each `WHEN` statement specifies its own expression:

```sh
MATCH (n)
RETURN
CASE
WHEN n.age < 18 THEN '0-18'
WHEN n.age < 30 THEN '18-30'
ELSE '30+'
END
```

#### Reduce
The `reduce()` function accepts a starting value and updates it by evaluating an expression against each element of the list:

```sh
RETURN reduce(sum = 0, n IN [1,2,3] | sum + n)
```

`sum` will successively have the values 0, 1, 3, and 6, with 6 being the output of the function call.

### Point
The `point()` function expects one map argument of the form:
```sh
RETURN point({latitude: lat_value, longitude: lon_val})
```

The key names `latitude` and `longitude` are case-sensitive.

The point constructed by this function can be saved as a node/relationship property or used within the query, such as in a `distance` function call.

### shortestPath
The `shortestPath()` function is invoked with the form:
```sh
MATCH (a {v: 1}), (b {v: 4}) RETURN shortestPath((a)-[:L*]->(b))
```

The sole `shortestPath` argument is a traversal pattern. This pattern's endpoints must be resolved prior to the function call, and no property filters may be introduced in the pattern. The relationship pattern may specify any number of relationship types (including zero) to be considered. If a minimum number of edges to traverse is specified, it may only be 0 or 1, while any number may be used for the maximum. If 0 is specified as the minimum, the source node will be included in the returned path. If no shortest path can be found, NULL is returned.

### JSON format
`toJSON()` returns the input value in JSON formatting. For primitive data types and arrays, this conversion is conventional. Maps and map projections (`toJSON(node { .prop} )`) are converted to JSON objects, as are nodes and relationships.

The format for a node object in JSON is:
```sh
{
  "type": "node",
  "id": id(int),
  "labels": [label(string) X N],
  "properties": {
    property_key(string): property_value X N
  }
}
```

The format for a relationship object in JSON is:
```sh
{
  "type": "relationship",
  "id": id(int),
  "relationship": type(string),
  "properties": {
    property_key(string): property_value X N
  }
  "start": src_node(node),
  "end": dest_node(node)
}
```

## Procedures
Procedures are invoked using the syntax:
```sh
GRAPH.QUERY social "CALL db.labels()"
```

Or the variant:
```sh
GRAPH.QUERY social "CALL db.labels() YIELD label"
```

YIELD modifiers are only required if explicitly specified; by default the value in the 'Yields' column will be emitted automatically.

| Procedure                       | Arguments                                       | Yields                        | Description                                                                                                                                                                            |
| -------                         | :-------                                        | :-------                      | :-----------                                                                                                                                                                           |
| db.labels                       | none                                            | `label`                       | Yields all node labels in the graph.                                                                                                                                                   |
| db.relationshipTypes            | none                                            | `relationshipType`            | Yields all relationship types in the graph.                                                                                                                                            |
| db.propertyKeys                 | none                                            | `propertyKey`                 | Yields all property keys in the graph.                                                                                                                                                 |
| db.indexes                      | none                                            | `type`, `label`, `properties`, `language`, `stopwords`, `entitytype`, `info` | Yield all indexes in the graph, denoting whether they are exact-match or full-text and which label and properties each covers and whether they are indexing node or relationship attributes. |
| db.constraints                  | none                                            | `type`, `label`, `properties`, `entitytype`, `status` | Yield all constraints in the graph, denoting constraint type (UNIQIE/MANDATORY), which label/relationship-type and properties each enforces. |
| db.idx.fulltext.createNodeIndex | `label`, `property` [, `property` ...]          | none                          | Builds a full-text searchable index on a label and the 1 or more specified properties.                                                                                                 |
| db.idx.fulltext.drop            | `label`                                         | none                          | Deletes the full-text index associated with the given label.                                                                                                                           |
| db.idx.fulltext.queryNodes      | `label`, `string`                               | `node`, `score`               | Retrieve all nodes that contain the specified string in the full-text indexes on the given label.                                                                                      |
| algo.pageRank                   | `label`, `relationship-type`                    | `node`, `score`               | Runs the pagerank algorithm over nodes of given label, considering only edges of given relationship type.                                                                              |
| [algo.BFS](#BFS)                | `source-node`, `max-level`, `relationship-type` | `nodes`, `edges`              | Performs BFS to find all nodes connected to the source. A `max level` of 0 indicates unlimited and a non-NULL `relationship-type` defines the relationship type that may be traversed. |
| dbms.procedures()               | none                                            | `name`, `mode`                | List all procedures in the DBMS, yields for every procedure its name and mode (read/write).                                                                                            |

### Algorithms

#### BFS
The breadth-first-search algorithm accepts 3 arguments:

`source-node (node)` - The root of the search.

`max-level (integer)` - If greater than zero, this argument indicates how many levels should be traversed by BFS. 1 would retrieve only the source's neighbors, 2 would retrieve all nodes within 2 hops, and so on.

`relationship-type (string)` - If this argument is NULL, all relationship types will be traversed. Otherwise, it specifies a single relationship type to perform BFS over.

It can yield two outputs:

`nodes` - An array of all nodes connected to the source without violating the input constraints.

`edges` - An array of all edges traversed during the search. This does not necessarily contain all edges connecting nodes in the tree, as cycles or multiple edges connecting the same source and destination do not have a bearing on the reachability this algorithm tests for. These can be used to construct the directed acyclic graph that represents the BFS tree. Emitting edges incurs a small performance penalty.

## Indexing

RedisGraph supports single-property indexes for node labels and for relationship type. String, numeric, and geospatial data types can be indexed.

### Creating an index for a node label

For a node label, the index creation syntax is:

```sh
GRAPH.QUERY DEMO_GRAPH "CREATE INDEX FOR (p:Person) ON (p.age)"
```

An old syntax is also supported:

```sh
GRAPH.QUERY DEMO_GRAPH "CREATE INDEX ON :Person(age)"
```

After an index is explicitly created, it will automatically be used by queries that reference that label and any indexed property in a filter.

```sh
GRAPH.EXPLAIN DEMO_GRAPH "MATCH (p:Person) WHERE p.age > 80 RETURN p"
1) "Results"
2) "    Project"
3) "        Index Scan | (p:Person)"
```

This can significantly improve the runtime of queries with very specific filters. An index on `:employer(name)`, for example, will dramatically benefit the query:

```sh
GRAPH.QUERY DEMO_GRAPH
"MATCH (:Employer {name: 'Dunder Mifflin'})-[:EMPLOYS]->(p:Person) RETURN p"
```

An example of utilizing a geospatial index to find `Employer` nodes within 5 kilometers of Scranton is:

```sh
GRAPH.QUERY DEMO_GRAPH
"WITH point({latitude:41.4045886, longitude:-75.6969532}) AS scranton MATCH (e:Employer) WHERE distance(e.location, scranton) < 5000 RETURN e"
```

Geospatial indexes can currently only be leveraged with `<` and `<=` filters; matching nodes outside of the given radius is performed using conventional matching.

### Creating an index for a relationship type

For a relationship type, the index creation syntax is:

```sh
GRAPH.QUERY DEMO_GRAPH "CREATE INDEX FOR ()-[f:FOLLOW]-() ON (f.created_at)"
```

Then the execution plan for using the index:

```sh
GRAPH.EXPLAIN DEMO_GRAPH "MATCH (p:Person {id: 0})-[f:FOLLOW]->(fp) WHERE 0 < f.created_at AND f.created_at < 1000 RETURN fp"
1) "Results"
2) "    Project"
3) "        Edge By Index Scan | [f:FOLLOW]"
4) "            Node By Index Scan | (p:Person)"
```

This can significantly improve the runtime of queries that traverse super nodes or when we want to start traverse from relationships.

### Deleting an index for a node label

For a node label, the index deletion syntax is:

```sh
GRAPH.QUERY DEMO_GRAPH "DROP INDEX ON :Person(age)"
```

### Deleting an index for a relationship type

For a relationship type, the index deletion syntax is:

```sh
GRAPH.QUERY DEMO_GRAPH "DROP INDEX ON :FOLLOW(created_at)"
```

## Full-text indexing

RedisGraph leverages the indexing capabilities of [RediSearch](/docs/stack/search/index.html) to provide full-text indices through procedure calls. 

### Creating a full-text index for a node label

To construct a full-text index on the `title` property of all nodes with label `Movie`, use the syntax:

```sh
GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.createNodeIndex('Movie', 'title')"
```

More properties can be added to this index by adding their names to the above set of arguments, or using this syntax again with the additional names.

```sh
GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.createNodeIndex('Person', 'firstName', 'lastName')"
```

RediSearch provide 2 index configuration options:
1. Language - Define which language to use for stemming text which is adding the base form of a word to the index. This allows the query for "going" to also return results for "go" and "gone", for example.
2. Stopwords - These are words that are usually so common that they do not add much information to search, but take up a lot of space and CPU time in the index.

To construct a full-text index on the `title` property using `German` language and using custom stopwords of all nodes with label `Movie`, use the syntax:

```sh
GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.createNodeIndex({ label: 'Movie', language: 'German', stopwords: ['a', 'ab'] }, 'title')"
```

RediSearch provide 3 additional field configuration options:
1. Weight - The importance of the text in the field
2. Nostem - Skip stemming when indexing text
3. Phonetic - Enable phonetic search on the text

To construct a full-text index on the `title` property with phonetic search of all nodes with label `Movie`, use the syntax:

```sh
GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.createNodeIndex('Movie', {field: 'title', phonetic: 'dm:en'})"
```

### Utilizing a full-text index for a node label

An index can be invoked to match any whole words contained within:

```sh
GRAPH.QUERY DEMO_GRAPH
"CALL db.idx.fulltext.queryNodes('Movie', 'Book') YIELD node RETURN node.title"
1) 1) "node.title"
2) 1) 1) "The Jungle Book"
   2) 1) "The Book of Life"
3) 1) "Query internal execution time: 0.927409 milliseconds"
```

This CALL clause can be interleaved with other Cypher clauses to perform more elaborate manipulations:
```sh
GRAPH.QUERY DEMO_GRAPH
"CALL db.idx.fulltext.queryNodes('Movie', 'Book') YIELD node AS m
WHERE m.genre = 'Adventure'
RETURN m ORDER BY m.rating"
1) 1) "m"
2) 1) 1) 1) 1) "id"
            2) (integer) 1168
         2) 1) "labels"
            2) 1) "Movie"
         3) 1) "properties"
            2) 1) 1) "genre"
                  2) "Adventure"
               2) 1) "rating"
                  2) "7.6"
               3) 1) "votes"
                  2) (integer) 151342
               4) 1) "year"
                  2) (integer) 2016
               5) 1) "title"
                  2) "The Jungle Book"
3) 1) "Query internal execution time: 0.226914 milliseconds"
```

In addition to yielding matching nodes, full-text index scans will return the score of each node. This is the [TF-IDF](/docs/stack/search/reference/scoring/#tfidf-default) score of the node, which is informed by how many times the search terms appear in the node and how closely grouped they are. This can be observed in the example:
```sh
GRAPH.QUERY DEMO_GRAPH
"CALL db.idx.fulltext.queryNodes('Node', 'hello world') YIELD node, score RETURN score, node.val"
1) 1) "score"
   2) "node.val"
2) 1) 1) "2"
      2) "hello world"
   2) 1) "1"
      2) "hello to a different world"
3) 1) "Cached execution: 1"
   2) "Query internal execution time: 0.335401 milliseconds"
```

### Deleting a full-text index for a node label

For a node label, the full-text index deletion syntax is:

```
GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.drop('Movie')"
```
