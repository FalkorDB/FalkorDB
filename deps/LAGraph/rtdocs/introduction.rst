Introduction
============

A graph is a set of vertices and a set of edges between them.  This pair of sets
leads directly to the familiar picture of a graph as a set of dots connected
by arcs (an undirected graphs) or arrows (a directed graph). You can also represent a 
graph in terms of matrices.   Usually, this is done with an adjacency matrix where
the rows and columns correspond to the vertices and the non-empty
elements represent the edges between vertices. Since fully connected graphs (i.e., every
vertex is connected to every other vertex) are rare, matrices used for Graphs are
typically sparse (most elements are empty so it makes sense to only store the non-empty 
elements).

Representing a graph as a sparse matrix results in graph algorithms expressed in 
terms of linear algebra.    For example, if a vector represents a set of vertices,
multiplication of that vector by an adjacency matrix returns a vector of the neighbors of those vertices.
A sequence of multiplications traverses the graph in a breadth first manner.  

To cover the full range of graph algorithms, one additional ingredient is needed.  We are used to 
thinking of matrix operations over real numbers: multiply pairs of matrix elements and then
combine the resulting products through addition.  There are times, however, when those operations do
not provide the functionality needed by an algorithm. For example, it may be better to combine elements by
only keeping the minimum value.   The elements of the matrices may be Boolean values or integers or
even a user-defined type.  If the goal is to cover the full range of graph algorithms, therefore,
we need a way to generalize the type and the operators to use instead of the usual addition and multiplication.

We do this through an algebraic semiring.   This algebraic structure consists of (1) an operator
corresponding to addition, (2) the identity of that operator, (3) an operator corresponding to multiplication,
and (4) the identity of that operator.  We are all familiar with the semiring used with real numbers
consisting of (+,0,*,1).  A common semiring in graph algorithms is the so-called tropical semiring 
consisting of (min,infinity,+,0).  This is used in shortest path algorithms.   These semirings 
give us a mathematically rigorous way to modify the operators used in our graph algorithms.

If you work with linear algebra, you most likely know about the Basic Linear Algebra subprograms or BLAS.
Introduced in the 70's and 80's, the BLAS had a huge impact on the practice of linear algebra.  By designing
linear algebra in terms of the BLAS, an algorithm can be expressed at a high level leaving specialization to the 
low level details of a particular hardware platform to the BLAS.  So if you want to use Linear Algebra for 
Graph Algorithms, it stands to reason that you need the Basic Linear Algebra Subprograms for Graph Algorithms.
We call these the GraphBLAS (`www.graphblas.org <https://www.graphblas.org>`_).

The GraphBLAS define opaque types for a matrix and a vector objects.  Since these objects are opaque, an implementation
has the freedom to specialize the data structures as needed to optimize the software for a particular platform.  The
GraphBLAS are great for people interested in sparse linear algebra and designing their own graph algorithms.
The GraphBLAS library, however, does not include any graph algorithms.  The GraphBLAS provide a software framework for
constructing graph algorithms, but it doesn't provide any actual Graph Algorithms.  Since most people working with
graphs use algorithms but don't develop them "from scratch", the GraphBLAS are not really useful to most people.

Hence, there is a need for a library of Graph Algorithms implemented on top of the GraphBLAS. We have created this
library.  It is called LAGraph.   The LAGraph library is a library of functions that implement the most common
high level graph algorithms used in graph analytics.  It includes types, utility functions and everything needed
to incorporate graph algorithms into your analytics work flows.  The library uses the GraphBLAS objects (e.g., GrB_matrix
and GrB_vector) inside the objects defined by LAGraph.  Consequently, GraphBLAS and LAGraph functions can be freely mixed
inside a single program.

A graph in LAGraph uses the LAGraph_Graph data type.  Unlike the GrB_matrix object, an LAGraph_Graph
object is not opaque.  The elements of the data structure are available to the user of the LAGraph
library.  The data associated with and LAGraph_Graph is represented by an GrB_matrix.  The data structure
includes information about the graph and key properties of the graph.  For example, many algorithms require
not only the matrix representing a graph, but also its transpose. These (and other) properties can be stored
within the LAGraph_Graph.  Storage of properties such as the transpose of a matrix requires additional storage,
but the performance impact can more than compensate for the cost associated with that extra memory.

The algorithms within LAGraph roughly break down into two categories: Basic (``LAGraph_*``) and
advanced (``LAGr_*``).  The idea is that users who are not familiar with the ways graph algorithms
are implemented and just want to apply an algorithm to their graphs, would use the Basic interface.
For advanced users who are comfortable working with key aspects of the algorithms they are working with
might see a significant performance benefit from working with the advanced algorithm.

For example, the basic and advanced algorithms deal with the properties of an LAGraph graph differently.
The basic algorithm assumes the user will not set-up the LAGraph_Graph with the properties needed by an algorithm.
Such properties will be computed as needed.  An advanced user, however, may know that the string of operations
in a workflow all requires a subset of key properties.   By computing them in advanced and storing them with the
LAGraph graph, the workflow can run much faster since it won't need to, for example, rearrange a matrix into its
transpose for each algorithm in a workflow.
