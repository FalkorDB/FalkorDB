Algorithms
==========

Algorithms come in two flavors: `Basic`_ and `Advanced`_.

Basic
-----

Basic algorithm are meant to be easy to use.  A single basic algorithm may
encompass many underlying Advanced algorithms, each with various parameters
that may be controlled.  For the Basic API, these parameters are determined
automatically.  Cached graph properties may be determined, and as a result,
the graph G is both an input and an output of these methods, since they may
be modified.

LAGraph Basic algorithms are named with the ``LAGraph_*`` prefix.

.. doxygenfunction:: LAGraph_TriangleCount

Advanced
--------

The Advanced algorithms require the caller to select the algorithm and choose
any parameter settings.  G is not modified, and so it is an input-only
parameter to these methods.  If an Advanced algorithm requires a cached
graph property to be computed, it must be computed prior to calling the
Advanced method.

Advanced algorithms are named with the ``LAGr_*`` prefix, to distinguish them
from Basic algorithms.

.. doxygenfunction:: LAGr_SortByDegree

.. doxygenfunction:: LAGr_SampleDegree

.. doxygenfunction:: LAGr_BreadthFirstSearch

.. doxygenfunction:: LAGr_ConnectedComponents

.. doxygenfunction:: LAGr_SingleSourceShortestPath

.. doxygenfunction:: LAGr_Betweenness

.. doxygenfunction:: LAGr_PageRank

.. doxygenfunction:: LAGr_TriangleCount

.. doxygenenum:: LAGr_TriangleCount_Method

.. doxygenenum:: LAGr_TriangleCount_Presort
