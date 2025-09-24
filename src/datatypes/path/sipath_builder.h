/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "sipath.h"

/* The API allows to build a SIPath. The current implementation allows to build the path according to
 * named path AST path pattern, when iterated in order, or in other words, when the given query is read
 * from left to right.
 * "MATCH p=(a)-[*]->(b)-[]->(c) RETURN p" will yield the following building sequence
 * 1. Create new empty path.
 * 2. Append node (a).
 * 3. Append path. All variable length traversal results, when part of path building sequence, are path themsevles.
 *    Note that the intermidiate path order might be reversed, during path building, to comply to the query.
 *    Since intermidate path are not projected, they can be mutable.
 * 4. Append node (b).
 * 5. Append edge. Note: If the edge value source and destination are reversed to the query pattern,
 *    a copy of the edge will be added to the path, with the source and destination swapped. This is due to
 *    SIEdge value is immutable, since it might be projected from the record.
 * 6. Append node (c). */

/**
 * @brief  Creates a new empty SIPath with allocated space to given number of entities.
 * @note   The size entity_count is just an initial capcity and can dynamically grow.
 * @param  entity_count: Initial number of entities.
 * @retval Empty SIPath.
 */
SIValue SIPathBuilder_New(uint entity_count);

/**
 * @brief  Appends a SINode into SIPath.
 * @param  p: SIPath.
 * @param  n: SINode.
 */
void SIPathBuilder_AppendNode(SIValue p, SIValue n);

/**
 * @brief  Appends a SIEdge into SIPath.
 * @note   The edge should be added after its source node has been inserted to its right position in the path.
 *         Edges insertion is done by interliving nodes and edges.
 *         If edge insertion is done not in the right order, an assertion will be thrown.
 * @param  p: SIPath.
 * @param  e: SIEdge.
 * @param  RTLEdge: Indicates if the edge is incoming or outgoing edge (RTL in query).
 */
void SIPathBuilder_AppendEdge(SIValue p, SIValue e, bool RTLEdge);

/**
 * @brief  Appends a path into an existing one.
 * @note   The path is added after its first or last node is added. The actual values from the path that will be taken
 *         are the edges and all the nodes besides the first and the last, since they are added in different steps of the
 *         path building.
 * @param  p: SIPath.
 * @param  other: SIPath needs to be appended to p.
 * @param  RTLEdge: Indicates edges direction if the path (RTL in query, incoming or outgoing).
 */
void SIPathBuilder_AppendPath(SIValue p, SIValue other, bool RTLEdge);
