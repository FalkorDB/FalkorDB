/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../value.h"
#include "path.h"
#include "../../graph/entities/qg_edge.h"
#include <stdlib.h>

/**
 * @brief  Creates a new SIPath out of path struct.
 * @param  p: Path struct pointer.
 * @retval SIValue which represents the given struct.
 */
SIValue SIPath_New(Path *p);

/**
 * @brief  Clones a given SIPath.
 * @param  p: SIPath.
 * @retval New SIPath with newly allocated path clone.
 */
SIValue SIPath_Clone(SIValue p);

/**
 * @brief  Returns a path as in a list represention of interliving nodes and edges.
 * @param  p: SIPath
 * @retval SIArray.
 */
SIValue SIPath_ToList(SIValue p);

/**
 * @brief  Returns a SIArray with the path edges as SIEdges.
 * @param  p: SIPath
 * @retval SIArray with the path edges.
 */
SIValue SIPath_Relationships(SIValue p);

/**
 * @brief  Retruns a SIEdge in a given position in the relationships array.
 * @note   Assertion will be raised for out of bound indices.
 * @param  p: SIPath.
 * @param  i: Requested index.
 * @retval SIEdge in the requested index.
 */
SIValue SIPath_GetRelationship(SIValue p, size_t i);

/**
 * @brief  Returns SIArray with the nodes in the path as SINodes.
 * @param  p: SIPath.
 * @retval SIArray with the path nodes.
 */
SIValue SIPath_Nodes(SIValue p);

/**
 * @brief  Returns a SINode in a given position in the nodes array.
 * @note   Assertion will be raised for out of bound indices.
 * @param  p: SIPath.
 * @param  i: Requested index.
 * @retval SINode in the requested index.
 */
SIValue SIPath_GetNode(SIValue p, size_t i);

/**
 * @brief  Returns the first node in the path.
 * @note   Assertion will be raised if path is empty;
 * @param  p: SIPath.
 * @retval SINode.
 */
SIValue SIPath_Head(SIValue p);

/**
 * @brief  Returns the last node in the path.
 * @note   Assertion will be raised if the path is empty.
 * @param  p: SIPath;
 * @retval SINode.
 */
SIValue SIPath_Last(SIValue p);

/**
 * @brief  Returns the path length.
 * @note   The return value is the amount of edges in the path.
 * @param  p: SIPath
 * @retval Path length.
 */
size_t SIPath_Length(SIValue p);

/**
 * @brief  Returns the number or nodes in the path.
 * @param  p: SIPath
 * @retval Number or nodes in the path..
 */
size_t SIPath_NodeCount(SIValue p);

/**
 * @brief  Returns 64 bit hash code of the path.
 * @param  p: SIPath.
 * @retval 64 bit hash code.
 */
XXH64_hash_t SIPath_HashCode(SIValue p);

/**
 * @brief  Prints a SIPath into a given buffer.
 * @param  p: SIPath.
 * @param  buf: print buffer (pointer to pointer to allow re allocation).
 * @param  len: print buffer length.
 * @param  bytesWritten: the actual number of bytes written to the buffer.
 */
void SIPath_ToString(SIValue p, char **buf, size_t *bufferLen, size_t *bytesWritten);

/**
 * @brief  Compares between two paths.
 * @note   From opencypher docs: Paths are tested for equality as if they were a list of
 * alternating nodes and relationships of the path from the start node to the end node.
 * Two paths are equal if and only if these lists of nodes and relationships are equal.
 * Paths are also compared in this way. For example, given nodes n1, n2, n3, and relationships
 * r1 and r2, and given that n1 < n2 < n3 and r1 < r2, then the path p1 from n1 to n3 via r1 would
 * be less than the path p2 to n1 from n2 via r2
 * @param  p1: lhs path.
 * @param  p2: rhs path.
 * @retval Order between the paths.
 */
int SIPath_Compare(SIValue p1, SIValue p2);

/**
 * @brief  Free SIPath.
 * @param  p: SIPath.
 */
void SIPath_Free(SIValue p);

