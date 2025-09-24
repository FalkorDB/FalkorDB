/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "sipath.h"
#include "RG.h"
#include "../../util/rmalloc.h"
#include "../../util/arr.h"
#include "../array.h"

SIValue SIPath_New(Path *p) {
	SIValue path;
	path.ptrval = Path_Clone(p);
	path.type = T_PATH;
	path.allocation = M_SELF;
	return path;
}

SIValue SIPath_Clone(SIValue p) {
	return SIPath_New((Path *)p.ptrval);
}

SIValue SIPath_ToList(SIValue p) {
	size_t nodeCount = SIPath_NodeCount(p);
	size_t edgeCount = SIPath_Length(p);
	SIValue array = SI_Array(nodeCount + edgeCount);
	for(size_t i = 0; i < nodeCount - 1 ; i++) {
		SIValue node = SIPath_GetNode(p, i);
		SIArray_Append(&array, node);
		SIValue edge = SIPath_GetRelationship(p, i);
		SIArray_Append(&array, edge);
	}
	if(nodeCount > 0) {
		SIValue node = SIPath_GetNode(p, nodeCount - 1);
		SIArray_Append(&array, node);
	}
	return array;
}

SIValue SIPath_Relationships(SIValue p) {
	Path *path = (Path *) p.ptrval;
	uint edgeCount = Path_EdgeCount(path);
	SIValue array = SIArray_New(edgeCount);
	for(uint i = 0; i < edgeCount; i++) {
		SIArray_Append(&array, SI_Edge(Path_GetEdge(path, i)));
	}
	return array;
}

SIValue SIPath_GetRelationship(SIValue p, size_t i) {
	ASSERT(i < SIPath_Length(p));
	Path *path = (Path *) p.ptrval;
	return SI_Edge(Path_GetEdge(path, i));
}

SIValue SIPath_Nodes(SIValue p) {
	Path *path = (Path *) p.ptrval;
	uint nodeCount = Path_NodeCount(path);
	SIValue array = SIArray_New(nodeCount);
	for(uint i = 0; i < nodeCount; i++) {
		SIArray_Append(&array, SI_Node(Path_GetNode(path, i)));
	}
	return array;
}

SIValue SIPath_GetNode(SIValue p, size_t i) {
	ASSERT(i < SIPath_NodeCount(p));
	Path *path = (Path *) p.ptrval;
	return SI_Node(Path_GetNode(path, i));
}

SIValue SIPath_Head(SIValue p) {
	return SIPath_GetNode(p, 0);
}

SIValue SIPath_Last(SIValue p) {
	return(SIPath_GetNode(p, SIPath_NodeCount(p) - 1));
}

size_t SIPath_Length(SIValue p) {
	Path *path = (Path *) p.ptrval;
	return Path_Len(path);
}

size_t SIPath_NodeCount(SIValue p) {
	Path *path = (Path *) p.ptrval;
	return Path_NodeCount(path);
}

size_t SIPath_EdgeCount(SIValue p) {
	Path *path = (Path *) p.ptrval;
	return Path_EdgeCount(path);
}

XXH64_hash_t SIPath_HashCode(SIValue p) {
	SIType t = SI_TYPE(p);
	XXH64_hash_t hashCode = XXH64(&t, sizeof(t), 0);
	size_t nodeCount = SIPath_NodeCount(p);
	for(size_t i = 0; i < nodeCount - 1 ; i++) {
		SIValue node = SIPath_GetNode(p, i);
		hashCode = 31 * hashCode + SIValue_HashCode(node);
		SIValue edge = SIPath_GetRelationship(p, i);
		hashCode = 31 * hashCode + SIValue_HashCode(edge);
	}
	// Handle last node.
	if(nodeCount > 0) {
		SIValue node = SIPath_GetNode(p, nodeCount - 1);
		hashCode = 31 * hashCode + SIValue_HashCode(node);
	}
	return hashCode;
}

void SIPath_ToString(SIValue p, char **buf, size_t *bufferLen, size_t *bytesWritten) {
	// 64 is defiend arbitrarily.
	if(*bufferLen - *bytesWritten < 64) {
		*bufferLen += 64;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}
	// open path with "["
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "[");

	size_t nodeCount = SIPath_NodeCount(p);
	for(size_t i = 0; i < nodeCount - 1; i ++) {
		// write the next value
		SIValue node = SIPath_GetNode(p, i);
		SIValue_ToString(node, buf, bufferLen, bytesWritten);
		* bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ", ");
		SIValue edge = SIPath_GetRelationship(p, i);
		SIValue_ToString(edge, buf, bufferLen, bytesWritten);
		* bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ", ");
	}
	// Handle last node.
	if(nodeCount > 0) {
		SIValue node = SIPath_GetNode(p, nodeCount - 1);
		SIValue_ToString(node, buf, bufferLen, bytesWritten);
	}

	if(*bufferLen - *bytesWritten < 2) {
		*bufferLen += 2;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}
	// close array with "]"
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "]");
}

int SIPath_Compare(SIValue p1, SIValue p2) {

	size_t p1NodeCount = SIPath_NodeCount(p1);
	size_t p2NodeCount = SIPath_NodeCount(p2);
	// Get minimal length
	size_t nodeCount = p1NodeCount <= p2NodeCount ? p1NodeCount : p2NodeCount;
	int res = 0;
	for(size_t i = 0; i < nodeCount - 1 ; i++) {
		SIValue p1node = SIPath_GetNode(p1, i);
		SIValue p2node = SIPath_GetNode(p2, i);
		res = SIValue_Compare(p1node, p2node, NULL);
		if(res) return res;
		SIValue p1edge = SIPath_GetRelationship(p1, i);
		SIValue p2edge = SIPath_GetRelationship(p2, i);
		res = SIValue_Compare(p1edge, p2edge, NULL);
		if(res) return res;
	}
	// Handle last node.
	if(nodeCount > 0) {
		SIValue p1node = SIPath_GetNode(p1, nodeCount - 1);
		SIValue p2node = SIPath_GetNode(p2, nodeCount - 1);
		res = SIValue_Compare(p1node, p2node, NULL);
		if(res) return res;
	}
	return p1NodeCount - p2NodeCount;
}

void SIPath_Free(SIValue p) {
	if(p.allocation == M_SELF) {
		Path *path = (Path *) p.ptrval;
		Path_Free(path);
	}
}

