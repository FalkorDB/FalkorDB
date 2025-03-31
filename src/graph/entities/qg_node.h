/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include "../../util/sds/sds.h"

// forward declaration of edge
struct QGEdge;

typedef struct {
	int *labelsID;                   // labels ID array
	const char *alias;               // user-provided alias associated with this node
	const char **labels;             // labels string array
	bool highly_connected;           // node degree > 2
	struct QGEdge **outgoing_edges;  // array of outgoing edges (ME)->(DEST)
	struct QGEdge **incoming_edges;  // array of incoming edges (ME)<-(SRC)
} QGNode;

// creates a new node
QGNode *QGNode_New
(
	const char *alias
);

// returns the alias of the node
const char *QGNode_Alias
(
	const QGNode *n
);

// returns true if the node is labeled
bool QGNode_Labeled
(
	const QGNode *n
);

// returns number of labels attached to node
uint QGNode_LabelCount
(
	const QGNode *n
);

// returns the 'idx' label ID of 'n'
int QGNode_GetLabelID
(
	const QGNode *n,
	uint idx
);

// returns the 'idx' label of 'n'
const char *QGNode_GetLabel
(
	const QGNode *n,
	uint idx
);

// label 'n' as 'l'
void QGNode_AddLabel
(
	QGNode *n,
	const char *l,
	int l_id
);

// returns true if node is highly connected, false otherwise
bool QGNode_HighlyConnected
(
	const QGNode *n
);

// returns the number of both incoming and outgoing edges
int QGNode_Degree
(
	const QGNode *n
);

// returns number of edges pointing into node
int QGNode_IncomeDegree
(
	const QGNode *n
);

// returns number of edges pointing out of node
int QGNode_OutgoingDegree
(
	const QGNode *n
);

// returns to total number of edges (incoming & outgoing)
int QGNode_EdgeCount
(
	const QGNode *n
);

// connects source node to destination node by edge
void QGNode_ConnectNode
(
	QGNode *src,
	QGNode *dest,
	struct QGEdge *e
);

// removes given incoming edge from node
void QGNode_RemoveIncomingEdge
(
	QGNode *n,
	struct QGEdge *e
);

// removes given outgoing edge from node
void QGNode_RemoveOutgoingEdge
(
	QGNode *n,
	struct QGEdge *e
);

// clones given node
QGNode *QGNode_Clone
(
	const QGNode *n
);

// gets a string representation of given node
void QGNode_ToString
(
	const QGNode *n,  // target node
	sds *buff         // result buffer (concatenated)
);

// frees allocated space by given node
void QGNode_Free
(
	QGNode *node
);

