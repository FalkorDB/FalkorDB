/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "graph/graphcontext.h"
#include "commands/cmd_context.h"

// initialize global variables
void Globals_Init(void);

// read global variable 'process_is_child'
bool Globals_Get_ProcessIsChild(void);

// set global variable 'process_is_child'
void Globals_Set_ProcessIsChild
(
	bool process_is_child
);

// return number of graphs in keyspace
int Globals_GetGraphCount(void);

// get direct access to 'graphs_in_keyspace'
GraphContext **Globals_Get_GraphsInKeyspace(void);

// add graph to global tracker
void Globals_AddGraph
(
	GraphContext *gc  // graph to add
);

// get graph by name
GraphContext Globals_GetGraph
(
	const char *name  // name of graph to retrieve
);

// remove graph from global tracker
void Globals_RemoveGraph
(
	GraphContext *gc  // graph to remove
);

// remove a graph by its name
void Globals_RemoveGraphByName
(
	const char *name  // graph name to remove
);

// clear all tracked graphs
void Globals_ClearGraphs(
	RedisModuleCtx *ctx
);

//------------------------------------------------------------------------------
// Command context tracking
//------------------------------------------------------------------------------

// track CommandCtx
void Globals_TrackCommandCtx
(
	CommandCtx *ctx  // CommandCtx to track
);

// untrack CommandCtx
void Globals_UntrackCommandCtx
(
	const CommandCtx *ctx  // CommandCtx to untrack
);

// get all currently tracked CommandCtxs
void Globals_GetCommandCtxs
(
	CommandCtx **commands,  // array to populate
	uint32_t *count         // [input/output] number of entries in 'commands'
);

// free globals
void Globals_Free(void);

//------------------------------------------------------------------------------
// Graphs iterator
//------------------------------------------------------------------------------

// graphs_in_keyspace iterator
typedef struct KeySpaceGraphIterator {
	uint64_t idx;       // current graph index
} KeySpaceGraphIterator;

// initialize iterator over graphs in keyspace
void Globals_ScanGraphs(KeySpaceGraphIterator *it);

// seek iterator to index
void GraphIterator_Seek
(
	KeySpaceGraphIterator *it,  // iterator
	uint64_t idx                // index to seek to
);

// advance iterator
// returns graph object in case iterator isn't depleted
// otherwise returns NULL
// NOTICE:
// if caller doesn't hold the GIL thoughout the iterator scan
// it is possible for the scanning thread to miss keys which are deleted
// consider the iterator is at position 4 and graph at position 0 is deleted
// in which case array_del_fast will move the last graph (say index 9) to
// index 0. in this case the iterator will miss that migrated graph
GraphContext *GraphIterator_Next
(
	KeySpaceGraphIterator *it  // iterator to advance
);

