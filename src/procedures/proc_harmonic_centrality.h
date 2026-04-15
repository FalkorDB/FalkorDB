/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"

// compute harmonic closeness centrality using HLL BFS propagation
int LAGr_HarmonicCentrality(
	// outputs:
	GrB_Vector *scores,          // FP64 scores by original node ID
	GrB_Vector *reachable_nodes, // [optional] estimate the number of reach-
	                             // able nodes from the given node.
	// inputs:
	const LAGraph_Graph G,         // input graph
	const GrB_Vector node_weights, // participating nodes and their weights
	char *msg
) ;

// run harmonic closeness centrality on sub graph using HLL BFS propagation
ProcedureCtx *Proc_HarmonicCentralityCtx(void);
