/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

/*
    LAGraph:  graph algorithms based on GraphBLAS
    Copyright 2019 LAGraph Contributors.
    (see Contributors.txt for a full list of Contributors; see
    ContributionInstructions.txt for information on how you can Contribute to
    this project).
    All Rights Reserved.
    NO WARRANTY. THIS MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. THE LAGRAPH
    CONTRIBUTORS MAKE NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
    AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR
    PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF
    THE MATERIAL. THE CONTRIBUTORS DO NOT MAKE ANY WARRANTY OF ANY KIND WITH
    RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
    Released under a BSD license, please see the LICENSE file distributed with
    this Software or contact permission@sei.cmu.edu for full terms.
    Created, in part, with funding and support from the United States
    Government.  (see Acknowledgments.txt file).
    This program includes and/or can make use of certain third party source
    code, object code, documentation and other files ("Third Party Software").
    See LICENSE file for more details.
*/

#pragma once

#include "GraphBLAS/Include/GraphBLAS.h"

typedef struct {
	GrB_Index page ;    // the node number itself
	double pagerank ;   // the pagerank of a node
}
LAGraph_PageRank ;

GrB_Info Pagerank               // GrB_SUCCESS or error condition
(
	LAGraph_PageRank **Phandle, // output: array of LAGraph_PageRank structs
	GrB_Matrix A,               // binary input graph, not modified
	int itermax,                // max number of iterations
	double tol,                 // stop when norm (r-rnew,2) < tol
	int *iters                  // number of iterations taken
);
