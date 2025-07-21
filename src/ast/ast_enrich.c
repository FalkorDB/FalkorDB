/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "enrichment/annotate_projected_named_paths.h"

//------------------------------------------------------------------------------
//  Main AST enrichment
//------------------------------------------------------------------------------

void AST_Enrich(AST *ast) {
	/* Directives like CREATE INDEX are not queries. */
	if(cypher_astnode_type(ast->root) != CYPHER_AST_QUERY) return;
	AST_AnnotateNamedPaths(ast);
}
