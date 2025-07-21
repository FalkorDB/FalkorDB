/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "ast.h"

// rewrite WITH/RETURN * clauses in query to project explicit identifiers,
// returning true if a rewrite has been performed
bool AST_RewriteStarProjections
(
    const cypher_astnode_t *root  // root for which to rewrite star projections
);
