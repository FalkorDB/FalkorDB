/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

// construct the execution-plan corresponding to a call {} clause
void buildCallSubqueryPlan
(
	ExecutionPlan *plan,            // execution plan to add plan to
	const cypher_astnode_t *clause  // call subquery clause
);
