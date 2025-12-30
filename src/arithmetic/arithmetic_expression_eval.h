/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// returns true if given node 'n' represents an aggregation expression
#define AGGREGATION_NODE(n) ((AR_EXP_IsOperation(n)) && (n)->op.f->aggregate)

// return number of child nodes of 'n'
#define NODE_CHILD_COUNT(n) (n)->op.child_count

// return child at position 'idx' of 'n'
#define NODE_CHILD(n, idx) (n)->op.children[(idx)]

// maximum size for which an array of SIValue will be stack-allocated, otherwise it will be heap-allocated.
#define MAX_ARRAY_SIZE_ON_STACK 32

