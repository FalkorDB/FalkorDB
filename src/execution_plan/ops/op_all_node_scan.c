/*
* Copyright 2018-2021 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "op_all_node_scan.h"
#include "shared/print_functions.h"

OpBase *NewAllNodeScanOp(const ExecutionPlan *plan, const char *alias) {
	AllNodeScan *op = rm_malloc(sizeof(AllNodeScan));
	op->alias = alias;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_ALL_NODE_SCAN, "All Node Scan", NULL,
		false, plan);

	OpBase_Modifies((OpBase *)op, alias);
	
	return (OpBase *)op;
}
