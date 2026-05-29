/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"
#include "src/graph/query_graph.h"
#include "src/execution_plan/ops/ops.h"
#include "src/execution_plan/execution_plan.h"
#include "src/execution_plan/execution_plan_build/execution_plan_modify.h"

#include <stdio.h>
#include <string.h>

void setup();

#define TEST_INIT setup();

#include "acutest.h"

void setup() {
	// skip if memory sanitizer is enabled
	if(getenv("SANITIZER") != NULL || getenv("VALGRIND") != NULL) {
		exit(0);
	}

	// use the malloc family for allocations
	Alloc_Reset();
	Logging_Reset();
}

static ExecutionPlan *_EmptyExecutionPlan(void) {
	ExecutionPlan *plan = ExecutionPlan_NewEmptyExecutionPlan();

	plan->record_map  = raxNew();
	plan->query_graph = QueryGraph_New(0, 0);

	return plan;
}

// helper function to wrap OpBase_Aware
bool _Aware
(
	OpBase *op,
	const char *alias
) {
	return OpBase_Aware(op, (const char**)&(alias), 1);
}

// validate that an operation is aware of its own modifiers
// if operation OP modifies alias 'A' then OP should be aware of 'A'
void test_self_awareness() {

	ExecutionPlan *plan = _EmptyExecutionPlan();

	const char *alias = "p";
	OpBase *op = NewAllNodeScanOp(plan, alias);

	// op should be aware of alias 'p'
	bool aware = _Aware(op, alias);
	TEST_ASSERT(aware);

	OpBase_Free(op);
	ExecutionPlan_Free(plan);
}

// validate that adding a child operation causes its parent to be aware of
// the aliases it's aware of
void test_inherit_awareness() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	const char *alias = "p";
	OpBase *scan      = NewAllNodeScanOp(p, alias);
	OpBase *limit     = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));

	// limit should NOT be aware of alias 'p'
	bool aware = _Aware(limit, alias);
	TEST_ASSERT(!aware);

	// set limit as label scan parent
	// scan
	//   limit

	ExecutionPlan_AddOp(limit, scan);

	// limit inherits its children awareness
	// limit should be aware of alias 'p'
	aware = _Aware(limit, alias);
	TEST_ASSERT(aware);

	OpBase_Free(scan);
	OpBase_Free(limit);
	ExecutionPlan_Free(p);
}

// validate that adding multiple child operations updates their parent awareness
// table accordingly
void test_multiple_inheritance_awareness() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	const char *aliases[5] = {"a", "b", "c", "d", "e"};

	OpBase *cp     = NewCartesianProductOp(p);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	OpBase *scan_3 = NewAllNodeScanOp(p, aliases[3]);
	OpBase *scan_4 = NewAllNodeScanOp(p, aliases[4]);

	// cp should NOT be aware of any of the aliases
	for(int i = 0; i < 5; i++) {
		bool aware = _Aware(cp, aliases[i]);
		TEST_ASSERT(!aware);
	}

	// set limit as label scan parent
	// label scan
	//   limit

	ExecutionPlan_AddOp(cp, scan_0);
	ExecutionPlan_AddOp(cp, scan_1);
	ExecutionPlan_AddOp(cp, scan_2);
	ExecutionPlan_AddOp(cp, scan_3);
	ExecutionPlan_AddOp(cp, scan_4);

	// cp should be aware of all aliases
	for(int i = 0; i < 5; i++) {
		bool aware = _Aware(cp, aliases[i]);
		TEST_ASSERT(aware);
	}

	OpBase_Free(cp);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);
	OpBase_Free(scan_3);
	OpBase_Free(scan_4);
	ExecutionPlan_Free(p);
}

// validate that adding a multiple chained child operations
// causes their parent to be aware of all aliases
void test_inherit_chain() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[5] = {"a", "b", "c"};
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));

	// limit should NOT be aware of any of the aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(!aware);
	}

	// form chain
	// limit
	//   scan_0
	//     scan_1
	//       scan_2
	ExecutionPlan_AddOp(scan_0, scan_1);
	ExecutionPlan_AddOp(scan_1, scan_2);
	ExecutionPlan_AddOp(limit, scan_0);

	// scan_2 should only be aware of  "c"
	aware = _Aware(scan_2, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_2, "b");
	TEST_ASSERT(!aware);
	aware = _Aware(scan_2, "a");
	TEST_ASSERT(!aware);

	// scan_1 should only be aware of "c" and "b"
	aware = _Aware(scan_1, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "a");
	TEST_ASSERT(!aware);

	// scan_0 should of all aliases
	aware = _Aware(scan_0, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	// limit should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(aware);
	}

	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);
	OpBase_Free(limit);

	//--------------------------------------------------------------------------
	// repeate the same test with a different child parent introduction order
	//--------------------------------------------------------------------------

	scan_0 = NewAllNodeScanOp(p, aliases[0]);
	scan_1 = NewAllNodeScanOp(p, aliases[1]);
	scan_2 = NewAllNodeScanOp(p, aliases[2]);
	limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));

	// form chain
	// limit
	//   scan_0
	//     scan_1
	//       scan_2
	ExecutionPlan_AddOp(limit, scan_0);
	ExecutionPlan_AddOp(scan_1, scan_2);
	ExecutionPlan_AddOp(scan_0, scan_1);

	// scan_2 should only be aware of  "c"
	aware = _Aware(scan_2, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_2, "b");
	TEST_ASSERT(!aware);
	aware = _Aware(scan_2, "a");
	TEST_ASSERT(!aware);

	// scan_1 should only be aware of "c" and "b"
	aware = _Aware(scan_1, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "a");
	TEST_ASSERT(!aware);

	// scan_0 should of all aliases
	aware = _Aware(scan_0, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	// limit should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(aware);
	}

	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);
	OpBase_Free(limit);
	ExecutionPlan_Free(p);
}

// validate that upon updating the root, the new root has its awareness table
// updated
void test_update_root() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));

	// limit should NOT be aware of any of the aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(!aware);
	}

	// form chain
	// limit
	//   scan_0
	//     scan_1
	//       scan_2
	ExecutionPlan_AddOp(scan_0, scan_1);
	ExecutionPlan_AddOp(scan_1, scan_2);

	// set scan0 as root follow by root update to limit
	ExecutionPlan_UpdateRoot(p, scan_0);
	ExecutionPlan_UpdateRoot(p, limit);

	// scan_2 should only be aware of  "c"
	aware = _Aware(scan_2, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_2, "b");
	TEST_ASSERT(!aware);
	aware = _Aware(scan_2, "a");
	TEST_ASSERT(!aware);

	// scan_1 should only be aware of "c" and "b"
	aware = _Aware(scan_1, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "a");
	TEST_ASSERT(!aware);

	// scan_0 should of all aliases
	aware = _Aware(scan_0, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	// limit should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(aware);
	}

	ExecutionPlan_Free(p);

	//--------------------------------------------------------------------------
	// repeate the same test use only ExecutionPlan_UpdateRoot to form the chain
	//--------------------------------------------------------------------------

	p = _EmptyExecutionPlan();

	scan_0 = NewAllNodeScanOp(p, aliases[0]);
	scan_1 = NewAllNodeScanOp(p, aliases[1]);
	scan_2 = NewAllNodeScanOp(p, aliases[2]);
	limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));

	// form chain
	// limit
	//   scan_0
	//     scan_1
	//       scan_2

	ExecutionPlan_UpdateRoot(p, scan_2);
	ExecutionPlan_UpdateRoot(p, scan_1);
	ExecutionPlan_UpdateRoot(p, scan_0);
	ExecutionPlan_UpdateRoot(p, limit);

	// scan_2 should only be aware of  "c"
	aware = _Aware(scan_2, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_2, "b");
	TEST_ASSERT(!aware);
	aware = _Aware(scan_2, "a");
	TEST_ASSERT(!aware);

	// scan_1 should only be aware of "c" and "b"
	aware = _Aware(scan_1, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_1, "a");
	TEST_ASSERT(!aware);

	// scan_0 should of all aliases
	aware = _Aware(scan_0, "c");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "b");
	TEST_ASSERT(aware);
	aware = _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	// limit should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware = _Aware(limit, aliases[i]);
		TEST_ASSERT(aware);
	}

	ExecutionPlan_Free(p);
}

// validate that upon setting a new root
// its awareness table is updated accordingly
void test_update_root_chain() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};

	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *skip   = NewSkipOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *filter = NewFilterOp(p, NULL);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);

	// limit
	//   skip
	//     filter
	//       scan_0

	ExecutionPlan_AddOp(limit, skip);
	ExecutionPlan_AddOp(skip,  filter);
	ExecutionPlan_AddOp(filter, scan_0);
	ExecutionPlan_UpdateRoot(p, limit);

	// limit, skip & filter should be aware of all 'a'
	aware =  _Aware(limit,  "a");
	aware &= _Aware(skip,   "a");
	aware &= _Aware(filter, "a");
	TEST_ASSERT(aware);

	// scan_2
	//   scan_1 
	//     limit
	//       skip
	//         filter
	//           scan_0

	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	ExecutionPlan_AddOp(scan_2, scan_1);
	ExecutionPlan_UpdateRoot(p, scan_2);

	// limit should only be aware of "a"
	aware = _Aware(limit, aliases[0]);
	TEST_ASSERT(aware);
	aware = _Aware(limit,  aliases[1]);
	TEST_ASSERT(!aware);
	aware = _Aware(limit,  aliases[2]);
	TEST_ASSERT(!aware);

	// scan_1 should aware of "a" & "b"
	aware = _Aware(scan_1, aliases[0]);
	TEST_ASSERT(aware);
	aware = _Aware(scan_1,  aliases[1]);
	TEST_ASSERT(aware);
	aware = _Aware(scan_1,  aliases[2]);
	TEST_ASSERT(!aware);

	// scan_2 should be aware of all aliases
	aware = _Aware(scan_2, aliases[0]);
	TEST_ASSERT(aware);
	aware = _Aware(scan_2,  aliases[1]);
	TEST_ASSERT(aware);
	aware = _Aware(scan_2,  aliases[2]);
	TEST_ASSERT(aware);

	ExecutionPlan_Free(p);
}

// validate that upon adding an op at a specific index, awareness is updated
// accordingly
void test_add_op_idx() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[4] = {"a", "b", "c", "d"};

	OpBase *cp     = NewCartesianProductOp(p);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	OpBase *scan_3 = NewAllNodeScanOp(p, aliases[3]);

	ExecutionPlan_AddOp(cp, scan_0);
	ExecutionPlan_AddOp(cp, scan_1);
	ExecutionPlan_AddOp(cp, scan_2);

	// put scan_3 at position 1, replacing scan_1
	// which will be automatically readded as the last child of cp
	ExecutionPlan_AddOpInd(cp, scan_3, 1);

	// cp
	//   scan_0
	//   scan_3
	//   scan_2
	//   scan_1

	// cartesian product should be aware of all aliases
	for(int i = 0; i < 4; i++) {
		aware = _Aware(cp, aliases[i]);
		TEST_ASSERT(aware);
	}

	OpBase_Free(cp);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);
	OpBase_Free(scan_3);
	ExecutionPlan_Free(p);
}

// validate that upon pushing an op below another one
// the op's awareness and its parent are updated accordingly
void test_push_below() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};

	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *skip   = NewSkipOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *filter = NewFilterOp(p, NULL);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);

	// limit
	//   skip
	//     filter
	//       scan_0

	ExecutionPlan_AddOp(limit, skip);
	ExecutionPlan_AddOp(skip,  filter);
	ExecutionPlan_AddOp(filter, scan_0);

	// limit, skip & filter should be aware of 'a'
	aware =  _Aware(limit,  "a");
	aware &= _Aware(skip,   "a");
	aware &= _Aware(filter, "a");
	TEST_ASSERT(aware);

	//  scan_2
	//    scan_1
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	ExecutionPlan_AddOp(scan_2, scan_1);

	// limit
	//   skip
	//     scan_2
	//       scan_1
	//         filter
	//           scan_0
	ExecutionPlan_PushBelow(filter, scan_2);

	// limit, skip & scan_2 should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware =  _Aware(limit,  aliases[i]);
		aware &= _Aware(skip,   aliases[i]);
		aware &= _Aware(scan_2, aliases[i]);
		TEST_ASSERT(aware);
	}

	// filter & scan_0 should be only aware of "a"
	aware  = _Aware(filter, "a");
	aware &= _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	aware  = _Aware(filter, "b");
	TEST_ASSERT(!aware);
	aware &= _Aware(scan_0, "b");
	TEST_ASSERT(!aware);
	aware  = _Aware(filter, "c");
	TEST_ASSERT(!aware);
	aware &= _Aware(scan_0, "c");
	TEST_ASSERT(!aware);

	OpBase_Free(skip);
	OpBase_Free(limit);
	OpBase_Free(filter);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);

	ExecutionPlan_Free(p);
}

// validate that upon replacing an op awareness table is updated accordingly
void test_replace_op() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};

	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *skip   = NewSkipOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *filter = NewFilterOp(p, NULL);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);

	// limit
	//   skip
	//     filter
	//       scan_0

	ExecutionPlan_AddOp(limit, skip);
	ExecutionPlan_AddOp(skip,  filter);
	ExecutionPlan_AddOp(filter, scan_0);

	// limit, skip & filter should be aware of all 'a'
	aware =  _Aware(limit,  "a");
	aware &= _Aware(skip,   "a");
	aware &= _Aware(filter, "a");
	TEST_ASSERT(aware);

	//  scan_2
	//    scan_1
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);
	ExecutionPlan_AddOp(scan_2, scan_1);

	// limit
	//   skip
	//     scan_2
	//       scan_1
	//       scan_0
	ExecutionPlan_ReplaceOp(p, filter, scan_2);

	// limit, skip & scan_2 should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware =  _Aware(limit,  aliases[i]);
		aware &= _Aware(skip,   aliases[i]);
		aware &= _Aware(scan_2, aliases[i]);
		TEST_ASSERT(aware);
	}

	// scan_0 should be only aware of "a"
	aware &= _Aware(scan_0, "a");
	TEST_ASSERT(aware);
	aware &= _Aware(scan_0, "b");
	TEST_ASSERT(!aware);
	aware &= _Aware(scan_0, "c");
	TEST_ASSERT(!aware);

	// scan_1 should be only aware of "b"
	aware  = _Aware(scan_1, "b");
	TEST_ASSERT(aware);
	aware  = _Aware(scan_1, "a");
	TEST_ASSERT(!aware);
	aware  = _Aware(scan_1, "c");
	TEST_ASSERT(!aware);

	OpBase_Free(skip);
	OpBase_Free(limit);
	OpBase_Free(filter);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);

	ExecutionPlan_Free(p);
}

// validate that upon removing an op, awareness table is updated accordingly
void test_remove_op() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};

	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *skip   = NewSkipOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *filter = NewFilterOp(p, NULL);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);

	// limit
	//   skip
	//     filter
	//       scan_2
	//         scan_1
	//           scan_0

	ExecutionPlan_AddOp(limit, skip);
	ExecutionPlan_AddOp(skip,  filter);
	ExecutionPlan_AddOp(filter, scan_2);
	ExecutionPlan_AddOp(scan_2, scan_1);
	ExecutionPlan_AddOp(scan_1, scan_0);

	// limit
	//   skip
	//     scan_2
	//       scan_1
	//         scan_0

	ExecutionPlan_RemoveOp(p, filter);

	// limit, skip & scan_2 should be aware of all aliases
	for(int i = 0; i < 3; i++) {
		aware =  _Aware(limit,  aliases[i]);
		aware &= _Aware(skip,   aliases[i]);
		aware &= _Aware(scan_2, aliases[i]);
		TEST_ASSERT(aware);
	}

	// limit
	//   skip
	//     scan_2
	//       scan_0

	ExecutionPlan_RemoveOp(p, scan_1);

	// limit, skip & scan_2 should be aware of "c" & "a"
	aware =  _Aware(limit,  "a");
	aware &= _Aware(limit,  "c");
	aware &= _Aware(skip,   "a");
	aware &= _Aware(skip,   "c");
	aware &= _Aware(scan_2, "a");
	aware &= _Aware(scan_2, "c");
	TEST_ASSERT(aware);

	aware =  _Aware(limit,  "b");
	aware |= _Aware(skip,   "b");
	aware |= _Aware(scan_2, "b");
	TEST_ASSERT(!aware);

	OpBase_Free(skip);
	OpBase_Free(limit);
	OpBase_Free(filter);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);

	ExecutionPlan_Free(p);
}

// validate that upon detaching an op, awareness table is updated accordingly
void test_detach_op() {
	ExecutionPlan *p = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[3] = {"a", "b", "c"};

	OpBase *limit  = NewLimitOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *skip   = NewSkipOp(p, AR_EXP_NewConstOperandNode(SI_LongVal(2)));
	OpBase *filter = NewFilterOp(p, NULL);
	OpBase *scan_0 = NewAllNodeScanOp(p, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p, aliases[2]);

	// limit
	//   skip
	//     filter
	//       scan_2
	//         scan_1
	//           scan_0

	ExecutionPlan_AddOp(limit, skip);
	ExecutionPlan_AddOp(skip,  filter);
	ExecutionPlan_AddOp(filter, scan_2);
	ExecutionPlan_AddOp(scan_2, scan_1);
	ExecutionPlan_AddOp(scan_1, scan_0);

	// limit
	//   skip
	//     filter
	//       scan_2

	ExecutionPlan_DetachOp(scan_1);

	// limit, skip & scan_2 should only be aware of "c"
	aware =  _Aware(limit,  "c");
	aware &= _Aware(skip,   "c");
	aware &= _Aware(scan_2, "c");
	TEST_ASSERT(aware);

	aware =  _Aware(limit,  "a");
	aware |= _Aware(limit,  "b");
	aware |= _Aware(skip,   "a");
	aware |= _Aware(skip,   "b");
	aware |= _Aware(scan_2, "a");
	aware |= _Aware(scan_2, "b");
	TEST_ASSERT(!aware);

	// scan_1
	//   scan_0

	// scan_1 should be aware of "a" & "b"
	aware =  _Aware(scan_1, "a");
	aware &= _Aware(scan_1, "b");
	TEST_ASSERT(aware);

	aware = _Aware(scan_1, "c");
	TEST_ASSERT(!aware);

	// scan_0 should be aware of "a"
	aware = _Aware(scan_0, "a");
	TEST_ASSERT(aware);

	aware =  _Aware(scan_0, "b");
	aware |= _Aware(scan_0, "c");
	TEST_ASSERT(!aware);

	OpBase_Free(skip);
	OpBase_Free(limit);
	OpBase_Free(filter);
	OpBase_Free(scan_0);
	OpBase_Free(scan_1);
	OpBase_Free(scan_2);

	ExecutionPlan_Free(p);
}

// validate that upon binding a tree from one plan to another
// awareness table is updated accordingly
void test_bind_to_plan() {
	ExecutionPlan *p_a = _EmptyExecutionPlan();
	ExecutionPlan *p_b = _EmptyExecutionPlan();
	ExecutionPlan *p_c = _EmptyExecutionPlan();

	bool aware;
	const char *aliases[6] = {"a", "b", "c", "d", "e", "f"};

	OpBase *scan_0 = NewAllNodeScanOp(p_a, aliases[0]);
	OpBase *scan_1 = NewAllNodeScanOp(p_a, aliases[1]);
	OpBase *scan_2 = NewAllNodeScanOp(p_b, aliases[2]);
	OpBase *scan_3 = NewAllNodeScanOp(p_b, aliases[3]);
	OpBase *scan_4 = NewAllNodeScanOp(p_c, aliases[4]);
	OpBase *scan_5 = NewAllNodeScanOp(p_c, aliases[5]);

	// scan_0
	//   scan_1

	ExecutionPlan_AddOp(scan_0, scan_1);
	ExecutionPlan_UpdateRoot(p_a, scan_0);

	// scan_2
	//   scan_3

	ExecutionPlan_AddOp(scan_2, scan_3);
	ExecutionPlan_UpdateRoot(p_b, scan_2);

	// scan_4
	//   scan_5

	ExecutionPlan_AddOp(scan_4, scan_5);
	ExecutionPlan_UpdateRoot(p_c, scan_4);

	// scan_2       - [plan_b]
	//   scan_3     - [plan_b]
	//     scan_4   - [plan_c]
	//       scan_5 - [plan_c]

	ExecutionPlan_AddOp(scan_3, scan_4);

	// scan_0           - [plan_a]
	//   scan_1         - [plan_a]
	//     scan_2       - [plan_a]
	//       scan_3     - [plan_a]
	//         scan_4   - [plan_a]
	//           scan_5 - [plan_a]

	ExecutionPlan_BindOpsToPlan(p_a, scan_2);
	ExecutionPlan_AddOp(scan_1, scan_2);


	// scan_0 should be aware of all aliases
	for(int i = 0; i < 6; i++) {
		aware = _Aware(scan_0, aliases[i]);
		TEST_ASSERT(aware);
	}

	// scan_2 should be aware of aliases: "c", "d", "e" & "f"
	for(int i = 2; i < 6; i++) {
		aware = _Aware(scan_2, aliases[i]);
		TEST_ASSERT(aware);
	}

	// scan_4 should be aware of aliases: "e" & "f"
	for(int i = 4; i < 6; i++) {
		aware = _Aware(scan_4, aliases[i]);
		TEST_ASSERT(aware);
	}

	ExecutionPlan_Free(p_a);
	ExecutionPlan_Free(p_b);
	ExecutionPlan_Free(p_c);
}

TEST_LIST = {
	{"self_awareness", test_self_awareness},
	{"inherit_awareness", test_inherit_awareness},
	{"test_multiple_inheritance_awareness", test_multiple_inheritance_awareness},
	{"inherit_chain", test_inherit_chain},
	{"update_root", test_update_root},
	{"update_root_chain", test_update_root_chain},
	{"add_op_idx", test_add_op_idx},
	{"push_below", test_push_below},
	{"replace_op", test_replace_op},
	{"remove_op", test_remove_op},
	{"detach_op", test_detach_op},
	{"bind_to_plan", test_bind_to_plan},
	{NULL, NULL}
};

