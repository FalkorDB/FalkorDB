/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// OpDistinct filters out non-unique records
// record uniqueness is determined by a set of expressions
// the operation computes hash based on the distinct expressions
// if the hash wasn't encountered the record will pass onward otherwise
// the record is dropped
//
// hash(record) = hash(record[expression]) for each distinct expression
//
// MATCH (n)
// RETURN DISTINCT n.first_name, n.last_name

#include "op_distinct.h"
#include "op_project.h"
#include "op_aggregate.h"
#include "xxhash.h"
#include "../../util/arr.h"
#include "../execution_plan_build/execution_plan_modify.h"

// forward declarations
static void DistinctFree(OpBase *opBase);
static OpResult DistinctInit(OpBase *opBase);
static Record DistinctConsume(OpBase *opBase);
static OpResult DistinctReset(OpBase *opBase);
static OpBase *DistinctClone(const ExecutionPlan *plan, const OpBase *opBase);

// compute hash on distinct values
// values that are required to be distinct are located at 'offset'
// positions within the record
static unsigned long long _compute_hash
(
	OpDistinct *op,
	Record r
) {
	// initialize the hash state
	XXH64_state_t state;
	XXH_errorcode res = XXH64_reset(&state, 0);
	ASSERT(res != XXH_ERROR);

	for(uint i = 0; i < op->offset_count; i++) {
		// retrieve the entry at 'idx' as an SIValue
		uint idx = op->offsets[i];
		SIValue v = Record_Get(r, idx);
		// update the hash state with the current value
		SIValue_HashUpdate(v, &state);
	}

	// finalize the hash
	unsigned long long const hash = XXH64_digest(&state);
	return hash;
}

// create a new distinct operation
OpBase *NewDistinctOp
(
	const ExecutionPlan *plan,  // execution plan
	const char **aliases,       // distinct aliases
	uint alias_count            // number of distinct expressions
) {
	ASSERT(plan        != NULL);
	ASSERT(aliases     != NULL);
	ASSERT(alias_count > 0);

	OpDistinct *op = rm_calloc(1, sizeof(OpDistinct));

	op->found        = HashTableCreate(&def_dt);
	op->aliases      = rm_malloc(alias_count * sizeof(const char *));
	op->offset_count = alias_count;
	op->offsets      = rm_calloc(op->offset_count, sizeof(int));

	// copy aliases into heap array managed by this op
	memcpy(op->aliases, aliases, alias_count * sizeof(const char *));

	OpBase_Init((OpBase *)op, OPType_DISTINCT, "Distinct", DistinctInit,
			DistinctConsume, DistinctReset, NULL, DistinctClone, DistinctFree,
			false, plan);

	return (OpBase *)op;
}

static OpResult DistinctInit
(
	OpBase *opBase  // CallSubquery operation to initialize
) {
	OpDistinct *op = (OpDistinct*)opBase;

	// set distinct expressions offsets
	for(uint i = 0; i < op->offset_count; i++) {
		bool aware = OpBase_Aware(opBase, op->aliases[i], op->offsets+i);
		ASSERT(aware == true);
	}

	return OP_OK;
}

static Record DistinctConsume
(
	OpBase *opBase
) {
	OpDistinct *op = (OpDistinct *)opBase;
	OpBase *child = op->op.children[0];

	// as long as there's data to consume
	// try to produce a unique record
	while(true) {
		Record r = OpBase_Consume(child);
		if(!r) return NULL;

		// compute distinct hash
		unsigned long long const hash = _compute_hash(op, r);

		// determine if we've seen this hash before
		int is_new = HashTableAddRaw(op->found, (void *)hash, NULL) != NULL;
		if(is_new) return r;

		// record isn't distinct, discard and pull a new record
		OpBase_DeleteRecord(&r);
	}
}

static inline OpBase *DistinctClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_DISTINCT);

	OpDistinct *op = (OpDistinct *)opBase;
	return NewDistinctOp(plan, op->aliases, op->offset_count);
}

static OpResult DistinctReset
(
	OpBase *opBase
) {
	OpDistinct *op = (OpDistinct *)opBase;

	HashTableEmpty(op->found, NULL);

	return OP_OK;
}

static void DistinctFree
(
	OpBase *ctx
) {
	OpDistinct *op = (OpDistinct *)ctx;

	if(op->found) {
		HashTableRelease(op->found);
		op->found = NULL;
	}

	if(op->aliases) {
		rm_free(op->aliases);
		op->aliases = NULL;
	}

	if(op->offsets) {
		rm_free(op->offsets);
		op->offsets = NULL;
	}
}

