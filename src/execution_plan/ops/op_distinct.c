/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_distinct.h"
#include "op_project.h"
#include "op_aggregate.h"
#include "xxhash.h"
#include "../../util/arr.h"
#include "../execution_plan_build/execution_plan_modify.h"

// forward declarations
static void DistinctFree(OpBase *opBase);
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
		// update the hash state with the current value.
		SIValue_HashUpdate(v, &state);
	}

	// finalize the hash
	unsigned long long const hash = XXH64_digest(&state);
	return hash;
}

// compute record offset to distinct values
static void _updateOffsets
(
	OpDistinct *op,
	Record r
) {
	ASSERT(op->aliases != NULL);
	ASSERT(op->offsets != NULL);

	for(uint i = 0; i < op->offset_count; i++) {
		uint offset = Record_GetEntryIdx(r, op->aliases[i], strlen(op->aliases[i]));
		ASSERT(offset != INVALID_INDEX);
		op->offsets[i] = offset;
	}
}

OpBase *NewDistinctOp
(
	const ExecutionPlan *plan,
	const char **aliases,
	uint alias_count
) {
	ASSERT(aliases != NULL);
	ASSERT(alias_count > 0);

	OpDistinct *op = rm_malloc(sizeof(OpDistinct));

	op->mapping      = NULL;
	op->aliases      = rm_malloc(alias_count * sizeof(const char *));
	op->offset_count = alias_count;
	op->offsets      = rm_calloc(op->offset_count, sizeof(uint));
	op->found        = hashmap_new_with_redis_allocator(0, 0, 0, 0, NULL, NULL,
			NULL, NULL);

	// copy aliases into heap array managed by this op
	memcpy(op->aliases, aliases, alias_count * sizeof(const char *));

	OpBase_Init((OpBase *)op, OPType_DISTINCT, "Distinct", NULL, DistinctConsume,
				DistinctReset, NULL, DistinctClone, DistinctFree, false, plan);

	return (OpBase *)op;
}

static Record DistinctConsume
(
	OpBase *opBase
) {
	OpDistinct *op = (OpDistinct *)opBase;
	OpBase *child = op->op.children[0];

	while(true) {
		Record r = OpBase_Consume(child);
		if(!r) return NULL;

		// update offsets if record mapping changed
		// it is possible for the record's mapping to be changed throughout
		// the execution as this distinct operation might receive records from
		// different sub execution plans, such as in the case of UNION
		// in which case the distinct values might be located at different offsets
		// within the record and we should adjust accordingly
		rax *record_mapping = Record_GetMappings(r);
		if(record_mapping != op->mapping) {
			// record mapping changed, update offsets
			_updateOffsets(op, r);
			// update operation mapping to records mapping
			op->mapping = record_mapping;
		}

		unsigned long long const hash = _compute_hash(op, r);
		int is_new = hashmap_set_with_hash(op->found, NULL, hash) == NULL;
		if(is_new) return r;
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

	if(op->found) {
		hashmap_clear(op->found, true);
	}

	return OP_OK;
}

static void DistinctFree
(
	OpBase *ctx
) {
	OpDistinct *op = (OpDistinct *)ctx;
	if(op->found) {
		hashmap_free(op->found);
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

