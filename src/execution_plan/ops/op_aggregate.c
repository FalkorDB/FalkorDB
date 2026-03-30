/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_aggregate.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/rmalloc.h"

// forward declarations
static void AggregateFree(OpBase *opBase);
static Record AggregateConsume(OpBase *opBase);
static OpResult AggregateReset(OpBase *opBase);
static OpBase *AggregateClone(const ExecutionPlan *plan, const OpBase *opBase);

// fake hash function
// hash of key is simply key
static uint64_t _id_hash
(
	const void *key
) {
	return ((uint64_t)key);
}

// hashtable entry free callback
static void freeCallback
(
	dict *d,
	void *val
) {
	Group_Free((Group*)val);
}

// hashtable callbacks
static dictType _dt = {_id_hash, NULL, NULL, NULL, NULL, freeCallback, NULL,
	NULL, NULL, NULL};

static void _collect_entities_outside_agg
(
	AR_ExpNode *exp,
	rax *entities
) {
	if(exp->type == AR_EXP_OPERAND) {
		if(exp->operand.type == AR_EXP_VARIADIC) {
			const char *alias = exp->operand.variadic.entity_alias;
			if(alias) {
				raxInsert(entities, (unsigned char*)alias,
					strlen(alias), NULL, NULL);
			}
		}
		return;
	}

	// stop at agg function boundaries — their children are consumed
	// during AR_EXP_Aggregate, not during finalization
	if(AR_EXP_IsAggregation(exp)) {
		return;
	}

	for(uint i = 0; i < exp->op.child_count; i++) {
		_collect_entities_outside_agg(exp->op.children[i], entities);
	}
}

// migrate each expression projected by this operation to either
// the array of keys or the array of aggregate functions as appropriate
static void _migrate_expressions
(
	OpAggregate *op,
	AR_ExpNode **exps
) {
	uint exp_count = array_len(exps);
	op->key_exps = array_new(AR_ExpNode *, 0);
	op->aggregate_exps = array_new(AR_ExpNode *, 1);

	for(uint i = 0; i < exp_count; i++) {
		AR_ExpNode *exp = exps[i];
		if(!AR_EXP_ContainsAggregation(exp)) {
			array_append(op->key_exps, exp);
		} else {
			array_append(op->aggregate_exps, exp);
		}
	}

	op->key_count       = array_len(op->key_exps);
	op->aggregate_count = array_len(op->aggregate_exps);

	// collect aliases referenced inside aggregate expressions
	// that are not already covered by key expressions
	// e.g. in { statement: l.value, facts: collect(f.value) }
	// 'l' is not a key but must survive into the representative record
	rax *key_aliases = raxNew();
	for(uint i = 0; i < op->key_count; i++) {
		AR_ExpNode *kexp = op->key_exps[i];
		// only treat as a resolved key if the expression IS a direct variadic
		// (plain alias like `t`), not a map projection containing entities
		if(!AR_EXP_IsOperation(kexp) &&
		kexp->operand.type == AR_EXP_VARIADIC) {
			const char *alias = kexp->operand.variadic.entity_alias;
			raxInsert(key_aliases, (unsigned char *)alias, strlen(alias), NULL, NULL);
		}
	}

	rax *seen = raxNew();
	op->mixed_aliases = array_new(char *, 0);

	for(uint i = 0; i < op->aggregate_count; i++) {
		rax *entities = raxNew();
		_collect_entities_outside_agg(op->aggregate_exps[i], entities);

		raxIterator it;
		raxStart(&it, entities);
		raxSeek(&it, "^", NULL, 0);
		while(raxNext(&it)) {
			if(raxFind(key_aliases, it.key, it.key_len) == raxNotFound &&
			raxFind(seen,       it.key, it.key_len) == raxNotFound) {
				char *alias = rm_malloc(it.key_len + 1);
				memcpy(alias, it.key, it.key_len);
				alias[it.key_len] = '\0';
				array_append(op->mixed_aliases, alias);
				raxInsert(seen, it.key, it.key_len, NULL, NULL);
			}
		}
		raxStop(&it);
		raxFree(entities);
	}

	raxFree(key_aliases);
	raxFree(seen);
	op->mixed_count = array_len(op->mixed_aliases);
}

// clone all aggregate expression templates to associate with a new group
static inline AR_ExpNode **_build_aggregate_exps
(
	OpAggregate *op
) {
	AR_ExpNode **agg_exps =
		rm_malloc(op->aggregate_count * sizeof(AR_ExpNode *));

	for(uint i = 0; i < op->aggregate_count; i++) {
		agg_exps[i] = AR_EXP_Clone(op->aggregate_exps[i]);
	}

	return agg_exps;
}

static Group *_CreateGroup
(
	OpAggregate *op,
	Record r
) {
	// create a new group

	// get a fresh copy of aggregation functions
	AR_ExpNode **agg_exps = _build_aggregate_exps(op);

	return Group_New(agg_exps, op->aggregate_count, r);
}

static XXH64_hash_t _ComputeGroupKey
(
	SIValue *keys,
	OpAggregate *op,
	Record r
) {
	// initialize the hash state
	XXH64_state_t state;
	XXH_errorcode res = XXH64_reset(&state, 0);
	ASSERT(res != XXH_ERROR);

	for(uint i = 0; i < op->key_count; i++) {
		AR_ExpNode *exp = op->key_exps[i];
		// note if AR_EXP_Evaluate throws a runtime exception we will leak
		keys[i] = AR_EXP_Evaluate(exp, r);
		// update the hash state with the current value.
		SIValue_HashUpdate(keys[i], &state);
	}

	// finalize the hash
	return XXH64_digest(&state);
}

// retrieves group under which given record belongs to
// creates group if it doesn't exists
static Group *_GetGroup
(
	OpAggregate *op,
	Record r
) {
	// construct group key
	// evaluate non-aggregated fields

	SIValue keys[op->key_count];
	XXH64_hash_t hash = _ComputeGroupKey(keys, op, r);

	// lookup group by hashed key
	Group *g;
	dictEntry *existing;
	dictEntry *entry = HashTableAddRaw(op->groups, (void *)hash, &existing);

	if(entry == NULL) {
		// group exists
		ASSERT(existing != NULL);

		// free computed keys
		for(uint i = 0; i < op->key_count; i++) {
			SIValue_Free(keys[i]);
		}

		g = HashTableGetVal(existing);
	} else {
		// group does not exists, create it

		// set keys in record
		Record representative = OpBase_CreateRecord((OpBase*)op);
		for(uint i = 0; i < op->key_count; i++) {
			SIType t = keys[i].type;

			if(!(t & SI_GRAPHENTITY)) {
				SIValue_Persist(keys+i);
			}

			Record_Add(representative, op->record_offsets[i], keys[i]);

			if((t & SI_GRAPHENTITY)) {
				SIValue_Free(keys[i]);
			}
		}

		// copy mixed alias values from original record into representative record
		// so they're available when finalizing aggregate expressions
		for(uint i = 0; i < op->mixed_count; i++) {
			const char *alias = op->mixed_aliases[i];
			int src_idx = Record_GetEntryIdx(r, alias, strlen(alias));
			ASSERT(src_idx != INVALID_INDEX && "mixed alias not found in record");
			if(src_idx == INVALID_INDEX) continue;
			SIValue val = Record_Get(r, src_idx);
			SIType t = val.type;
			// graph entities are pointers into graph storage — do NOT deep copy them
			// scalars must be persisted so they survive after r is deleted
			if(!(t & SI_GRAPHENTITY)) {
				val = SI_CloneValue(val);
			}
			int dst_idx = op->record_offsets[op->key_count + op->aggregate_count + i];
			Record_Add(representative, dst_idx, val);
		}

		g = _CreateGroup(op, representative);

		HashTableSetVal(op->groups, entry, g);
	}

	return g;
}

static void _aggregateRecord
(
	OpAggregate *op,
	Record r
) {
	// get group
	Group *g = _GetGroup(op, r);
	ASSERT(g != NULL);

	// aggregate group exps
	for(uint i = 0; i < op->aggregate_count; i++) {
		AR_ExpNode *exp = g->agg[i];
		AR_EXP_Aggregate(exp, r);
	}

	// delete record only if it isn't group representative
	OpBase_DeleteRecord(&r);
}

// returns a record populated with group data
static Record _handoff
(
	OpAggregate *op
) {
	dictEntry *entry = HashTableNext(op->group_iter);
	if(entry == NULL) {
		return NULL;
	}

	Group *g = (Group*)HashTableGetVal(entry);
	Record r = g->r;
	g->r = NULL;

	// compute the final value of all aggregate expressions and add to Record
	for(uint i = 0; i < op->aggregate_count; i++) {
		int rec_idx = op->record_offsets[i + op->key_count];
		AR_ExpNode *exp = g->agg[i];

		SIValue agg = AR_EXP_FinalizeAggregations(exp, r);
		SIValue_Persist(&agg);
		Record_AddScalar(r, rec_idx, agg);
	}

	// free group
	Group_Free(g);
	HashTableSetVal(op->groups, entry, NULL);

	return r;
}

OpBase *NewAggregateOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps
) {
	OpAggregate *op = rm_calloc (1, sizeof(OpAggregate)) ;

	op->groups = HashTableCreate (&_dt) ;

	OpBase_Init((OpBase *)op, OPType_AGGREGATE, "Aggregate", NULL,
			AggregateConsume, AggregateReset, NULL, AggregateClone,
			AggregateFree, false, plan);

	// expand hashtable to 2048 slots
	int res = HashTableExpand(op->groups, 2048);
	ASSERT(res == DICT_OK);

	// migrate each expression to the keys array or
	// the aggregations array as appropriate
	_migrate_expressions(op, exps);
	array_free(exps);

	// the projected record will associate values with their resolved name
	// to ensure that space is allocated for each entry
	op->record_offsets = array_new(uint, op->aggregate_count + op->key_count);
	for(uint i = 0; i < op->key_count; i++) {
		// store the index of each key expression
		int record_idx = OpBase_Modifies((OpBase *)op,
				op->key_exps[i]->resolved_name);
		array_append(op->record_offsets, record_idx);
	}
	for(uint i = 0; i < op->aggregate_count; i++) {
		// store the index of each aggregating expression
		int record_idx = OpBase_Modifies((OpBase *)op,
				op->aggregate_exps[i]->resolved_name);
		array_append(op->record_offsets, record_idx);
	}
	// reserve record slots for mixed aliases
	for(uint i = 0; i < op->mixed_count; i++) {
		int record_idx = OpBase_Modifies((OpBase *)op, op->mixed_aliases[i]);
		array_append(op->record_offsets, record_idx);
	}

	return (OpBase *)op;
}

static Record AggregateConsume
(
	OpBase *opBase
) {
	OpAggregate *op = (OpAggregate *)opBase;
	if(op->group_iter != NULL) {
		return _handoff(op);
	}

	Record r;
	if(op->op.childCount == 0) {
		// RETURN max (1)
		// create a 'fake' record
		r = OpBase_CreateRecord(opBase);
		_aggregateRecord(op, r);
	} else {
		OpBase *child = op->op.children[0];
		// eager consumption!
		while((op->r = OpBase_Consume(child))) {
			_aggregateRecord(op, op->r);
		}
		op->r = NULL;
	}

	// did we process any records?
	// does aggregation contains keys?
	// e.g.
	// MATCH (n:N) WHERE n.noneExisting = 2 RETURN count(n)
	if(HashTableElemCount(op->groups) == 0 && op->key_count == 0) {

		// no data was processed and aggregation doesn't have a key
		// in this case we want to return aggregation default value
		// aggregate on an empty record
		ASSERT(op->op.childCount > 0);

		// use child record
		// this is required in case this aggregation is perford within the
		// context of a WITH projection as we need the child's mapping for
		// expression evaluation
		// there's no harm in doing so when not in a WITH aggregation,
		// as we'll be using the same mapping;
		// this operation and it child are in the same scope
		OpBase *child = op->op.children[0];
		r = OpBase_CreateRecord(child);

		// get group
		_GetGroup(op, r);
	}

	// create group iterator
	op->group_iter = HashTableGetIterator(op->groups);

	return _handoff(op);
}

static OpResult AggregateReset
(
	OpBase *opBase
) {
	OpAggregate *op = (OpAggregate *)opBase;

	if(op->group_iter != NULL) {
		HashTableReleaseIterator(op->group_iter);
		op->group_iter = NULL;
	}

	// re-create hashtable
	unsigned long elem_count = HashTableElemCount(op->groups);
	HashTableRelease(op->groups);

	op->groups = HashTableCreate(&_dt);

	// expand hashtable to previous element count
	int res = HashTableExpand(op->groups, elem_count);
	ASSERT(res == DICT_OK);

	return OP_OK;
}

static OpBase *AggregateClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_AGGREGATE);

	OpAggregate *op = (OpAggregate *)opBase;
	uint key_count = op->key_count;
	uint aggregate_count = op->aggregate_count;
	AR_ExpNode **exps = array_new(AR_ExpNode *, aggregate_count + key_count);

	for(uint i = 0; i < key_count; i++) {
		array_append(exps, AR_EXP_Clone(op->key_exps[i]));
	}

	for(uint i = 0; i < aggregate_count; i++) {
		array_append(exps, AR_EXP_Clone(op->aggregate_exps[i]));
	}

	return NewAggregateOp(plan, exps);
}

// bind the Aggregate operation to the execution plan
void AggregateBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
) {
	OpAggregate *op = (OpAggregate *)opBase;
	opBase->plan = plan;

	// introduce the projected aliases to the plan record-mapping, and reset the
	// record offsets to the correct indexes
	array_clear(op->record_offsets);

	for(uint i = 0; i < op->key_count; i ++) {
		// The projected record will associate values with their resolved name
		// to ensure that space is allocated for each entry.
		int record_idx = OpBase_Modifies((OpBase *)op, op->key_exps[i]->resolved_name);
		array_append(op->record_offsets, record_idx);
	}
	for(uint i = 0; i < op->aggregate_count; i++) {
		// store the index of each aggregating expression
		int record_idx = OpBase_Modifies((OpBase *)op,
				op->aggregate_exps[i]->resolved_name);
		array_append(op->record_offsets, record_idx);
	}
	for(uint i = 0; i < op->mixed_count; i++) {
		int record_idx = OpBase_Modifies((OpBase *)op, op->mixed_aliases[i]);
		array_append(op->record_offsets, record_idx);
	}
}

static void AggregateFree
(
	OpBase *opBase
) {
	OpAggregate *op = (OpAggregate *)opBase;
	if(op == NULL) {
		return;
	}

	if(op->group_iter) {
		HashTableReleaseIterator(op->group_iter);
		op->group_iter = NULL;
	}

	if(op->key_exps) {
		for(uint i = 0; i < op->key_count; i++) {
			AR_EXP_Free(op->key_exps[i]);
		}
		array_free(op->key_exps);
		op->key_exps = NULL;
	}

	if(op->aggregate_exps) {
		for(uint i = 0; i < op->aggregate_count; i++) {
			AR_EXP_Free(op->aggregate_exps[i]);
		}
		array_free(op->aggregate_exps);
		op->aggregate_exps = NULL;
	}

	if(op->groups) {
		HashTableRelease(op->groups);
		op->groups = NULL;
	}

	if(op->record_offsets) {
		array_free(op->record_offsets);
		op->record_offsets = NULL;
	}

	if(op->mixed_aliases) {
		for(uint i = 0; i < op->mixed_count; i++) {
			rm_free(op->mixed_aliases[i]);
		}
		array_free(op->mixed_aliases);
		op->mixed_aliases = NULL;
	}

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}
}

