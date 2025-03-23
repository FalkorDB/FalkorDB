/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../util/dict.h"
#include "execution_plan_awareness.h"

// extract keys from hashtable
static void HashTableKeys
(
	unsigned long n,    // number of keys to extract
	const char **keys,  // [output] keys array
	dict *d             // hashtable
) {
	ASSERT(n    > 0);
	ASSERT(keys != NULL);
	ASSERT(n    == HashTableElemCount(d));

	// collect variables
	dictIterator it;
	HashTableInitIterator(&it, d);

	for(int i = 0; i < n; i++) {
		dictEntry *de = HashTableNext(&it);
		const char *key = (const char*)HashTableGetKey(de);
		keys[i] = key;
	}
}

// compute op's awareness by inspecting its children awareness
static void inheritAwareness
(
	const OpBase *op
) {
	ASSERT(op != NULL);

	// update op's awareness
	int n = OpBase_ChildCount(op);
	for(int i = 0; i < n; i++) {
		OpBase *child = OpBase_GetChild(op, i);

		// do not access operations from a different plan
		if(child->plan != op->plan) {
			continue;
		}

		dictEntry *de;
		dictIterator it;
		dict *awareness = child->awareness;
		HashTableInitIterator(&it, awareness);

		while((de = HashTableNext(&it)) != NULL) {
			const char *key = (const char*)HashTableGetKey(de);
			HashTableAdd(op->awareness, (void*)key, NULL);
		}
	}
}

// set op's awareness to only its modifiers
void ExecutionPlanAwareness_SelfAware
(
	OpBase *op  // op to update
) {
	ASSERT(op            != NULL);
	ASSERT(op->awareness != NULL);

	HashTableEmpty(op->awareness, NULL);

	int n = array_len(op->modifies);
	for(int i = 0; i < n; i++) {
		const char *alias = op->modifies[i];
		HashTableAdd(op->awareness, (void*)alias, NULL);
	}
}

// propagate op's awareness downward throughout the parent chain
void ExecutionPlanAwareness_PropagateAwareness
(
	const OpBase *op  // op to propagate awareness from
) {
	ASSERT(op != NULL);

	unsigned long n = HashTableElemCount(op->awareness);
	if(n == 0) {
		return;
	}

	const char *keys[n];
	HashTableKeys(n, keys, op->awareness);

	// update parent awareness
	// do not cross to a different execution-plan
	OpBase *parent = op->parent;
	while(parent != NULL && parent->plan == op->plan) {
		// break if no new aliases were added to the awareness table
		bool short_circuit = true;

		for(int i = 0; i < n; i++) {
			const char *alias = keys[i];
			short_circuit &=
				HashTableAdd(parent->awareness, (void*)alias, NULL) == DICT_ERR;
		}

		// in case current op didn't changed parent awareness we can break
		if(short_circuit) {
			break;
		}

		parent = parent->parent;
	}
}

// update execution plan awareness due to the addition of an operation
// when an operation is added by a call to 'ExecutionPlan_AddOp'
// we need to op's awareness to each parent operation
void ExecutionPlanAwareness_AddOp
(
	const OpBase *op
) {
	ASSERT(op != NULL);

	inheritAwareness(op);
	ExecutionPlanAwareness_PropagateAwareness(op);
}

// update execution plan awareness due to the removal of an entire branch
// rooted at op
// when a branch is removed by a call to 'ExecutionPlan_DetachOp'
// we need to remove all variables the detached branch is aware of
// from each parent operation
void ExecutionPlanAwareness_RemoveAwareness
(
	const OpBase *root  // branch root
) {
	// TODO: might need to ref count the aliased, drop an alias only when
	// its count reaches 0
	//ASSERT("consider CP where both branches introduce the same aliases" && false);
	ASSERT(root != NULL);

	dict *awareness = root->awareness;
	unsigned long n = HashTableElemCount(awareness);
	
	// op isn't aware of any variables, it has no effect on awareness
	if(n == 0) {
		return;
	}

	const char *variables[n];
	HashTableKeys(n, variables, awareness);

	// remove variables from each parent
	// do not cross to a different plan
	OpBase *parent = root->parent;
	while(parent != NULL && parent->plan == root->plan) {
		for(int i = 0; i < n; i++) {
			const char *var = variables[i];
			HashTableDelete(parent->awareness, var);
		}
		parent = parent->parent;
	}
}

// update execution plan awareness due to the removal of an operation
// when an operation is removed by a call to 'ExecutionPlan_RemoveOp'
// we need to remove each of the op's modifiers (aliases introduced by the op)
// from each parent operation
void ExecutionPlanAwareness_RemoveOp
(
	const OpBase *op  // removed op
) {
	ASSERT(op != NULL);

	int n = array_len(op->modifies);

	// op doesn't introduce any variabels, it has no effect on awareness
	if(n == 0) {
		return;
	}

	// remove modifiers from each parent
	// do not cross to a different plan
	OpBase *parent = op->parent;
	while(parent != NULL && parent->plan == op->plan) {
		for(int i = 0; i < n; i++) {
			const char *alias = op->modifies[i];
			// TODO: might need to ref count the aliased, drop an alias only when
			// its count reaches 0
			//ASSERT("consider CP where both branches introduce the same aliases" && false);
			HashTableDelete(parent->awareness, alias);
		}
		parent = parent->parent;
	}
}

