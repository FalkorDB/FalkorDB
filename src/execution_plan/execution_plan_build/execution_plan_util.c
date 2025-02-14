/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "xxhash.h"
#include "execution_plan_util.h"
#include "../ops/op_skip.h"
#include "../../util/dict.h"
#include "../ops/op_limit.h"

// returns true if an operation in the op-tree rooted at `root` is eager
bool ExecutionPlan_isEager
(
    OpBase *root
) {
	return ExecutionPlan_LocateOpMatchingTypes(root, EAGER_OPERATIONS,
			EAGER_OP_COUNT) != NULL;
}

OpBase *ExecutionPlan_LocateOpResolvingAlias
(
    OpBase *root,
    const char *alias
) {
	ASSERT(root  != NULL);
	ASSERT(alias != NULL);

	OpBase **queue = array_new(OpBase*, 1);
	array_append(queue, root);

	OpBase *ret = NULL;
	while(array_len(queue) > 0) {
		OpBase *op = array_pop(queue);
		uint count = array_len(op->modifies);

		for(uint i = 0; i < count; i++) {
			const char *resolved_alias = op->modifies[i];
			// NOTE - if this function is later used to modify the returned
			// operation, we should return the deepest operation that modifies
			// the alias rather than the shallowest, as done here
			if(strcmp(resolved_alias, alias) == 0) {
				ret = op;
				break;
			}
		}

		for(int i = 0; i < op->childCount; i++) {
			array_append(queue, OpBase_GetChild(op, i));
		}
	}

	array_free(queue);
	return ret;
}

// locate the first operation matching one of the given types in the op tree by
// performing DFS
// returns NULL if no matching operation was found
OpBase *ExecutionPlan_LocateOpMatchingTypes
(
    OpBase *root,
    const OPType *types,
    uint type_count
) {
	for(int i = 0; i < type_count; i++) {
		// Return the current op if it matches any of the types we're searching for.
		if(root->type == types[i]) return root;
	}

	for(int i = 0; i < root->childCount; i++) {
		// Recursively visit children.
		OpBase *op = ExecutionPlan_LocateOpMatchingTypes(root->children[i], types, type_count);
		if(op) return op;
	}

	return NULL;
}

OpBase *ExecutionPlan_LocateOp
(
    OpBase *root,
    OPType type
) {
	if(!root) return NULL;

	const OPType type_arr[1] = {type};
	return ExecutionPlan_LocateOpMatchingTypes(root, type_arr, 1);
}

// searches for an operation of a given type, up to the given depth in the
// execution-plan
OpBase *ExecutionPlan_LocateOpDepth
(
    OpBase *root,
    OPType type,
    uint depth
) {
	if(root == NULL) {
		return NULL;
	}

	if(root->type == type) {
		return root;
	}

	if(depth == 0) {
		return NULL;
	}

	for(int i = 0; i < root->childCount; i++) {
		OpBase *op = ExecutionPlan_LocateOpDepth(root->children[i], type,
			depth - 1);
		if(op) {
			return op;
		}
	}

	return NULL;
}

// returns all operations of a certain type in a execution plan
void ExecutionPlan_LocateOps
(
	OpBase ***plans,  // array in which ops are stored
	OpBase *root,     // root operation of the plan to traverse
	OPType type       // operation type to search
) {
	if(root->type == type) {
		array_append(*plans, root);
	}

	for(uint i = 0; i < root->childCount; i++) {
		ExecutionPlan_LocateOps(plans, root->children[i], type);
	}
}

// checks if op is marked as blacklisted
static inline bool _blacklisted
(
	const OpBase* op,               // operation to inspect
	const OPType *blacklisted_ops,  // list of blacklisted operation types
	int n                           // length of blacklisted_ops
) {
	bool blacklisted = false;

	for(int i = 0; i < n; i++) {
		blacklisted |= (OpBase_Type(op) == blacklisted_ops[i]);
	}

	return blacklisted;
}

// checks to see if op is aware of all references
static inline bool _aware
(
	dict *awareness_tbl,      // awareness table
	const OpBase *op,         // inspected operation
	const char **references,  // references to resolve
	uint n                    // number of refereces
) {
	// get op's awareness table
	dictEntry *entry = HashTableFind(awareness_tbl, (void*)op);
	ASSERT(entry != NULL);

	dict *ht = HashTableGetVal(entry);
	ASSERT(ht != NULL);

	// make sure op resolves all references
	for(uint i = 0; i < n; i++) {
		const char *ref = references[i];
		if(HashTableFind(ht, ref) == NULL) {
			return false;
		}
	}

	return true;
}

uint64_t _hashFunction
(
	const void *key
) {
	return XXH64(key, strlen(key), 0);
}

int _hashCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	const char *a = (const char*)key1;
	const char *b = (const char*)key2;

	return (strcmp(a, b) == 0);
}

OpBase *ExecutionPlan_LocateReferencesExcludingOps
(
	OpBase *root,                   // start point
	const OpBase *recurse_limit,    // boundry
	const OPType *blacklisted_ops,  // blacklisted operations
	int nblacklisted_ops,           // number of blacklisted operations
	rax *refs_to_resolve            // references to resolve
) {
	// compute variabels awareness of each reachable operation from root
	OpBase **taps  = array_new(OpBase*, 1);
	OpBase **queue = array_new(OpBase*, 1);  // operation queue

	// push root into queue and process queue in a BFS fasion
	// head of queue will contain leafs while the tail will contain the root
	uint idx = 0;
	array_append(queue, root);
	while(idx < array_len(queue)) {
		OpBase *op = queue[idx++];
		OPType t = OpBase_Type(op);

		// make sure we're allowed to inspect current op
		if(op != root && (t == OPType_PROJECT || t == OPType_AGGREGATE)) {
			array_append(taps, op);
			continue;
		}

		// add operation's children to queue
		uint n = OpBase_ChildCount(op);
		if(n == 0) {
			array_append(taps, op);
		}

		for(uint i = 0; i < n; i++) {
			array_append(queue, OpBase_GetChild(op, i));
		}
	}

	// compute variabels awareness
	// op's awareness = op's modified variabels + children's awareness
	// hashtable callbacks
	dictType _dt = {
		_hashFunction,  // key hash function
		NULL,           // key dup
		NULL,           // val dup
		_hashCompare,   // key compare
		NULL,           // key destructor
		NULL,           // val destructor
		NULL,           // expand allowed
		NULL,           //
		NULL,           // dict metadata
		NULL            // entry reallocated
	};

	dictIterator it;
	dict *awareness = HashTableCreate(&def_dt);
	uint n = array_len(taps);

	for(uint i = 0; i < n; i++) {
		OpBase *op = taps[i];
		OpBase *prev_op = NULL;

		// traverse downward torwards root using the parent chain
		while(true) {
			// get hashtable for current op
			dict *ht = HashTableFetchValue(awareness, (void*)op);

			// create a new hashtable incase this is the first time
			// we encounter op
			if(ht == NULL) {
				ht = HashTableCreate(&_dt);
				HashTableAdd(awareness, (void*)op, ht);

				// add each modifier to op's awareness table
				const char **modifiers = op->modifies;
				uint nmod = array_len(modifiers);
				for(uint j = 0; j < nmod; j++) {
					HashTableAdd(ht, (void*)modifiers[j], NULL);
				}
			}

			// union with previous op
			if(prev_op != NULL) {
				dict *prev_ht = HashTableFetchValue(awareness, (void*)prev_op);
				ASSERT(prev_ht != NULL);

				// add each child modifier to op's awareness table
				HashTableInitIterator(&it, prev_ht);
				dictEntry *e = NULL;
				while((e = HashTableNext(&it)) != NULL) {
					void *modifier = HashTableGetKey(e);
					HashTableAdd(ht, (void*)modifier, NULL);
				}
			}

			// processed root, stop here
			if(op == root) {
				break;
			}

			// add op's awareness hashtable to the global awareness hashtable
			prev_op = op;
			op = op->parent;
		}
	}

	// locate earliest op under which all references are resolved
	// TODO: optimization make sure root is aware of all references
	//       before we compute the awareness mapping

	OpBase *ret = NULL;
	n = raxSize(refs_to_resolve);
	const char **references = (const char**)raxKeys(refs_to_resolve);

	//ASSERT(array_len(queue) == 0);
	array_clear(queue);
	array_append(queue, root);

	while(array_len(queue) > 0) {
		OpBase *op = array_pop(queue);

		// check if current op is aware of all references
		if(!_aware(awareness, op, references, n)) {
			continue;
		}

		// op is aware of all references, see if one of its children
		// is also aware of all of them

		ret = op;  // set op as the returned operation

		// inspect children
		uint c;
		if(_blacklisted(op, blacklisted_ops, nblacklisted_ops) ||
				op == recurse_limit) {
			c = 0;
		} else {
			c = OpBase_ChildCount(op);
		}

		for(uint i = 0; i < c; i++) {
			array_append(queue, OpBase_GetChild(op, i));
		}
	}

	// clean up
	array_free(taps);
	array_free(queue);
	array_free(references);

	// free each allocated hash-table
	HashTableInitIterator(&it, awareness);
	dictEntry *de = NULL;
	while((de = HashTableNext(&it)) != NULL) {
		dict *d = (dict*)HashTableGetVal(de);
		HashTableRelease(d);
	}
	HashTableRelease(awareness);

	return ret;
}

// scans plan from root via parent nodes until a limit operation is found
// eager operation will terminate the scan
// return true if a limit operation was found, in which case 'limit' is set
// otherwise return false
bool ExecutionPlan_ContainsLimit
(
	OpBase *root,    // root to start the scan from
	uint64_t *limit  // limit value
) {
	ASSERT(root  != NULL);
	ASSERT(limit != NULL);

	while(root != NULL) {
		// halt if we encounter an eager operation
		if(OpBase_IsEager(root)) return false;

		// found a limit operation
		if(root->type == OPType_LIMIT) {
			*limit = ((const OpLimit*)root)->limit;
			return true;
		}

		root = root->parent;
	}

	return false;
}

// scans plan from root via parent nodes until a skip operation is found
// eager operation will terminate the scan
// return true if a skip operation was found, in which case 'skip' is set
// otherwise return false
bool ExecutionPlan_ContainsSkip
(
	OpBase *root,   // root to start the scan from
	uint64_t *skip  // skip value
) {
	ASSERT(root != NULL);
	ASSERT(skip != NULL);

	while(root != NULL) {
		// halt if we encounter an eager operation
		if(OpBase_IsEager(root)) return false;

		// found a skip operation
		if(root->type == OPType_SKIP) {
			*skip = ((const OpSkip*)root)->skip;
			return true;
		}

		root = root->parent;
	}

	return false;
}

OpBase *ExecutionPlan_LocateReferences
(
	OpBase *root,
	const OpBase *recurse_limit,
	rax *refs_to_resolve
) {
	return ExecutionPlan_LocateReferencesExcludingOps(
			   root, recurse_limit, NULL, 0, refs_to_resolve);
}

// populates `ops` with all operations with a type in `types` in an
// execution plan, based at `root`
static void _ExecutionPlan_CollectOpsMatchingTypes
(
	OpBase *root,
	const OPType *types,
	int type_count,
	OpBase ***ops
) {
	for(int i = 0; i < type_count; i++) {
		// check to see if the op's type matches any of the types provided
		if(root->type == types[i]) {
			array_append(*ops, root);
			break;
		}
	}

	for(int i = 0; i < root->childCount; i++) {
		// recursively visit children
		_ExecutionPlan_CollectOpsMatchingTypes(root->children[i], types,
				type_count, ops);
	}
}

// returns an array of all operations with a type in `types` in an
// execution plan, based at `root`
OpBase **ExecutionPlan_CollectOpsMatchingTypes
(
    OpBase *root,
    const OPType *types,
    uint type_count
) {
	OpBase **ops = array_new(OpBase *, 0);
	_ExecutionPlan_CollectOpsMatchingTypes(root, types, type_count, &ops);
	return ops;
}

OpBase **ExecutionPlan_CollectOps
(
    OpBase *root,
    OPType type
) {
	OpBase **ops = array_new(OpBase *, 0);
	_ExecutionPlan_CollectOpsMatchingTypes(root, &type, 1, &ops);
	return ops;
}

// fills `ops` with all operations from `op` an upward (towards parent) in the
// execution plan
// returns the amount of ops collected
uint ExecutionPlan_CollectUpwards
(
    OpBase *ops[],
    OpBase *op
) {
	ASSERT(op != NULL);
	ASSERT(ops != NULL);

	uint i = 0;
	while(op != NULL) {
		ops[i] = op;
		op = op->parent;
		i++;
	}

	return i;
}

// collect all aliases that have been resolved by the given tree of operations
void ExecutionPlan_BoundVariables
(
    const OpBase *op,
    rax *modifiers,
	const ExecutionPlan *plan
) {
	ASSERT(op != NULL && modifiers != NULL);
	if(op->modifies && op->plan == plan) {
		uint modifies_count = array_len(op->modifies);
		for(uint i = 0; i < modifies_count; i++) {
			const char *modified = op->modifies[i];
			raxTryInsert(modifiers, (unsigned char *)modified, strlen(modified),
					(void *)modified, NULL);
		}
	}

	// Project and Aggregate operations demarcate variable scopes
	// collect their projections but do not recurse into their children
	// note that future optimizations which operate across scopes will require
	// different logic than this for application
	if(op->type == OPType_PROJECT || op->type == OPType_AGGREGATE) return;

	for(int i = 0; i < op->childCount; i++) {
		ExecutionPlan_BoundVariables(op->children[i], modifiers, plan);
	}
}
