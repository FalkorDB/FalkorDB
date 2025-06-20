/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../util/arr.h"
#include "../ops/op_skip.h"
#include "../../util/dict.h"
#include "../ops/op_limit.h"
#include "execution_plan_util.h"

// returns true if an operation in the op-tree rooted at `root` is eager
bool ExecutionPlan_isEager
(
    OpBase *root
) {
	return ExecutionPlan_LocateOpMatchingTypes(root, EAGER_OPERATIONS,
			EAGER_OP_COUNT, NULL, 0) != NULL;
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

// traverse upwards as long as an operation that resolves all aliases is found
// returns NULL if all aliases are not resolved
static OpBase *_LocateOpResolvingAliases
(
    OpBase *root,                   // root
    const char **aliases,           // aliases to locate
	int n,                          // number of aliases
	const OPType *blacklisted_ops,  // blacklisted operations
	int nblacklisted_ops            // number of blacklisted operations
) {
	ASSERT(n       >  0);
	ASSERT(root    != NULL);
	ASSERT(aliases != NULL);

	if(!OpBase_Aware(root, aliases, n)) {
		// early return if root isn't aware of alias
		return NULL;
	}

	// don't venture into blacklisted ops
	if(_blacklisted(root, blacklisted_ops, nblacklisted_ops)) {
		return root;
	}

	OpBase *ret = root;
	const ExecutionPlan *plan = root->plan;

	// search for a child who's aware of the alias
	// prefer 'left' children
	while(true) {
		bool new_ret = false;

		// scan each child of ret in the hope of finding a child that is
		// aware of all aliases
		for(int i = 0; i < ret->childCount; i++) {
			OpBase *child = OpBase_GetChild(ret, i);

			// do not cross execution-plan boundries
			if(child->plan != plan ||
			   _blacklisted(child, blacklisted_ops, nblacklisted_ops)) {
				continue;
			}

			// see if current child is aware of all aliases
			// update 'ret' and break if child is aware of all aliases
			new_ret = OpBase_Aware(child, aliases, n);
			if(new_ret) {
				ret = child;
				break;
			}
		}

		// return if we did not found a child which is aware of the alias
		if(!new_ret) {
			break;
		}
	}

	return ret;
}

// locate the first operation matching one of the given types in the op tree by
// performing DFS
// returns NULL if no matching operation was found
OpBase *ExecutionPlan_LocateOpMatchingTypes
(
    OpBase *root,         // search starts here
    const OPType *types,  // types to match
    uint type_count,      // number of types
	OPType *blacklist,    // [optional] list of ops search won't expand from
	uint blacklist_n      // number of blacklisted ops
) {
	// return current op if it matches any of the types we're searching for
	for(int i = 0; i < type_count; i++) {
		if(root->type == types[i]) return root;
	}

	// scan through root children
	for(int i = 0; i < root->childCount; i++) {
		OpBase *child = OpBase_GetChild(root, i);

		// make sure child isn't blacklisted
		bool blacklisted = false;
		for(int j = 0; j < blacklist_n; j++) {
			if(child->type == blacklist[j]) {
				blacklisted = true;
				break;
			}
		}

		if(blacklisted) {
			// child is blacklisted move on to the next child
			continue;
		}

		// recursively visit child
		OpBase *op = ExecutionPlan_LocateOpMatchingTypes(child, types,
				type_count, blacklist, blacklist_n);

		if(op) {
			return op;
		}
	}

	return NULL;
}

OpBase *ExecutionPlan_LocateOp
(
    OpBase *root,
    OPType type
) {
	if(!root) return NULL;

	return ExecutionPlan_LocateOpMatchingTypes(root, &type, 1, NULL, 0);
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

// find the earliest operation at which all references are resolved, if any,
// without recursing past a blacklisted op
OpBase *ExecutionPlan_LocateReferencesExcludingOps
(
	OpBase *root,                   // start point
	const OPType *blacklisted_ops,  // blacklisted operations
	int nblacklisted_ops,           // number of blacklisted operations
	rax *refs_to_resolve            // references to resolve
) {
	// locate earliest op under which all references are resolved
	OpBase *ret = NULL;
	int n = raxSize(refs_to_resolve);
	char **references = (char**)raxKeys(refs_to_resolve);

	OpBase *op = _LocateOpResolvingAliases(root, (const char**)references, n,
			blacklisted_ops, nblacklisted_ops);

	array_free_cb(references, rm_free);

	return op;
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

// collect all operations belonging to the given plan
// it is the callers responsibility to free the returned array
OpBase **ExecutionPlan_CollectAllOps
(
	const ExecutionPlan *plan,  // plan to collect ops from
	uint *n                     // [output] number of ops collected
) {
	ASSERT(n    != NULL);
	ASSERT(plan != NULL);

	*n = 0;

	// can't scan plan, return NULL
	if(plan->root == NULL) {
		return NULL;
	}

	OpBase **ops = rm_malloc(sizeof(OpBase*) * 1);

	uint s = 0;  // current index into ops array
	uint e = 1;  // length of ops array
	ops[0] = plan->root;

	// collect upwards using child pointers
	while(s < e) {
		OpBase *current = ops[s++];

		// try adding current children
		for(uint i = 0; i < OpBase_ChildCount(current); i++) {
			OpBase *child = OpBase_GetChild(current, i);

			// add child only if it belong to given plan
			if(child->plan == plan) {
				ops = rm_realloc(ops, sizeof(OpBase*) * (e + 1));
				ops[e++] = child;
			}
		}
	}

	// no need to collect downwards
	OpBase *current = plan->root->parent;
	ASSERT(current == NULL || current->plan != plan);

	*n = e;
	return ops;
}

// fills `ops` with all operations from `op` an upward (towards parent) in the
// execution plan
// returns the amount of ops collected
uint ExecutionPlan_CollectUpwards
(
    OpBase **restrict ops,  // array to populate
	uint n,                 // size of ops array
    OpBase *restrict root   // root operation to scan from
) {
	ASSERT(n    > 0);
	ASSERT(ops  != NULL);
	ASSERT(root != NULL);

	uint i = 0;
	while(root != NULL) {
		ops[i] = root;
		root = root->parent;
		i++;
	}

	ASSERT(i <= n);  // make sure we didn't overflow
	return i;
}

// collect all aliases that have been resolved by the given tree of operations
void ExecutionPlan_BoundVariables
(
	const OpBase *op,           // operation to start collection from
	rax *modifiers,             // [output] collected modifiers
	const ExecutionPlan *plan   // scoped plan
) {
	// validations
	ASSERT(op        != NULL);
	ASSERT(modifiers != NULL);

	// TODO: switch from rax to dict,
	// TODO: see if we can simply return op's awareness?
	dictIterator it;
	dictEntry    *de;
	HashTableInitIterator(&it, op->awareness);
	while((de = HashTableNext(&it)) != NULL) {
		char *key = HashTableGetKey(de);
		raxInsert(modifiers, (unsigned char *)key, strlen(key), (void *)key,
				NULL);
	}
}

