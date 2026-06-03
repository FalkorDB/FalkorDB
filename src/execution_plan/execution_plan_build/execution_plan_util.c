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
			EAGER_OP_COUNT, NULL) != NULL;
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
	OpBase *root,         // start lookup from here
	const OPType *types,  // types to match
	uint type_count,      // number of types
	uint *depth           // [optional] depth of returned op
) {
	if (root == NULL) {
		return NULL ;
	}

	// check if root is of one of the specified types
	for (uint i = 0; i < type_count; i++) {
		// root matched, return
		if (root->type == types[i]) {
			if (depth != NULL) {
				*depth = 0 ;
			}
			return root ;
		}
	}

	// continue searching
	for (int i = 0; i < root->childCount; i++) {
		// recursively visit children
		OpBase *op =
			ExecutionPlan_LocateOpMatchingTypes (OpBase_GetChild (root, i),
					types, type_count, depth) ;
		if (op != NULL) {
			if (depth != NULL) {
				*depth += 1 ;
			}
			return op ;
		}
	}

	return NULL ;
}

OpBase *ExecutionPlan_LocateOp
(
    OpBase *root,
    OPType type
) {
	if(!root) return NULL;

	const OPType type_arr[1] = {type};
	return ExecutionPlan_LocateOpMatchingTypes(root, type_arr, 1, NULL);
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
	int n = raxSize(refs_to_resolve);
	char **references = (char**)raxKeys(refs_to_resolve);

	OpBase *op = _LocateOpResolvingAliases(root, (const char**)references, n,
			blacklisted_ops, nblacklisted_ops);

	arr_free_cb(references, rm_free);

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
			arr_append(*ops, root);
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
	OpBase **ops = arr_new(OpBase *, 0);
	_ExecutionPlan_CollectOpsMatchingTypes(root, types, type_count, &ops);
	return ops;
}

OpBase **ExecutionPlan_CollectOps
(
    OpBase *root,
    OPType type
) {
	OpBase **ops = arr_new(OpBase *, 0);
	_ExecutionPlan_CollectOpsMatchingTypes(root, &type, 1, &ops);
	return ops;
}

// performs a DFS traversal of the execution plan rooted at `root`,
// collecting all leaf operations — those with no children
// these are called "taps" because they are the entry points where
// data flows into the plan (e.g. scan operations)
//
// returns a heap-allocated array of leaf OpBase pointers (may be empty),
// or NULL if `root` is NULL
// the caller is responsible for freeing the returned array via arr_free()
OpBase **ExecutionPlan_CollectTaps
(
	OpBase *root  // root of the execution plan subtree to scan
) {
	if (root == NULL) {
		return NULL ;
	}

	OpBase **taps  = arr_new (OpBase*, 1) ;
	OpBase **stack = arr_new (OpBase*, 1) ;

	arr_append (stack, root) ;

	while (arr_len (stack) > 0) {
		OpBase *op = arr_pop (stack) ;
		uint child_count = OpBase_ChildCount (op) ;

		if (child_count == 0) {
			arr_append (taps, op) ;
			continue ;
		}

		for (uint i = 0 ; i < child_count ; i++) {
			arr_append (stack, OpBase_GetChild (op, i)) ;
		}
	}

	arr_free (stack) ;
	return taps ;
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
	const OpBase *op,           // operation to start collection from
	rax *modifiers,             // [output] collected modifiers
	const ExecutionPlan *plan   // scoped plan
) {
	// validations
	ASSERT (modifiers != NULL) ;

	if (op == NULL) {
		return ;
	}

	// TODO: switch from rax to dict,
	// TODO: see if we can simply return op's awareness?
	dictIterator it ;
	dictEntry    *de ;
	HashTableInitIterator (&it, op->awareness) ;
	while ((de = HashTableNext (&it)) != NULL) {
		char *key = HashTableGetKey (de) ;
		raxInsert (modifiers, (unsigned char *)key, strlen (key), (void *)key,
				NULL) ;
	}
}

