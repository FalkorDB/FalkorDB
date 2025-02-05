/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_plan_util.h"
#include "../ops/op_skip.h"
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
	if(!root) return NULL;

	uint count = array_len(root->modifies);

	for(uint i = 0; i < count; i++) {
		const char *resolved_alias = root->modifies[i];
		// NOTE - if this function is later used to modify the returned
		// operation, we should return the deepest operation that modifies the
		// alias rather than the shallowest, as done here
		if(strcmp(resolved_alias, alias) == 0) return root;
	}

	for(int i = 0; i < root->childCount; i++) {
		OpBase *op = ExecutionPlan_LocateOpResolvingAlias(root->children[i], alias);
		if(op) return op;
	}

	return NULL;
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

OpBase *ExecutionPlan_LocateReferencesExcludingOps
(
	OpBase *root,
	const OpBase *recurse_limit,
	const OPType *blacklisted_ops,
	int nblacklisted_ops,
	rax *refs_to_resolve
) {
	int dependency_count = 0;
	bool blacklisted = false;
	OpBase *resolving_op = NULL;
	bool all_refs_resolved = false;

	// check if this op is blacklisted
	for(int i = 0; i < nblacklisted_ops && !blacklisted; i++) {
		blacklisted = (root->type == blacklisted_ops[i]);
	}

	// we're not allowed to inspect child operations of blacklisted ops
	// also we're not allowed to venture further than 'recurse_limit'
	if(blacklisted == false && root != recurse_limit) {
		for(int i = 0; i < root->childCount && !all_refs_resolved; i++) {
			// Visit each child and try to resolve references, storing a pointer to the child if successful.
			OpBase *tmp_op = ExecutionPlan_LocateReferencesExcludingOps(root->children[i],
																		recurse_limit, blacklisted_ops, nblacklisted_ops, refs_to_resolve);

			// Count how many children resolved references
			if(tmp_op) {
				dependency_count ++;
			}
			// If there is more than one child resolving an op, set the root as the resolver.
			resolving_op = resolving_op ? root : tmp_op;
			all_refs_resolved = (raxSize(refs_to_resolve) == 0); // We're done when the rax is empty.
		}
	}

	// If we've resolved all references, our work is done.
	if(all_refs_resolved) return resolving_op;

	char **modifies = NULL;
	if(blacklisted) {
		// If we've reached a blacklisted op, all variables in its subtree are
		// considered to be modified by it, as we can't recurse farther.
		rax *bound_vars = raxNew();
		ExecutionPlan_BoundVariables(root, bound_vars, root->plan);
		modifies = (char **)raxKeys(bound_vars);
		raxFree(bound_vars);
	} else {
		modifies = (char **)root->modifies;
	}

	// Try to resolve references in the current operation.
	bool refs_resolved = false;
	uint modifies_count = array_len(modifies);
	for(uint i = 0; i < modifies_count; i++) {
		const char *ref = modifies[i];
		// Attempt to remove the current op's references, marking whether any removal was succesful.
		refs_resolved |= raxRemove(refs_to_resolve, (unsigned char *)ref, strlen(ref), NULL);
	}

	// Free the modified array and its contents if it was generated to represent a blacklisted op.
	if(blacklisted) {
		for(uint i = 0; i < modifies_count; i++) rm_free(modifies[i]);
		array_free(modifies);
	}

	if(refs_resolved) resolving_op = root;
	return resolving_op;
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
			raxTryInsert(modifiers, (unsigned char *)modified, strlen(modified), (void *)modified, NULL);
		}
	}

	// Project and Aggregate operations demarcate variable scopes,
	// collect their projections but do not recurse into their children.
	// Note that future optimizations which operate across scopes will require different logic
	// than this for application
	if(op->type == OPType_PROJECT || op->type == OPType_AGGREGATE) return;

	for(int i = 0; i < op->childCount; i++) {
		ExecutionPlan_BoundVariables(op->children[i], modifiers, plan);
	}
}
