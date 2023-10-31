/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op.h"
#include "RG.h"
#include "op_project.h"
#include "op_aggregate.h"
#include "../../util/rmalloc.h"
#include "../../util/simple_timer.h"

// forward declarations
Record ExecutionPlan_BorrowRecord(struct ExecutionPlan *plan);
rax *ExecutionPlan_GetMappings(const struct ExecutionPlan *plan);
void ExecutionPlan_ReturnRecord(const struct ExecutionPlan *plan, Record r);

//------------------------------------------------------------------------------
// hashtable callbacks
//------------------------------------------------------------------------------

// fake hash function
// hash of key is simply key
static uint64_t _aware_id_hash
(
	const void *key
) {
	return ((uint64_t)key);
}

static int _aware_key_cmp
(
    dict *d,
	const void *key1,
	const void *key2
) {
	return strcmp(key1, key2) == 0;
}

dictType awareness_dt = {_aware_id_hash, NULL, NULL, _aware_key_cmp, NULL, NULL,
	NULL, NULL, NULL, NULL};

static void _OpBase_StatsToString
(
	const OpBase *op,
	sds *buff
) {
	*buff = sdscatprintf(*buff,
					" | Records produced: %d, Execution time: %f ms",
					op->stats->profileRecordCount,
					op->stats->profileExecTime);
}

// collects writing operations under `op` into `write_ops`, and resets the
// reading ops (including `op` itself)
static void _OpBase_PropagateReset
(
	OpBase *op,
	OpBase ***write_ops
) {
	if(op->reset) {
		if(OpBase_IsWriter(op)) {
			array_append(*write_ops, op);
		} else {
			OpResult res = op->reset(op);
			ASSERT(res == OP_OK);
		}
	}

	// recursively reset children
	for(int i = 0; i < op->childCount; i++) {
		_OpBase_PropagateReset(op->children[i], write_ops);
	}
}

// default reset function for operations
OpResult _OpBase_reset_noop
(
	OpBase *op
) {
	ASSERT(op != NULL);
	return OP_OK;
}

// initialize op
void OpBase_Init
(
	OpBase *op,                 // op to initialize
	OPType type,                // op type
	const char *name,           // op name
	fpInit init,                // op's init function
	fpConsume consume,          // op's consume function
	fpReset reset,              // op's reset function
	fpToString toString,        // op's toString function
	fpClone clone,              // op's clone function
	fpFree free,                // op's free function
	bool writer,  			    // writer indicator
	struct ExecutionPlan *plan  // op's execution plan
) {
	op->type       = type;
	op->name       = name;
	op->plan       = plan;
	op->stats      = NULL;
	op->parent     = NULL;
	op->parent     = NULL;
	op->aware      = HashTableCreate(&awareness_dt);
	op->writer     = writer;
	op->modifies   = NULL;
	op->children   = NULL;
	op->childCount = 0;

	// function pointers
	op->init     = init;
	op->free     = free;
	op->clone    = clone;
	op->reset    = (reset) ? reset : _OpBase_reset_noop;
	op->profile  = NULL;
	op->consume  = consume;
	op->toString = toString;
}

inline Record OpBase_Consume
(
	OpBase *op
) {
	return op->consume(op);
}

Record OpBase_Profile
(
	OpBase *op
) {
	double tic [2];
	// Start timer.
	simple_tic(tic);
	Record r = op->profile(op);
	// Stop timer and accumulate.
	op->stats->profileExecTime += simple_toc(tic);
	if(r) op->stats->profileRecordCount++;
	return r;
}

void OpBase_ToString
(
	const OpBase *op,
	sds *buff
) {
	int bytes_written = 0;

	if(op->toString) op->toString(op, buff);
	else *buff = sdscatprintf(*buff, "%s", op->name);

	if(op->stats) _OpBase_StatsToString(op, buff);
}

OpBase *OpBase_Clone
(
	struct ExecutionPlan *plan,
	const OpBase *op
) {
	ASSERT(op        != NULL);
	ASSERT(plan      != NULL);
	ASSERT(op->clone != NULL);

	return op->clone(plan, op);
}

inline OPType OpBase_Type
(
	const OpBase *op
) {
	ASSERT(op != NULL);
	return op->type;
}

// sets child's parent
void OpBase_SetParent
(
	OpBase *child,  // child to set parent for
	OpBase *parent  // parent to set
) {
	ASSERT(child         != NULL);
	ASSERT(child         != parent);
	ASSERT(child->parent != parent);

#ifdef RG_DEBUG
	if(parent == NULL) {
		// detaching child from parent
		// make sure child was removed from parent's children list
		OpBase *old_parent = child->parent;
		ASSERT(OpBase_LocateChild(old_parent, child, NULL) == false);
	} else {
		// attaching child to parent
		// make sure child is in parent's children list
		ASSERT(OpBase_LocateChild(parent, child, NULL) == true);
	}
#endif

	child->parent = parent;
}

// add child
void OpBase_AddChild
(
	OpBase *parent,  // parent op
	OpBase *child    // new child op
) {
	ASSERT(child         != NULL);
	ASSERT(parent        != NULL);
	ASSERT(child         != parent);
	ASSERT(child->parent == NULL);

	// make sure child isn't in parent's children list
	ASSERT(OpBase_LocateChild(parent, child, NULL) == false);

	// make room for child
	if(parent->children == NULL) {
		parent->children = rm_malloc(sizeof(OpBase *));
	} else {
		parent->children = rm_realloc(parent->children,
				sizeof(OpBase *) * (parent->childCount + 1));
	}

	// add as the last child
	parent->children[parent->childCount++] = child;

	// attach child to parent
	OpBase_SetParent(child, parent);
}

// add child at specific index
void OpBase_AddChildAt
(
	OpBase *parent,  // parent op
	OpBase *child,   // child op
	uint idx         // index of op
) {
	ASSERT(child         != NULL);
	ASSERT(parent        != NULL);
	ASSERT(child->parent == NULL);
	ASSERT(idx < parent->childCount);
	ASSERT(OpBase_LocateChild(parent, child, NULL) == false);

	// replace child at idx with new child
	// readd replaced child to the end of parent's children array
	OpBase *replaced_child = parent->children[idx];
	ASSERT(replaced_child != child);
	replaced_child->parent = NULL;  // temporarily detach from parent

	parent->children[idx] = child;
	OpBase_SetParent(child, parent);
	OpBase_AddChild(parent, replaced_child);
}

// remove child from parent
void OpBase_RemoveChild
(
	OpBase *op,            // parent op
	OpBase *child,         // child to remove
	bool inharit_children  // if true, child's children will be inharited
) {
	ASSERT(op     != NULL);
	ASSERT(child  != NULL);
	ASSERT(child->parent == op);

	//--------------------------------------------------------------------------
	// locate child in op's children array
	//--------------------------------------------------------------------------

	int i = 0;
	bool found = OpBase_LocateChild(op, child, &i);
	ASSERT(found == true);

	if(inharit_children && child->childCount > 0) {
		//----------------------------------------------------------------------
		// inharit child's children
		//----------------------------------------------------------------------

		// make room for child's children
		size_t n = op->childCount - 1 + child->childCount;
		OpBase **children = rm_malloc(sizeof(OpBase *) * n);

		// copy children before i
		for(int j = 0; j < i; j++) {
			children[j] = op->children[j];
		}
		// copy child's children
		for(int j = 0; j < child->childCount; j++) {
			children[i + j] = child->children[j];
		}
		// copy remaining children
		for(int j = i+1; j < op->childCount; j++) {
			children[i + child->childCount] = op->children[j];
		}

		// update parent's children
		rm_free(op->children);
		op->children   = children;
		op->childCount = n;

		// clear child's children
		rm_free(child->children);
		child->children   = NULL;
		child->childCount = 0;
	} else {
		// no children to inharit
		if(op->childCount == 1) {
			// op has no children
			rm_free(op->children);
			op->children   = NULL;
			op->childCount = 0;
		} else {
			// shift left children
			for(int j = i; j < op->childCount-1; j++) {
				op->children[j] = op->children[j + 1];
			}
			// realloc children array
			op->childCount--;
			op->children = rm_realloc(op->children,
					sizeof(OpBase *) * (op->childCount));
		}
	}

	// child in now orphan
	OpBase_SetParent(child, NULL);
}

// locate child in parent's children array
// returns true if child was found, false otherwise
// sets 'idx' to the index of child in parent's children array
bool OpBase_LocateChild
(
	const OpBase *parent,  // parent op
	const OpBase *child,   // child op to locate
	int *idx               // [optional out] index of child in parent
) {
	ASSERT(parent != NULL);
	ASSERT(child  != NULL);

	// scan child in parent's children array
	for(int i = 0; i < parent->childCount; i++) {
		if(parent->children[i] == child) {
			// set index if requested
			if(idx) {
				*idx = i;
			}
			return true;
		}
	}

	return false;
}

// returns the number of children of the op
inline uint OpBase_ChildCount
(
	const OpBase *op
) {
	ASSERT(op != NULL);
	return op->childCount;
}

// returns the i'th child of the op
OpBase *OpBase_GetChild
(
	OpBase *op,  // op
	uint i       // child index
) {
	ASSERT(op != NULL);
	ASSERT(i < op->childCount);
	return op->children[i];
}

// mark alias as being modified by operation
// returns the ID associated with alias
int OpBase_Modifies
(
	OpBase *op,
	const char *alias
) {
	ASSERT(op    != NULL);
	ASSERT(alias != NULL);

	// create modifies array if it doesn't exist
	if(!op->modifies) op->modifies = array_new(const char *, 1);

	// add alias to both modifiers array and awareness hash table
	array_append(op->modifies, alias);
	HashTableAdd(op->aware, (void*)alias, NULL);

	// let plan know this operation resolves alias
	ExecutionPlan_AddResolver(op->plan, alias, op);

	return ExecutionPlan_AddMappings(op->plan, alias);
}

// returns op's modifiers
const char **OpBase_GetModifiers
(
	const OpBase *op,  // op to get modifiers from
	int *n             // number of modifiers
) {
	ASSERT(n  != NULL);
	ASSERT(op != NULL);

	if(op->modifies != NULL) *n = array_len(op->modifies);

	return op->modifies;
}

bool OpBase_ChildrenAware
(
	OpBase *op,
	const char *alias,
	int *idx
) {
	for (int i = 0; i < op->childCount; i++) {
		OpBase *child = op->children[i];
		if(op->plan == child->plan && child->modifies != NULL) {
			uint count = array_len(child->modifies);
			for (uint i = 0; i < count; i++) {
				if(strcmp(alias, child->modifies[i]) == 0) {
					if(idx) {
						rax *mapping = ExecutionPlan_GetMappings(op->plan);
						void *rec_idx = raxFind(mapping, (unsigned char *)alias,
								strlen(alias));
						*idx = (intptr_t)rec_idx;
					}
					return true;				
				}
			}
		}
		if(OpBase_ChildrenAware(child, alias, idx)) return true;
	}
	
	return false;
}

bool OpBase_Aware
(
	OpBase *op,
	const char *alias,
	int *idx
) {
	rax *mapping = ExecutionPlan_GetMappings(op->plan);
	void *rec_idx = raxFind(mapping, (unsigned char *)alias, strlen(alias));
	if(idx) *idx = (intptr_t)rec_idx;
	return (rec_idx != raxNotFound);
}

// computes op awareness
// an op is aware of all alises its children are aware of in addition to
// aliases it modifies
void OpBase_ComputeAwareness
(
	OpBase *op  // op to compute awareness for
) {
	ASSERT(op != NULL);

	if(op->aware != NULL) {
		// free previous awareness
		HashTableRelease(op->aware);
		op->aware = NULL;
	}

	op->aware = HashTableCreate(&awareness_dt);

	// collect children awareness
	for(int i = 0; i < op->childCount; i++) {
		OpBase *child = op->children[i];

		dictIterator iter;
		dictEntry    *entry;

		HashTableInitIterator(&iter, child->aware);
		while((entry = HashTableNext(&iter))) {
			char *id = (char*)HashTableGetKey(entry);
			HashTableAdd(op->aware, (void*)id, NULL);
		}
	}

	// add op's modifiers to its awareness
	int n = array_len(op->modifies);
	for(int i = 0; i < n; i++) {
		const char *id = op->modifies[i];
		HashTableAdd(op->aware, (void*)id, NULL);
	}
}

void OpBase_PropagateReset
(
	OpBase *op
) {
	// hold write operations until the read operations have been reset
	OpBase **write_ops = array_new(OpBase *, 0);

	// reset read operations
	_OpBase_PropagateReset(op, &write_ops);

	// reset write operations
	uint write_op_count = array_len(write_ops);
	for(uint i = 0; i < write_op_count; i++) {
		OpBase *write_op = write_ops[i];
		OpResult res = write_op->reset(write_op);
		ASSERT(res == OP_OK);
	}

	array_free(write_ops);
}

bool OpBase_IsWriter
(
	OpBase *op
) {
	return op->writer;
}

void OpBase_UpdateConsume
(
	OpBase *op,
	fpConsume consume
) {
	ASSERT(op != NULL);
	// if Operation is profiled, update profiled function
	// otherwise update consume function
	if(op->profile != NULL) op->profile = consume;
	else op->consume = consume;
}

// updates the plan of an operation
void OpBase_BindOpToPlan
(
	OpBase *op,
	struct ExecutionPlan *plan
) {
	ASSERT(op != NULL);

	OPType type = OpBase_Type(op);
	if(type == OPType_PROJECT) {
		ProjectBindToPlan(op, plan);
	} else if(type == OPType_AGGREGATE) {
		AggregateBindToPlan(op, plan);
	} else {
		op->plan = plan;
	}
}

inline Record OpBase_CreateRecord
(
	const OpBase *op
) {
	return ExecutionPlan_BorrowRecord((struct ExecutionPlan *)op->plan);
}

Record OpBase_CloneRecord
(
	Record r
) {
	Record clone = ExecutionPlan_BorrowRecord((struct ExecutionPlan *)r->owner);
	Record_Clone(r, clone);
	return clone;
}

Record OpBase_DeepCloneRecord
(
	Record r
) {
	Record clone = ExecutionPlan_BorrowRecord((struct ExecutionPlan *)r->owner);
	Record_DeepClone(r, clone);
	return clone;
}

inline void OpBase_DeleteRecord
(
	Record r
) {
	ExecutionPlan_ReturnRecord(r->owner, r);
}

void OpBase_Free
(
	OpBase *op
) {
	// free internal operation
	if(op->free)     op->free(op);
	if(op->stats)    rm_free(op->stats);
	if(op->children) rm_free(op->children);
	if(op->modifies) array_free(op->modifies);
	HashTableRelease(op->aware);
	rm_free(op);
}

