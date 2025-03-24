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

// default reset function for operations
// does nothing
static OpResult _OpBase_reset_noop
(
	OpBase *op
) {
	ASSERT(op != NULL);
	return OP_OK;
}

// defualt init function for operations
// does nothing
static OpResult _OpBase_init_noop
(
	OpBase *op
) {
	ASSERT(op != NULL);
	return OP_OK;
}

// before the fist call to consume is made, we need to initialize the operation
// operation initializion is done lazily right before the fist invocation
// further invocation go stright to the operation's consume function
static Record _InitialConsume
(
	OpBase *op  // operation to initialize and consume from
) {
	// validations
	ASSERT(op           != NULL);
	ASSERT(op->init     != NULL);
	ASSERT(op->_consume != NULL);
	ASSERT(op->consume  == _InitialConsume);

	// first and ONLY call to operation initialization
	op->init(op);

	// overwrite op's initial consume WRAPPER function (this one)
	// with the op's original consume func
	op->consume = op->_consume;

	// run consume
	return op->consume(op);
}

// initialize operation
void OpBase_Init
(
	OpBase *op,
	OPType type,
	const char *name,
	fpInit init,
	fpConsume consume,
	fpReset reset,
	fpToString toString,
	fpClone clone,
	fpFree free,
	bool writer,
	const struct ExecutionPlan *plan
) {
	op->type       = type;
	op->name       = name;
	op->plan       = plan;
	op->stats      = NULL;
	op->parent     = NULL;
	op->writer     = writer;
	op->modifies   = NULL;
	op->children   = NULL;
	op->childCount = 0;

	// set op's function pointers
	op->free     = free;
	op->clone    = clone;
	op->consume  = _InitialConsume;  // initial consume wrapper function
	op->_consume = consume;          // op's consume function
	op->toString = toString;
	op->awareness = HashTableCreate(&string_dt);

	op->init  = (init)  ? init  : _OpBase_init_noop;
	op->reset = (reset) ? reset : _OpBase_reset_noop;
}

inline Record OpBase_Consume
(
	OpBase *op
) {
	ASSERT(op != NULL);

	return op->consume(op);
}

// returns true if operation is aware of alias
bool OpBase_Aware
(
	const OpBase *op,  // op
	const char *alias  // alias
) {
	return (HashTableFind(op->awareness, alias) != NULL);
}

// mark alias as being modified by operation
// returns the ID associated with alias
int OpBase_Modifies
(
	OpBase *op,
	const char *alias
) {
	if(!op->modifies) {
		op->modifies = array_new(const char *, 1);
	}

	array_append(op->modifies, alias);

	// make sure alias has an entry associated with it
	// within the record mapping
	rax *mapping = ExecutionPlan_GetMappings(op->plan);

	void *id = raxFind(mapping, (unsigned char *)alias, strlen(alias));
	if(id == raxNotFound) {
		id = (void *)raxSize(mapping);
		raxInsert(mapping, (unsigned char *)alias, strlen(alias), id, NULL);
	}

	// add alias to op's awareness table
	HashTableAdd(op->awareness, (void*)alias, NULL);

	return (intptr_t)id;
}

// adds an alias to an existing modifier
// such that record[modifier] = record[alias]
int OpBase_AliasModifier
(
	OpBase *op,            // op
	const char *modifier,  // existing alias
	const char *alias      // new alias
) {
	rax *mapping = ExecutionPlan_GetMappings(op->plan);
	void *id = raxFind(mapping, (unsigned char *)modifier, strlen(modifier));
	ASSERT(id != raxNotFound);

	// make sure to not introduce the same modifier twice
	if(raxInsert(mapping, (unsigned char *)alias, strlen(alias), id, NULL)) {
		array_append(op->modifies, alias);
	}

	return (intptr_t)id;
}

bool OpBase_ChildrenAware
(
	const OpBase *op,
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
						void *rec_idx = raxFind(mapping, (unsigned char *)alias, strlen(alias));
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

// returns true if alias is mapped
bool OpBase_AliasMapping
(
	const OpBase *op,   // op
	const char *alias,  // alias
	int *idx            // alias map id
) {
	ASSERT(op    != NULL);
	ASSERT(alias != NULL);

	rax *mapping  = ExecutionPlan_GetMappings(op->plan);
	void *rec_idx = raxFind(mapping, (unsigned char *)alias, strlen(alias));

	if(idx != NULL) {
		*idx = (intptr_t)rec_idx;
	}

	return (rec_idx != raxNotFound);
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

// profile function
// used to profile an operation consume function
Record OpBase_Profile
(
	OpBase *op
) {
	double tic [2];
	// start timer
	simple_tic(tic);

	// call op's consume function
	Record r = op->_consume(op);

	// stop timer and accumulate
	op->stats->profileExecTime += simple_toc(tic);

	if(r) op->stats->profileRecordCount++;
	return r;
}

bool OpBase_IsWriter
(
	const OpBase *op
) {
	return op->writer;
}

// indicates if the operation is an eager operation
bool OpBase_IsEager
(
	const OpBase *op
) {
	ASSERT(op != NULL);

	for(int i = 0; i < EAGER_OP_COUNT; i++) {
		if(op->type == EAGER_OPERATIONS[i]) return true;
	}

	return false;
}

void OpBase_UpdateConsume
(
	OpBase *op,
	fpConsume consume
) {
	ASSERT(op != NULL);

	// update both consume and backup consume function
	op->consume  = consume;  // in case update performed within op consume
	op->_consume = consume;  // in case update performed within op init
}

// updates the plan of an operation
void OpBase_BindOpToPlan
(
	OpBase *op,
	const struct ExecutionPlan *plan
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

	// increase r's ref count and set r as clone's parent
	r->ref_count++;
	clone->parent = r;

	return clone;
}

inline OPType OpBase_Type
(
	const OpBase *op
) {
	ASSERT(op != NULL);
	return op->type;
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
	const OpBase *op,  // op
	uint i             // child index
) {
	ASSERT(op != NULL);
	ASSERT(i < op->childCount);
	return op->children[i];
}

inline void OpBase_DeleteRecord
(
	Record *r
) {
	ASSERT(r != NULL);

	if(unlikely(*r == NULL)) return;

	ExecutionPlan_ReturnRecord((*r)->owner, *r);
	// nullify record
	*r = NULL;
}

// merge src into dest and deletes src
void OpBase_MergeRecords
(
	Record dest,  // entries are merged into this record
	Record *src   // entries are merged from this record
) {
	ASSERT(dest != NULL);
	ASSERT(src  != NULL && *src != NULL);
	ASSERT(dest != *src);

	Record_Merge(dest, *src);
	OpBase_DeleteRecord(src);
}

OpBase *OpBase_Clone
(
	const struct ExecutionPlan *plan,
	const OpBase *op
) {
	if(op->clone) return op->clone(plan, op);
	return NULL;
}

void OpBase_Free
(
	OpBase *op
) {
	// free internal operation
	if(op->free)     op->free(op);
	if(op->children) rm_free(op->children);
	if(op->modifies) array_free(op->modifies);
	if(op->stats)    rm_free(op->stats);

	HashTableRelease(op->awareness);
	rm_free(op);
}

