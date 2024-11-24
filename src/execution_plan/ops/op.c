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
OpResult _OpBase_reset_noop
(
	OpBase *op
) {
	ASSERT(op != NULL);
	return OP_OK;
}

Record _OpBase_consume
(
	OpBase *op
) {
	ASSERT(op != NULL);
	op->init(op);
	op->consume = op->op_consume;
	return op->consume(op);
}

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
	op->type           = type;
	op->name           = name;
	op->plan           = plan;
	op->stats          = NULL;
	op->parent         = NULL;
	op->writer         = writer;
	op->modifies       = NULL;
	op->children       = NULL;
	op->childCount     = 0;

	// function pointers
	op->init     = init;
	op->free     = free;
	op->clone    = clone;
	op->reset    = (reset) ? reset : _OpBase_reset_noop;
	op->consume  = init ? _OpBase_consume : consume;
	op->op_consume = consume;
	op->toString = toString;
}

inline Record OpBase_Consume
(
	OpBase *op
) {
	ASSERT(op != NULL);

	return op->consume(op);
}

// mark alias as being modified by operation
// returns the ID associated with alias
int OpBase_Modifies
(
	OpBase *op,
	const char *alias
) {
	if(!op->modifies) op->modifies = array_new(const char *, 1);
	array_append(op->modifies, alias);

	// make sure alias has an entry associated with it
	// within the record mapping
	rax *mapping = ExecutionPlan_GetMappings(op->plan);

	void *id = raxFind(mapping, (unsigned char *)alias, strlen(alias));
	if(id == raxNotFound) {
		id = (void *)raxSize(mapping);
		raxInsert(mapping, (unsigned char *)alias, strlen(alias), id, NULL);
	}

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

bool OpBase_Aware
(
	const OpBase *op,
	const char *alias,
	int *idx
) {
	rax *mapping = ExecutionPlan_GetMappings(op->plan);
	void *rec_idx = raxFind(mapping, (unsigned char *)alias, strlen(alias));
	if(idx) *idx = (intptr_t)rec_idx;
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

Record OpBase_Profile
(
	OpBase *op
) {
	double tic [2];
	// Start timer.
	simple_tic(tic);
	Record r = op->op_consume(op);
	// Stop timer and accumulate.
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
	// if no wrapper consume function is set, set it
	if(op->consume ==  op->op_consume) {
		op->consume = consume;
	}
	op->op_consume = consume;
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
	if(op->free) op->free(op);
	if(op->children) rm_free(op->children);
	if(op->modifies) array_free(op->modifies);
	if(op->stats) rm_free(op->stats);
	rm_free(op);
}

