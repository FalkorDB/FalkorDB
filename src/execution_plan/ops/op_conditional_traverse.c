#include "../../value.h"
#include "../../util/arr.h"
#include "../ops.h"
#include "op_conditional_traverse.h"
#include "shared/traverse_functions.h"

// Setup traversal state for conditional traverse
static bool _CondTraverse_SetupTraversal(OpCondTraverse *op, Record r) {
	if(op == NULL || r == NULL) return false;
	
	// Validate record has required entities
	if(Record_GetEntity(r, op->startNodeIdx) == NULL) return false;
	if(Record_GetEntity(r, op->relationIdx) == NULL) return false;
	if(Record_GetEntity(r, op->endNodeIdx) == NULL) return false;
	
	return true;
}

static Record CondTraverseConsume(OpBase *opBase) {
	OpCondTraverse *op = (OpCondTraverse *)opBase;
	OpBase *child = op->op.children[0];

	// return cached records first
	if(op->output_records != NULL && array_len(op->output_records) > 0) {
		return array_pop(op->output_records);
	}

	Record r = NULL;
	while(true) {
		if(op->iter == NULL) {
			r = OpBase_Consume(child);
			if(r == NULL) return NULL;
			
			// Initialize traversal iterator
			if(!_CondTraverse_SetupTraversal(op, r)) {
				OpBase_DeleteRecord(r);
				continue;
			}
		}
		
		// Consume from iterator and return results
		if(op->iter != NULL) {
			// Iterator will be consumed in next call
			return r;
		}
		
		break;
	}
	
	return r;
}

static OpBase* CondTraverseClone(const ExecutionPlan *plan, const OpBase *opBase) {
	Assert(opBase->type == OPType_CONDITIONAL_TRAVERSE);
	OpCondTraverse *op = (OpCondTraverse *)opBase;
	return (OpBase *)OpCondTraverse_New(plan->graph, op->src, op->relationFilter, op->dest);
}

static void CondTraverseFree(OpBase *opBase) {
	OpCondTraverse *op = (OpCondTraverse *)opBase;
	if(op->output_records) array_free(op->output_records);
	if(op->r) OpBase_DeleteRecord(op->r);
	if(op->relationFilter) FilterNode_Free(op->relationFilter);
}

OpBase *OpCondTraverse_New(Graph *g, Node *src, FilterNode *relationFilter, Node *dest) {
	OpCondTraverse *op = (OpCondTraverse *)malloc(sizeof(OpCondTraverse));
	OpBase_Init((OpBase *)op, OPType_CONDITIONAL_TRAVERSE, "Conditional Traverse", NULL, NULL, CondTraverseConsume, CondTraverseClone, NULL, CondTraverseFree, false, g);
	
	op->src = src;
	op->relationFilter = relationFilter;
	op->dest = dest;
	op->output_records = array_new(Record, 0);
	op->r = NULL;
	op->iter = NULL;
	
	return (OpBase *)op;
}
