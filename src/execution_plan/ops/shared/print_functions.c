/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "print_functions.h"
#include "../../execution_plan.h"
#include "../op_node_by_label_scan.h"

static inline void _NodeToString(sds *buf, const char *alias, const char *label) {
	// Print a representation of a node with an optional alias and optional label.
	*buf = sdscatprintf(*buf, "(");
	if(alias) *buf = sdscatprintf(*buf, "%s", alias);
	if(label) *buf = sdscatprintf(*buf, ":%s", label);
	*buf = sdscatprintf(*buf, ")");
}

// return the labels of the src of ae splitted by ':'
static char* ae_labels_src
(
	AlgebraicExpression **ae  // AlgebraicExpression to extract labels from
) {
	sds labels = sdsempty();
	AlgebraicExpression *operand =
		(AlgebraicExpression *) AlgebraicExpression_SrcOperand(*ae);

	// as long as:
	// 1. algebraic expression isn't empty
	// 2. current operand is diagonal
	while(*ae                            &&
		  operand->type == AL_OPERAND &&
		  operand->operand.diagonal) {

		// remove source from expression
		operand = AlgebraicExpression_RemoveSource(ae);

		// update labels string
		labels = sdscatprintf(labels, ":%s", operand->operand.label);

		// free operand
		AlgebraicExpression_Free(operand);

		// advance to next source operand
		if(*ae != NULL) {
			operand =
				(AlgebraicExpression *) AlgebraicExpression_SrcOperand(*ae);
		}
	}

	return labels;
}

// return the labels of the dest of ae splitted by ':'
static char* ae_labels_dest
(
	AlgebraicExpression **ae  // AlgebraicExpression to extract labels from
) {
	sds labels = sdsempty();
	AlgebraicExpression *operand =
		(AlgebraicExpression *) AlgebraicExpression_DestOperand(*ae);

	// as long as:
	// 1. algebraic expression isn't empty
	// 2. current operand is diagonal
	while(*ae                         &&
		  operand->type == AL_OPERAND &&
		  operand->operand.diagonal) {

		// remove destination from expression
		operand = AlgebraicExpression_RemoveDest(ae);

		// update labels string
		labels = sdscatprintf(labels, ":%s", operand->operand.label);

		// free operand
		AlgebraicExpression_Free(operand);

		// advance to next source operand
		if(*ae) {
			operand =
				(AlgebraicExpression *) AlgebraicExpression_DestOperand(*ae);
		}
	}

	return labels;
}

void TraversalToString
(
	const OpBase *op,
	sds *buf,
	const AlgebraicExpression *ae
) {
	ASSERT(op  != NULL);
	ASSERT(ae  != NULL);
	ASSERT(buf != NULL);

	*buf = sdscatprintf(*buf, "%s | ", op->name);

	// this edge should be printed right-to-left if the edge matrix is transposed
	const char *edge       = AlgebraicExpression_Edge(ae);
	const char *src_alias  = AlgebraicExpression_Src(ae);
	const char *dest_alias = AlgebraicExpression_Dest(ae);

	bool transpose  = (edge && AlgebraicExpression_Transposed(ae));
	bool same_alias = !strcmp(src_alias, dest_alias);

	AlgebraicExpression *clone = AlgebraicExpression_Clone(ae);

	//--------------------------------------------------------------------------
	// print source labels
	//--------------------------------------------------------------------------

	*buf = sdscatprintf(*buf, "(%s", src_alias);
	char *src_labels = ae_labels_src(&clone);
	*buf = sdscatprintf(*buf, "%s", src_labels);
	if(!same_alias) {
		sdsfree(src_labels);
	}

	*buf = sdscatprintf(*buf, ")");

	//--------------------------------------------------------------------------
	// print edge
	//--------------------------------------------------------------------------

	if(edge) {
		QGEdge *e = QueryGraph_GetEdgeByAlias(op->plan->query_graph, edge);
		ASSERT(e != NULL);
		if(transpose) {
			*buf = sdscatprintf(*buf, "<-");
			QGEdge_ToString(e, buf);
			*buf = sdscatprintf(*buf, "-");
		} else {
			*buf = sdscatprintf(*buf, "-");
			QGEdge_ToString(e, buf);
			*buf = sdscatprintf(*buf, "->");
		}
	} else {
		*buf = sdscatprintf(*buf, "->");
	}

	//--------------------------------------------------------------------------
	// print dest labels
	//--------------------------------------------------------------------------

	if(same_alias) {
		*buf = sdscatprintf(*buf, "(%s", src_alias);
		*buf = sdscatprintf(*buf, "%s", src_labels);
		sdsfree(src_labels);
	} else {
		*buf = sdscatprintf(*buf, "(%s", AlgebraicExpression_Dest(clone));
		char *dest_labels = ae_labels_dest(&clone);
		*buf = sdscatprintf(*buf, "%s", dest_labels);
		sdsfree(dest_labels);
	}
	*buf = sdscatprintf(*buf, ")");

	// if clone is yet to be free'd, free it
	if(clone != NULL) {
		AlgebraicExpression_Free(clone);
	}
}

void ScanToString(const OpBase *op, sds *buf, const char *alias, const char *label) {
	*buf = sdscatprintf(*buf, "%s | ", op->name);
	_NodeToString(buf, alias, label);
}

