/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../ast/ast_shared.h"
#include "../../datatypes/array.h"
#include "../../graph/query_graph.h"
#include "../ops/op_cond_var_len_traverse.h"
#include "../../filter_tree/filter_tree.h"
#include "../../graph/graphcontext.h"
#include "../execution_plan_build/execution_plan_util.h"

// foldVariableLengthEdgeRelationFilter
//
// folds relationship-type predicates that a variable-length traversal applies
// to its traversed edge (op->ft) into the traversal's relationship-type set
// (op->edgeRelationTypes), so excluded types are never expanded and no
// per-edge type() check is evaluated
//
// e.g. MATCH (a)-[e*0..2]->(b) WHERE type(e) <> 'R' AND type(e) <> 'Z'
// folds the two predicates into "traverse every type except R and Z"
//
// each recognised predicate is intersected into a running relation-type set,
// so contradictions collapse naturally: type(e)='R' AND type(e)='Z' => {}
//
// only PURE relationship-type predicates are folded (type(e) = / <> / IN,
// combined with AND / OR); a predicate mixed with anything else (e.g.
// type(e)='R' OR e.v>2) is left untouched in op->ft, since restricting the
// traversal to 'R' would wrongly drop edges that satisfy the other branch
//
// this is a run-time optimization: a negation folds against the current number
// of relationship-types, which can grow over time

// returns true if 'exp' is the expression type(<edge>)
static bool _isEdgeType
(
	const AR_ExpNode *exp,  // expression to inspect
	const char *edge        // traversed edge alias
) {
	if(exp->type != AR_EXP_OP)                    return false;
	if(exp->op.f == NULL || exp->op.f->name == NULL) return false;
	if(strcasecmp(exp->op.f->name, "type") != 0)  return false;
	if(exp->op.child_count != 1)                  return false;

	const AR_ExpNode *arg = exp->op.children[0];
	if(arg->type != AR_EXP_OPERAND)               return false;
	if(arg->operand.type != AR_EXP_VARIADIC)      return false;

	return strcmp(arg->operand.variadic.entity_alias, edge) == 0;
}

// resolve a relationship-type name to its id, or GRAPH_UNKNOWN_RELATION
static RelationID _relID
(
	GraphContext *gc,  // graph context
	const char *name   // relationship-type name
) {
	Schema *s = GraphContext_GetSchema(gc, name, SCHEMA_EDGE);
	return (s != NULL) ? Schema_GetID(s) : GRAPH_UNKNOWN_RELATION;
}

// reduce 'exp' (a constant or a parameter, e.g. $REL) to a scalar and map it
// to a relationship-type id: a string naming an existing type resolves to its
// id, anything else (non-existent name, non-string value) stays
// GRAPH_UNKNOWN_RELATION. returns false only when 'exp' can't be reduced to a
// scalar (e.g. it references a bound entity), in which case we can't fold
static bool _reduceRelID
(
	AR_ExpNode *exp,   // value expression (rhs/lhs of the predicate)
	GraphContext *gc,  // graph context
	RelationID *id     // [output] resolved id, or GRAPH_UNKNOWN_RELATION
) {
	*id = GRAPH_UNKNOWN_RELATION;

	SIValue v;
	if(!AR_EXP_ReduceToScalar(exp, true, &v)) return false;

	if(SI_TYPE(v) == T_STRING) *id = _relID(gc, v.stringval);
	return true;
}

// mark in 'set' every relationship-type named by a string element of the array
// 'arr'; non-string and non-existent elements are skipped (a string type can
// never equal them), so a list of only such elements yields an empty set
static void _addArrayTypes
(
	SIValue arr,       // constant array value
	GraphContext *gc,  // graph context
	int n,             // domain size
	bool *set          // [output] set to mark into, size n
) {
	uint32_t len = SIArray_Length(arr);
	for(uint32_t i = 0; i < len; i++) {
		SIValue elem = SIArray_Get(arr, i);
		if(SI_TYPE(elem) != T_STRING) continue;
		RelationID id = _relID(gc, elem.stringval);
		if(id >= 0 && id < n) set[id] = true;
	}
}

// recursively evaluate an arithmetic expression that is expected to be a pure
// edge relationship-type predicate built from in() / not(); on success fills
// 'set' (size 'n') with the satisfying relationship-type ids and returns true
//
// handles the expression forms a filter tree keeps as FT_N_EXP nodes:
//   type(e) IN [...]      => in(type(e), [...])
//   NOT type(e) IN [...]  => not(in(type(e), [...]))   (post De-Morgan)
static bool _expTypeSet
(
	const AR_ExpNode *exp,  // expression to evaluate
	const char *edge,       // traversed edge alias
	GraphContext *gc,       // graph context
	int n,                  // number of relationship-types (domain size)
	bool *set               // [output] satisfying set, size n
) {
	if(exp->type != AR_EXP_OP)                       return false;
	if(exp->op.f == NULL || exp->op.f->name == NULL) return false;

	const char *fname = exp->op.f->name;

	if(strcasecmp(fname, "in") == 0) {
		// in(type(edge), <string array; constant or parameter, e.g. $RELS>)
		if(exp->op.child_count != 2)               return false;
		if(!_isEdgeType(exp->op.children[0], edge)) return false;

		SIValue arr;
		if(!AR_EXP_ReduceToScalar(exp->op.children[1], true, &arr)) return false;
		if(SI_TYPE(arr) != T_ARRAY) return false;

		memset(set, 0, n * sizeof(bool));
		_addArrayTypes(arr, gc, n, set);
		return true;
	}

	if(strcasecmp(fname, "not") == 0) {
		// not(<pure type expression>) => complement
		if(exp->op.child_count != 1) return false;

		bool inner[n];
		if(!_expTypeSet(exp->op.children[0], edge, gc, n, inner)) return false;
		for(int i = 0; i < n; i++) set[i] = !inner[i];
		return true;
	}

	return false;
}

// recursively determine whether 'node' is a pure edge relationship-type
// predicate expression; on success fills 'set' (size 'n') with the
// relationship-type ids that satisfy it and returns true
static bool _typeSet
(
	const FT_FilterNode *node,  // filter node to evaluate
	const char *edge,           // traversed edge alias
	GraphContext *gc,           // graph context
	int n,                      // number of relationship-types (domain size)
	bool *set                   // [output] satisfying set, size n
) {
	switch(node->t) {
		case FT_N_PRED: {
			const FT_PredicateNode *p = &node->pred;

			// identify type(edge) on one side of the predicate; the other side
			// is the value to compare against (a constant or a parameter)
			AR_ExpNode *val = NULL;
			if(_isEdgeType(p->lhs, edge))      val = p->rhs;
			else if(_isEdgeType(p->rhs, edge)) val = p->lhs;
			else return false;  // not a type predicate

			// only equality / inequality are relationship-type constraints
			// (IN arrives as an FT_N_EXP, handled below)
			if(p->op != OP_EQUAL && p->op != OP_NEQUAL) return false;

			RelationID id;
			if(!_reduceRelID(val, gc, &id)) return false;  // dynamic, can't fold

			// a non-string / non-existent value leaves id == UNKNOWN, which the
			// range check below turns into: '=' matches nothing, '<>' matches
			// everything
			if(p->op == OP_EQUAL) {
				memset(set, 0, n * sizeof(bool));   // only edges of this type
				if(id >= 0 && id < n) set[id] = true;
			} else {
				memset(set, 1, n * sizeof(bool));   // every type but this one
				if(id >= 0 && id < n) set[id] = false;
			}
			return true;
		}

		case FT_N_COND: {
			if(node->cond.op != OP_AND && node->cond.op != OP_OR) return false;

			bool left[n];
			bool right[n];
			if(!_typeSet(node->cond.left,  edge, gc, n, left))  return false;
			if(!_typeSet(node->cond.right, edge, gc, n, right)) return false;

			if(node->cond.op == OP_AND) {
				for(int i = 0; i < n; i++) set[i] = left[i] && right[i];
			} else {
				for(int i = 0; i < n; i++) set[i] = left[i] || right[i];
			}
			return true;
		}

		case FT_N_EXP:
			// e.g. type(e) IN [...] / NOT type(e) IN [...]
			return _expTypeSet(node->exp.exp, edge, gc, n, set);

		default:
			return false;
	}
}

// fold relationship-type predicates of a single variable-length traversal
static void _foldOp
(
	const ExecutionPlan *plan,  // owning plan
	CondVarLenTraverse *op      // traversal to fold into
) {
	// nothing to fold, or already resolved
	if(op->ft == NULL || op->edgeRelationTypes != NULL) return;

	GraphContext *gc = QueryCtx_GetGraphCtx();
	int n = Graph_RelationTypeCount(GraphContext_GetGraph(gc));
	if(n == 0) return;  // no relationship-types, nothing to traverse or fold

	const char *edge = AlgebraicExpression_Edge(op->ae);
	QGEdge     *e    = QueryGraph_GetEdgeByAlias(plan->query_graph, edge);

	// seed the allowed set with the edge's declared types, or every type
	// when the edge is untyped
	bool allowed[n];
	uint declared = QGEdge_RelationCount(e);
	if(declared == 0) {
		for(int i = 0; i < n; i++) allowed[i] = true;
	} else {
		for(int i = 0; i < n; i++) allowed[i] = false;
		for(uint i = 0; i < declared; i++) {
			RelationID id = QGEdge_RelationID(e, i);
			if(id == GRAPH_UNKNOWN_RELATION) id = _relID(gc, QGEdge_Relation(e, i));
			if(id >= 0 && id < n) allowed[id] = true;
		}
	}

	// break the edge filter into its AND-conjuncts (OR-subtrees stay whole)
	const FT_FilterNode **conjuncts = FilterTree_SubTrees(op->ft);
	uint c = arr_len(conjuncts);

	const FT_FilterNode **residual = arr_new(const FT_FilterNode *, c);
	bool folded = false;
	bool set[n];

	for(uint i = 0; i < c; i++) {
		if(_typeSet(conjuncts[i], edge, gc, n, set)) {
			// intersect the conjunct's satisfying set into the allowed set
			for(int k = 0; k < n; k++) allowed[k] = allowed[k] && set[k];
			folded = true;
		} else {
			// keep as a residual per-edge filter
			arr_append(residual, conjuncts[i]);
		}
	}

	if(folded) {
		// materialize the folded relationship-type set
		// (may be empty => the traversal yields no edges)
		RelationID *types = arr_new(RelationID, n);
		for(int k = 0; k < n; k++) {
			if(allowed[k]) arr_append(types, (RelationID)k);
		}
		op->edgeRelationTypes = types;
		op->edgeRelationCount = arr_len(types);

		// rebuild op->ft from the residual conjuncts (Combine clones them)
		FT_FilterNode *ft = (arr_len(residual) > 0)
			? FilterTree_Combine(residual, arr_len(residual))
			: NULL;

		FilterTree_Free(op->ft);
		op->ft = ft;
	}

	arr_free(conjuncts);
	arr_free(residual);
}

void foldVariableLengthEdgeRelationFilter
(
	ExecutionPlan *plan  // plan to optimize
) {
	ASSERT(plan != NULL);

	const OPType types[] = {OPType_CONDITIONAL_VAR_LEN_TRAVERSE,
							OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO};

	OpBase **ops =
		ExecutionPlan_CollectOpsMatchingTypes(plan->root, types, 2);

	uint count = arr_len(ops);
	for(uint i = 0; i < count; i++) {
		_foldOp(plan, (CondVarLenTraverse *)ops[i]);
	}

	arr_free(ops);
}
