/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../util/arr.h"
#include "../ops/op_filter.h"
#include "../ops/op_all_node_scan.h"
#include "../ops/op_node_by_id_seek.h"
#include "../ops/op_node_by_label_scan.h"
#include "../../util/range/numeric_range.h"
#include "../../arithmetic/arithmetic_op.h"
#include "../../arithmetic/arithmetic_expression.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// the seek by ID optimization searches for a SCAN operation on which
// a filter of the form ID(n) = X is applied in which case
// both the SCAN and FILTER operations can be reduced into a single
// NODE_BY_ID_SEEK operation
// X may be any expression independent of 'n', e.g. a constant, a parameter,
// a variable resolved upstream, or a compound expression such as 'ID(a) + 1';
// it is evaluated against the incoming record at runtime

// returns true if 'exp' is exactly ID(<scanned_alias>) - an id() call whose
// single argument is the scanned entity itself (a bare variadic operand), and
// not an attribute or other expression such as ID(n.v).
//
// the optimization REPLACES the ID(...) side with a range seek and discards
// that expression, so accepting anything other than the scanned node here
// would silently drop the real predicate (e.g. ID(n.v) = 5 would be executed
// as ID(n) IN {5}, returning wrong rows instead of evaluating ID(n.v))
static bool _is_id_call
(
	const AR_ExpNode *exp,
	const char *scanned_alias
) {
	if(exp->type          != AR_EXP_OP ||
	   exp->op.child_count != 1        ||
	   strcasecmp(exp->op.f->name, "id") != 0) {
		return false;
	}

	// the id() argument must be the scanned entity itself: a bare variadic
	// operand whose alias matches the scan's alias (a property access such as
	// n.v is an AR_EXP_OP, not a variadic, so it is rejected here)
	const AR_ExpNode *arg = exp->op.children[0];
	return arg->type         == AR_EXP_OPERAND  &&
		   arg->operand.type == AR_EXP_VARIADIC &&
		   strcmp(arg->operand.variadic.entity_alias, scanned_alias) == 0;
}

// returns true if 'alias' is referenced anywhere within 'exp'
static bool _references_alias
(
	AR_ExpNode *exp,    // expression to inspect
	const char *alias   // alias to search for
) {
	rax *entities = raxNew();
	AR_EXP_CollectEntities(exp, entities);

	bool res = raxFind(entities, (unsigned char *)alias, strlen(alias))
		!= raxNotFound;

	raxFree(entities);
	return res;
}

// a filter qualifies for the seek-by-id optimization when it is a predicate
// of the form:  ID(n) <op> expr   (or  expr <op> ID(n))
// where 'n' is the node resolved by the scan being optimized and 'expr'
// does not reference 'n', so it can be evaluated from the incoming record
static bool _idFilter
(
	FT_FilterNode *f,           // filter to inspect
	const char *scanned_alias,  // alias of the node resolved by the scan
	AST_Operator *rel,          // out: comparison op, oriented as ID(n) <op> expr
	AR_ExpNode **val_exp,       // out: expression ID(n) is compared against
	bool *reverse               // out: true if ID(n) was on the right hand side
) {
	if(f->t       != FT_N_PRED) return false;
	if(f->pred.op == OP_NEQUAL) return false;

	AR_ExpNode *lhs = f->pred.lhs;
	AR_ExpNode *rhs = f->pred.rhs;
	*rel = f->pred.op;

	// determine on which side the scanned node is referenced
	bool on_lhs = _references_alias(lhs, scanned_alias);
	bool on_rhs = _references_alias(rhs, scanned_alias);

	// the scanned node must be referenced on exactly one side
	// reject 'ID(n) = n.v' (both sides) and filters unrelated to 'n' (neither)
	if(on_lhs == on_rhs) return false;

	AR_ExpNode *id_side  = on_lhs ? lhs : rhs;
	AR_ExpNode *val_side = on_lhs ? rhs : lhs;

	// the side describing the scanned node must be exactly ID(n), where n is
	// the scanned entity itself - rejects e.g. 'ID(n) + 1 = 5', 'n.v = 5' and
	// 'ID(n.v) = 5' (id of an attribute rather than the scanned node)
	if(!_is_id_call(id_side, scanned_alias)) return false;

	// 'val_side' is guaranteed not to reference the scanned node,
	// hence it is resolvable from the incoming record
	*val_exp = val_side;
	*reverse = on_rhs;

	return true;
}

static void _UseIdOptimization
(
	ExecutionPlan *plan,
	OpBase *scan_op
) {
	// alias of the node resolved by this scan
	const char *alias;
	if(scan_op->type == OPType_NODE_BY_LABEL_SCAN) {
		alias = ((NodeByLabelScan *)scan_op)->n->alias;
	} else {
		alias = ((AllNodeScan *)scan_op)->alias;
	}

	// see if there's a filter of the form
	// ID(n) op X
	// where X is an expression independent of 'n' and op in [EQ, GE, LE, GT, LT]
	OpBase *grandparent;
	OpBase *parent = scan_op->parent;
	RangeExpression *ranges = arr_new(RangeExpression, 1);
	while(parent && parent->type == OPType_FILTER) {
		// track the next op to visit in case we free parent
		grandparent = parent->parent;
		OpFilter *filter = (OpFilter *)parent;
		FT_FilterNode *f = filter->filterTree;

		bool         reverse;
		AR_ExpNode  *val_exp;
		AST_Operator op;

		if(_idFilter(f, alias, &op, &val_exp, &reverse)) {
			if(reverse) op = ArithmeticOp_ReverseOp(op);
			val_exp = AR_EXP_Clone(val_exp);

			// Free replaced operations.
			ExecutionPlan_RemoveOp(plan, (OpBase *)filter);
			arr_append(ranges, ((RangeExpression){.op = op, .exp = val_exp}));
			OpBase_Free((OpBase *)filter);
		}
		// advance
		parent = grandparent;
	}
	if(arr_len(ranges) > 0) {
		/* Don't replace label scan, but set it to have range query.
		 * Issue 818 https://github.com/RedisGraph/RedisGraph/issues/818
		 * This optimization caused a range query over the entire range of ids in the graph
		 * regardless to the label. */
		if(scan_op->type == OPType_NODE_BY_LABEL_SCAN) {
			NodeByLabelScan *label_scan = (NodeByLabelScan *) scan_op;
			NodeByLabelScanOp_SetIDRange(label_scan, ranges);
		} else {
			OpBase *opNodeByIdSeek = NewNodeByIdSeekOp(scan_op->plan, alias, ranges);

			// Managed to reduce!
			ExecutionPlan_ReplaceOp(plan, scan_op, opNodeByIdSeek);
			OpBase_Free(scan_op);
		}
	} else {
		arr_free(ranges);
	}
}

void seekByID
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	const OPType types[] = {OPType_ALL_NODE_SCAN, OPType_NODE_BY_LABEL_SCAN};
	OpBase **scan_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root, types, 2);

	for(int i = 0; i < arr_len(scan_ops); i++) {
		_UseIdOptimization(plan, scan_ops[i]);
	}

	arr_free(scan_ops);
}

