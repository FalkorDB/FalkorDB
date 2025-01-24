/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_plan.h"
#include "../RG.h"
#include "./ops/ops.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "./optimizations/optimizer.h"
#include "../ast/ast_build_filter_tree.h"
#include "execution_plan_build/execution_plan_modify.h"
#include "execution_plan_build/execution_plan_construct.h"

#include <setjmp.h>

// allocate a new ExecutionPlan segment
inline ExecutionPlan *ExecutionPlan_NewEmptyExecutionPlan(void) {
	return rm_calloc(1, sizeof(ExecutionPlan));
}

void ExecutionPlan_PopulateExecutionPlan(ExecutionPlan *plan) {
	AST *ast = QueryCtx_GetAST();
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// Initialize the plan's record mapping if necessary.
	// It will already be set if this ExecutionPlan has been created to populate a single stream.
	if(plan->record_map == NULL) {
		plan->record_map = raxNew();
	}

	// Build query graph
	// Query graph is set if this ExecutionPlan has been created to populate a single stream.
	if(plan->query_graph == NULL) plan->query_graph = BuildQueryGraph(ast);

	uint clause_count = cypher_ast_query_nclauses(ast->root);
	for(uint i = 0; i < clause_count; i ++) {
		// Build the appropriate operation(s) for each clause in the query.
		const cypher_astnode_t *clause = cypher_ast_query_get_clause(ast->root, i);
		ExecutionPlanSegment_ConvertClause(gc, ast, plan, clause);
	}
}

// build an execution plan composed of several sub-plans
// one for each joint query
// RETURN 1 AS x
// UNION
// RETURN 2 AS x
static ExecutionPlan *_ExecutionPlan_UnionPlans
(
	AST *ast
) {
	uint start_offset   = 0;  // UNION query start index
	uint end_offset     = 0;  // UNION query end index

	// break down AST into UNION sections:
	//--------------------------------------------------------------------------
	// -> MATCH (n)
	//    WHERE n.v > 2
	//    RETURN n.v AS v
	//--------------------------------------------------------------------------
	//    UNION
	//--------------------------------------------------------------------------
	// -> WITH 4 AS x
	//    MATCH (n:N)
	//    WHERE n.score > x
	//    RETURN n.v AS v
	//--------------------------------------------------------------------------
	uint clause_count   = cypher_ast_query_nclauses(ast->root);
	uint *union_indices = AST_GetClauseIndices(ast, CYPHER_AST_UNION);

	array_append(union_indices, clause_count);  // add last clause index
	int union_count = array_len(union_indices);
	ASSERT(union_count > 1);

	//--------------------------------------------------------------------------
	// collect UNION projected expressions
	//--------------------------------------------------------------------------

	// e.g.
	//
	// RETURN 1 AS A, 2 AS B
	// UNION
	// RETURN 3 AS A, 4 AS B
	//
	// projections[0] = 'A' & projections[1] = 'B'

	const cypher_astnode_t *last_clause =
		cypher_ast_query_get_clause(ast->root, clause_count - 1);

	// last clause must be RETURN
	ASSERT(cypher_astnode_type(last_clause) == CYPHER_AST_RETURN);

	uint n_projections = cypher_ast_return_nprojections(last_clause);
	const char *projections[n_projections];

	for(uint i = 0; i < n_projections; i++) {
		// collect aliases from the RETURN clause
		const cypher_astnode_t *projection =
			cypher_ast_return_get_projection(last_clause, i);

		const cypher_astnode_t *alias =
			cypher_ast_projection_get_alias(projection);

		if(alias == NULL) {
			alias = cypher_ast_projection_get_expression(projection);
		}

		ASSERT(alias != NULL);
		projections[i] = cypher_ast_identifier_get_name(alias);
	}

	//--------------------------------------------------------------------------
	// build individual plans
	//--------------------------------------------------------------------------

	ExecutionPlan *plans[union_count];

	for(int i = 0; i < union_count; i++) {
		// create an AST segment from which we will build an execution plan
		end_offset = union_indices[i];
		AST *ast_segment = AST_NewSegment(ast, start_offset, end_offset);

		plans[i] = ExecutionPlan_FromTLS_AST();
		AST_Free(ast_segment); // free the AST segment

		// next segment starts where this one ends
		start_offset = union_indices[i] + 1;
	}

	array_free(union_indices);
	QueryCtx_SetAST(ast); // restore master AST

	// join streams:
	//
	// MATCH (a) RETURN a
	//
	// UNION
	//
	// MATCH (a) RETURN a
	//
	// left stream:     [Scan]->[Project]->[Results]
	// right stream:    [Scan]->[Project]->[Results]
	//
	// Joined:
	// left stream:     [Scan]->[Project]
	// right stream:    [Scan]->[Project]
	// union:           [Join]->[Distinct]->[Result]

	ExecutionPlan *joint_plan = ExecutionPlan_NewEmptyExecutionPlan();
	joint_plan->record_map = raxNew();

	//--------------------------------------------------------------------------
	// result operation
	//--------------------------------------------------------------------------

	OpBase *results_op = NewResultsOp(joint_plan);
	OpBase *parent     = results_op;
	ExecutionPlan_UpdateRoot(joint_plan, results_op);

	//--------------------------------------------------------------------------
	// distinct operation
	//--------------------------------------------------------------------------

	// introduce distinct only if `UNION ALL` isn't specified
	const cypher_astnode_t *u = AST_GetClause(ast, CYPHER_AST_UNION, NULL);
	if(!cypher_ast_union_has_all(u)) {
		// build a Distinct op and add it to the op tree
		OpBase *distinct_op = NewDistinctOp(joint_plan, projections,
				n_projections);
		ExecutionPlan_AddOp(results_op, distinct_op);
		parent = distinct_op;
	}

	//--------------------------------------------------------------------------
	// join operation
	//--------------------------------------------------------------------------

	OpBase *join_op = NewJoinOp(joint_plan);
	ExecutionPlan_AddOp(parent, join_op);

	//--------------------------------------------------------------------------
	// join sub-plans
	//--------------------------------------------------------------------------

	for(int i = 0; i < union_count; i++) {
		ExecutionPlan *sub_plan = plans[i];
		ASSERT(sub_plan->root->type == OPType_RESULTS);

		// remove OP_Result
		OpBase *op_result = sub_plan->root;
		ExecutionPlan_RemoveOp(sub_plan, sub_plan->root);
		OpBase_Free(op_result);

		// migrate projection from the sub-plan to joint-plan
		// we want the sub-plan projection operation to belong to the joint-plan
		// this will gurentee that each stream
		// will place the unioned projections at the same position within the
		// passed record
		// otherwise we risk accessing wrong record indices when switching from
		// one joint stream to another
		OpBase *op = sub_plan->root;
		OPType  t  = OpBase_Type(op);

		// migrate projection
		if(t == OPType_PROJECT || t == OPType_AGGREGATE) {
			// TODO: see if there's a migrate_op function
			// remove projection operation from sub-plan
			ExecutionPlan_RemoveOp(sub_plan, op);

			// bind projection operation to joint plan
			OpBase_BindOpToPlan(op, joint_plan);

			if(sub_plan->root != NULL) {
				// sub_plan isn't empty
				// connect projection operation to the sub-plan root
				OpBase_AddChild(op, sub_plan->root);
			} else {
				// empty plan, free it
				ExecutionPlan_Free(sub_plan);
			}
		} else {
			ASSERT(t == OPType_SORT     ||
				   t == OPType_SKIP     ||
				   t == OPType_LIMIT    ||
				   t == OPType_DISTINCT);

			// as much as we would like to bind the sub-plan projection op
			// to the joint-plan, it's possible for other operations e.g.
			// SORT / DISTINCT / ORDER BY to sit between the sub-plan root
			// and the projection opeartion, in such cases we don't have much
			// of a choice but to introduce a new projection operation
			// which will project the unioned expressions and be bounded to the
			// joint-plan, note the additional clauses mentioned above have the
			// potential to extend the record to included unwanted entries
			// e.g.
			//
			// MATCH (a:A) RETURN a ORDER BY a.v - introduce a.v to the record
			// UNION
			// MATCH (b:B) RETURN b AS a

			// create a new projection operation, create unioned expressions
			AR_ExpNode **exps = array_new(AR_ExpNode*, n_projections);
			for(int j = 0; j < n_projections; j++) {
				const char *alias = projections[j];
				AR_ExpNode *exp = AR_EXP_NewVariableOperandNode(alias);
				exp->resolved_name = alias;
				array_append(exps, exp);
			}
			// add new projection to joint-plan and connect it to the root
			// of the sub-plan
			op = NewProjectOp(joint_plan, exps);
			OpBase_AddChild(op, sub_plan->root);
		}

		// add projection operation as a child of JOIN
		ExecutionPlan_AddOp(join_op, op);
	}

	return joint_plan;
}

static ExecutionPlan *_process_segment
(
	AST *ast,
	uint segment_start_idx,
	uint segment_end_idx
) {
	ASSERT(ast != NULL);
	ASSERT(segment_start_idx <= segment_end_idx);

	ExecutionPlan *segment = NULL;

	// Construct a new ExecutionPlanSegment.
	segment = ExecutionPlan_NewEmptyExecutionPlan();
	segment->ast_segment = ast;
	ExecutionPlan_PopulateExecutionPlan(segment);

	return segment;
}

static ExecutionPlan **_process_segments(AST *ast) {
	uint nsegments = 0;               // number of segments
	uint seg_end_idx = 0;             // segment clause end index
	uint clause_count = 0;            // number of clauses
	uint seg_start_idx = 0;           // segment clause start index
	AST *ast_segment = NULL;          // segment AST
	uint *segment_indices = NULL;     // array segment bounds
	ExecutionPlan *segment = NULL;    // portion of the entire execution plan
	ExecutionPlan **segments = NULL;  // constructed segments

	clause_count = cypher_ast_query_nclauses(ast->root);

	//--------------------------------------------------------------------------
	// bound segments
	//--------------------------------------------------------------------------

	// retrieve the indices of each WITH clause to properly set
	// the segment's bounds.
	// Every WITH clause demarcates the beginning of a new segment
	segment_indices = AST_GetClauseIndices(ast, CYPHER_AST_WITH);

	// last segment
	array_append(segment_indices, clause_count);
	nsegments = array_len(segment_indices);
	segments = array_new(ExecutionPlan *, nsegments);

	//--------------------------------------------------------------------------
	// process segments
	//--------------------------------------------------------------------------

	seg_start_idx = 0;
	for(uint i = 0; i < nsegments; i++) {
		seg_end_idx = segment_indices[i];

		if((seg_end_idx - seg_start_idx) == 0) continue; // skip empty segment

		// slice the AST to only include the clauses in the current segment
		AST *ast_segment = AST_NewSegment(ast, seg_start_idx, seg_end_idx);

		// create ExecutionPlan segment that represents this slice of the AST
		segment = _process_segment(ast_segment, seg_start_idx, seg_end_idx);
		array_append(segments, segment);

		// The next segment will start where the current one ended.
		seg_start_idx = seg_end_idx;
	}

	// Restore the overall AST.
	QueryCtx_SetAST(ast);
	array_free(segment_indices);

	return segments;
}

static bool _ExecutionPlan_HasLocateTaps
(
	OpBase *root
) {
	if((root->childCount == 0 && root->type != OPType_ARGUMENT &&
		root->type != OPType_ARGUMENT_LIST) ||
		// when Foreach or Call {} have a single child, they don't need a tap
		(root->childCount == 1 &&
			(root->type == OPType_FOREACH || root->type == OPType_CALLSUBQUERY))
		) {
			return true;
	}

	// recursively visit children
	for(int i = 0; i < root->childCount; i++) {
		if(_ExecutionPlan_HasLocateTaps(root->children[i])) {
			return true;
		}
	}

	return false;
}

static ExecutionPlan *_tie_segments
(
	ExecutionPlan **segments,
	uint segment_count
) {
	FT_FilterNode  *ft                  =  NULL; // filters following WITH
	OpBase         *connecting_op       =  NULL; // op connecting one segment to another
	OpBase         *prev_connecting_op  =  NULL; // root of previous segment
	ExecutionPlan  *prev_segment        =  NULL;
	ExecutionPlan  *current_segment     =  NULL;
	AST            *master_ast          =  QueryCtx_GetAST();  // top-level AST of plan

	//--------------------------------------------------------------------------
	// merge segments
	//--------------------------------------------------------------------------

	for(int i = 0; i < segment_count; i++) {
		ExecutionPlan *segment = segments[i];
		AST *ast = segment->ast_segment;

		// find the first non-argument op with no children in this segment
		prev_connecting_op = connecting_op;
		// in the case of a single segment with FOREACH as its root, there is no
		// tap (of the current definition)
		// for instance: FOREACH(i in [i] | CREATE (n:N))
		// in any other case, there must be a tap

		ASSERT(_ExecutionPlan_HasLocateTaps(segment->root) == true);

		connecting_op = segment->root;
		while(connecting_op->childCount > 0) {
			connecting_op = connecting_op->children[0];
		}

		// tie the current segment's tap to the previous segment's root op
		if(prev_segment != NULL) {
			// validate the connecting operation
			// the connecting operation may already have children
			// if it's been attached to a previous scope
			ASSERT(connecting_op->type == OPType_PROJECT ||
			       connecting_op->type == OPType_AGGREGATE);

			ExecutionPlan_AddOp(connecting_op, prev_segment->root);
		}

		//----------------------------------------------------------------------
		// build pattern comprehension ops
		//----------------------------------------------------------------------

		// WITH projections
		if(prev_segment != NULL) {
			const cypher_astnode_t *opening_clause = cypher_ast_query_get_clause(ast->root, 0);
			ASSERT(cypher_astnode_type(opening_clause) == CYPHER_AST_WITH);
			uint projections = cypher_ast_with_nprojections(opening_clause);
			for (uint j = 0; j < projections; j++) {
				const cypher_astnode_t *projection = cypher_ast_with_get_projection(opening_clause, j);
				buildPatternComprehensionOps(prev_segment, connecting_op, projection);
				buildPatternPathOps(prev_segment, connecting_op, projection);
			}
		}

		// RETURN projections
		if (segment->root->type == OPType_RESULTS) {
			uint clause_count = cypher_ast_query_nclauses(ast->root);
			const cypher_astnode_t *closing_clause = cypher_ast_query_get_clause(ast->root, clause_count - 1);
			OpBase *op = segment->root;
			while(op->type != OPType_PROJECT && op->type != OPType_AGGREGATE) op = op->children[0];
			uint projections = cypher_ast_return_nprojections(closing_clause);
			for (uint j = 0; j < projections; j++) {
				const cypher_astnode_t *projection = cypher_ast_return_get_projection(closing_clause, j);
				buildPatternComprehensionOps(segment, op, projection);
				buildPatternPathOps(segment, op, projection);
			}
		}

		prev_segment = segment;

		//----------------------------------------------------------------------
		// introduce projection filters
		//----------------------------------------------------------------------

		// Retrieve the current projection clause to build any necessary filters
		const cypher_astnode_t *opening_clause = cypher_ast_query_get_clause(ast->root, 0);
		cypher_astnode_type_t type = cypher_astnode_type(opening_clause);
		// Only WITH clauses introduce filters at this level;
		// all other scopes will be fully built at this point.
		if(type != CYPHER_AST_WITH) continue;

		// Build filters required by current segment.
		QueryCtx_SetAST(ast);
		ft = AST_BuildFilterTreeFromClauses(ast, &opening_clause, 1);
		if(ft == NULL) continue;

		// If any of the filtered variables operate on a WITH alias,
		// place the filter op above the projection.
		if(FilterTree_FiltersAlias(ft, opening_clause)) {
			OpBase *filter_op = NewFilterOp(current_segment, ft);
			ExecutionPlan_PushBelow(connecting_op, filter_op);
		} else {
			// None of the filtered variables are aliases;
			// filter ops may be placed anywhere in the scope.
			ExecutionPlan_PlaceFilterOps(segment, connecting_op, prev_connecting_op, ft);
		}
	}

	// Restore the master AST.
	QueryCtx_SetAST(master_ast);

	// The last ExecutionPlan segment is the master ExecutionPlan.
	ExecutionPlan *plan = segments[segment_count - 1];

	return plan;
}

// Add an implicit "Result" operation to ExecutionPlan if necessary.
static inline void _implicit_result(ExecutionPlan *plan) {
	// If the query culminates in a procedure call, it implicitly returns results.
	if(plan->root->type == OPType_PROC_CALL) {
		OpBase *results_op = NewResultsOp(plan);
		ExecutionPlan_UpdateRoot(plan, results_op);
	}
}

ExecutionPlan *ExecutionPlan_FromTLS_AST(void) {
	AST *ast = QueryCtx_GetAST();

	// handle UNION if there are any
	bool union_query = AST_ContainsClause(ast, CYPHER_AST_UNION);
	if(union_query) return _ExecutionPlan_UnionPlans(ast);

	// execution plans are created in 1 or more segments
	ExecutionPlan **segments = _process_segments(ast);
	ASSERT(segments != NULL);
	uint segment_count = array_len(segments);
	ASSERT(segment_count > 0);

	// connect all segments into a single ExecutionPlan
	ExecutionPlan *plan = _tie_segments(segments, segment_count);

	// the root operation is OpResults only if the query culminates in a RETURN
	// or CALL clause
	_implicit_result(plan);

	// clean up
	array_free(segments);

	return plan;
}

void ExecutionPlan_PreparePlan(ExecutionPlan *plan) {
	// Plan should be prepared only once.
	ASSERT(!plan->prepared);
	Optimizer_RuntimeOptimize(plan);
	plan->prepared = true;
}

inline rax *ExecutionPlan_GetMappings(const ExecutionPlan *plan) {
	ASSERT(plan && plan->record_map);
	return plan->record_map;
}

Record ExecutionPlan_BorrowRecord
(
	ExecutionPlan *plan
) {
	rax *mapping = ExecutionPlan_GetMappings(plan);
	ASSERT(plan->record_pool);

	// get a Record from the pool and set its owner and mapping
	Record r = ObjectPool_NewItem(plan->record_pool);

	r->owner     = plan;
	r->mapping   = mapping;
	r->ref_count = 1;

	return r;
}

void ExecutionPlan_ReturnRecord
(
	const ExecutionPlan *plan,
	Record r
) {
	ASSERT(plan && r);
	ASSERT(r->ref_count > 0);

	// decrease record ref count
	r->ref_count--;

	// free record when ref count reached 0
	if(r->ref_count == 0) {
		// call recursively for parent
		if(r->parent != NULL) {
			ExecutionPlan_ReturnRecord(r->parent->owner, r->parent);
		}
		ObjectPool_DeleteItem(plan->record_pool, r);
	}
}

//------------------------------------------------------------------------------
// Execution plan initialization
//------------------------------------------------------------------------------

static inline void _ExecutionPlan_InitRecordPool
(
	ExecutionPlan *plan
) {
	ASSERT(plan->record_pool == NULL);

	// initialize record pool
	// determine Record size to inform ObjectPool allocation
	uint entries_count = raxSize(plan->record_map);
	uint rec_size = sizeof(_Record) + (sizeof(Entry) * entries_count);

	// create a data block with initial capacity of 256 records
	plan->record_pool = ObjectPool_New(256, rec_size,
			(fpDestructor)Record_FreeEntries);
}

static void _ExecutionPlanInit
(
	OpBase *root
) {
	// TODO: would have been better to get a direct access to every sub-plan
	// stack of operations
	OpBase **ops = array_new(OpBase*, 1);
	array_append(ops, root);

	// as long as there are ops to process
	while(array_len(ops) > 0) {
		// get current op from ops stack
		OpBase *current = array_pop(ops);

		// add child ops to stack
		for(int i = 0; i < current->childCount; i++) {
			array_append(ops, current->children[i]);
		}

		// if the plan associated with this op hasn't built a record pool
		if(current->plan->record_pool == NULL) {
			_ExecutionPlan_InitRecordPool((ExecutionPlan *)current->plan);
		}
	}

	array_free(ops);
}

void ExecutionPlan_Init(ExecutionPlan *plan) {
	_ExecutionPlanInit(plan->root);
}

ResultSet *ExecutionPlan_Execute(ExecutionPlan *plan) {
	ASSERT(plan->prepared)
	// Set an exception-handling breakpoint to capture run-time errors.
	// encountered_error will be set to 0 when setjmp is invoked, and will be nonzero if
	// a downstream exception returns us to this breakpoint
	int encountered_error = SET_EXCEPTION_HANDLER();

	// Encountered a run-time error - return immediately.
	if(encountered_error) return QueryCtx_GetResultSet();

	ExecutionPlan_Init(plan);

	Record r = NULL;
	// Execute the root operation and free the processed Record until the data stream is depleted.
	while((r = OpBase_Consume(plan->root)) != NULL) ExecutionPlan_ReturnRecord(r->owner, r);

	return QueryCtx_GetResultSet();
}

//------------------------------------------------------------------------------
// Execution plan draining
//------------------------------------------------------------------------------

// NOP operation consume routine for immediately terminating execution.
static Record deplete_consume(struct OpBase *op) {
	return NULL;
}

// return true if execution plan been drained
// false otherwise
bool ExecutionPlan_Drained(ExecutionPlan *plan) {
	ASSERT(plan != NULL);
	ASSERT(plan->root != NULL);
	return (plan->root->consume == deplete_consume);
}

static void _ExecutionPlan_Drain(OpBase *root) {
	root->consume = deplete_consume;
	for(int i = 0; i < root->childCount; i++) {
		_ExecutionPlan_Drain(root->children[i]);
	}
}

// Resets each operation consume function to simply return NULL
// this will cause the execution-plan to quickly deplete
void ExecutionPlan_Drain(ExecutionPlan *plan) {
	ASSERT(plan && plan->root);
	_ExecutionPlan_Drain(plan->root);
}

//------------------------------------------------------------------------------
// Execution plan profiling
//------------------------------------------------------------------------------

// initial op's consume function
// performs init followed by consume
Record OpBase_Profile_init
(
	OpBase *op
) {
	ASSERT(op);
	ASSERT(op->init     != NULL);
	ASSERT(op->consume  == OpBase_Profile_init);
	ASSERT(op->_consume != NULL);

	// call op init function
	op->init(op);

	// update op's consume wrapper function
	op->consume = OpBase_Profile;  // calls _consume internally

	// profile
	return OpBase_Profile(op);
}

static void _ExecutionPlan_InitProfiling
(
	OpBase *root
) {
	ASSERT(root != NULL);

	OpBase **ops = array_new(OpBase*, 1);
	array_append(ops, root);

	// as long as there are operations to process
	while(array_len(ops) > 0) {
		// get current op
		OpBase *current = array_pop(ops);

		// add child operations to stack
		for(int i = 0; i < current->childCount; i++) {
			array_append(ops, current->children[i]);
		}

		// set operation consume wrapper function
		current->consume                   = OpBase_Profile_init;
		current->stats                     = rm_malloc(sizeof(OpStats));
		current->stats->profileExecTime    = 0;
		current->stats->profileRecordCount = 0;
	}

	// clean up
	array_free(ops);
}

static void _ExecutionPlan_FinalizeProfiling
(
	OpBase *root
) {
	if(root->childCount) {
		for(int i = 0; i < root->childCount; i++) {
			OpBase *child = root->children[i];
			root->stats->profileExecTime -= child->stats->profileExecTime;
			_ExecutionPlan_FinalizeProfiling(child);
		}
	}
	root->stats->profileExecTime *= 1000;   // Milliseconds.
}

ResultSet *ExecutionPlan_Profile
(
	ExecutionPlan *plan
) {
	_ExecutionPlan_InitProfiling(plan->root);
	ResultSet *rs = ExecutionPlan_Execute(plan);
	_ExecutionPlan_FinalizeProfiling(plan->root);
	return rs;
}

//------------------------------------------------------------------------------
// Execution plan free functions
//------------------------------------------------------------------------------

static void _ExecutionPlan_FreeInternals
(
	ExecutionPlan *plan
) {
	if(plan == NULL) return;

	if(plan->query_graph) {
		QueryGraph_Free(plan->query_graph);
	}
	if(plan->record_map != NULL) {
		raxFree(plan->record_map);
	}
	if(plan->record_pool != NULL) {
		ObjectPool_Free(plan->record_pool);
	}
	if(plan->ast_segment != NULL) {
		AST_Free(plan->ast_segment);
	}
	rm_free(plan);
}

// free the execution plans and all of the operations
void ExecutionPlan_Free
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);
	if(plan->root == NULL) {
		_ExecutionPlan_FreeInternals(plan);
		return;
	}

	// -------------------------------------------------------------------------
	// free op tree and collect execution-plans
	// -------------------------------------------------------------------------

	// traverse the execution-plan graph (DAG -> no endless cycles), while
	// collecting the different segments, and freeing the op tree
	dict *plans = HashTableCreate(&def_dt);
	OpBase **visited = array_new(OpBase *, 1);
	OpBase **to_visit = array_new(OpBase *, 1);

	OpBase *op = plan->root;
	array_append(to_visit, op);

	while(array_len(to_visit) > 0) {
		op = array_pop(to_visit);

		// add the plan this op is affiliated with
		HashTableAdd(plans, (void *)op->plan, (void *)op->plan);

		// add all direct children of op to to_visit
		for(uint i = 0; i < op->childCount; i++) {
			if(op->children[i] != NULL) {
				array_append(to_visit, op->children[i]);
			}
		}

		// add op to `visited` array
		array_append(visited, op);
	}

	// free the collected ops
	for(int i = array_len(visited)-1; i >= 0; i--) {
		op = visited[i];
		OpBase_Free(op);
	}
	array_free(visited);
	array_free(to_visit);

	// -------------------------------------------------------------------------
	// free internals of the plans
	// -------------------------------------------------------------------------

	dictEntry *entry;
	ExecutionPlan *curr_plan;
	dictIterator *it = HashTableGetIterator(plans);
	while((entry = HashTableNext(it)) != NULL) {
		curr_plan = (ExecutionPlan *)HashTableGetVal(entry);
		_ExecutionPlan_FreeInternals(curr_plan);
	}

	HashTableReleaseIterator(it);
	HashTableRelease(plans);
}
