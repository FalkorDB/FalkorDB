/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../algebraic_expression.h"
#include "utils.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../algorithms/algorithms.h"

// node with (income + outcome degree) > 2
// is considered a highly connected node
static bool _highly_connected_node
(
	const QueryGraph *qg,
	const char *alias
) {
	ASSERT(qg    != NULL);
	ASSERT(alias != NULL);

	// look up node in qg
	QGNode *n = QueryGraph_GetNodeByAlias(qg, alias);
	ASSERT(n != NULL);
	return QGNode_HighlyConnected(n);
}

static inline bool _referred_entity
(
	const char *alias
) {
	ASSERT(alias != NULL);

	AST *ast = QueryCtx_GetAST();
	return AST_AliasIsReferenced(ast, alias);
}

// if the edge is referenced or of a variable length
// it should populate the AlgebraicExpression
static inline bool _should_populate_edge
(
	QGEdge *e
) {
	ASSERT(e != NULL);
	return (_referred_entity(e->alias) ||
			QGEdge_VariableLength(e)   ||
			QGEdge_GhostEdge(e));
}

// checks if given expression contains a variable length edge
static bool _AlgebraicExpression_ContainsVariableLengthEdge
(
	const QueryGraph *qg,
	const AlgebraicExpression *exp
) {
	ASSERT(qg != NULL);
	ASSERT(exp != NULL);

	uint child_count = 0;
	switch(exp->type) {
	case AL_OPERATION:
		child_count = AlgebraicExpression_ChildCount(exp);
		for(uint i = 0; i < child_count; i++) {
			AlgebraicExpression *child = exp->operation.children[i];
			if(_AlgebraicExpression_ContainsVariableLengthEdge(qg, child)) {
				return true;
			}
		}
		return false;
	case AL_OPERAND:
		if(exp->operand.edge) {
			QGEdge *e = QueryGraph_GetEdgeByAlias(qg, exp->operand.edge);
			return QGEdge_VariableLength(e) || QGEdge_GhostEdge(e);
		}
		break;
	default:
		ASSERT("Unknow algebraic expression node type" && false);
	}
	return false;
}

static void _RemovePathFromGraph
(
	QueryGraph *g,
	QGEdge **path
) {
	ASSERT(g    != NULL);
	ASSERT(path != NULL);

	uint edge_count = array_len(path);
	for(uint i = 0; i < edge_count; i++) {
		QGEdge *e = path[i];
		QGNode *src = e->src;
		QGNode *dest = e->dest;

		QueryGraph_RemoveEdge(g, e);
		QGEdge_Free(e);

		if(QGNode_EdgeCount(src) == 0) {
			QueryGraph_RemoveNode(g, src);
			QGNode_Free(src);
		}

		if(src != dest && QGNode_EdgeCount(dest) == 0) {
			QueryGraph_RemoveNode(g, dest);
			QGNode_Free(dest);
		}
	}
}

static inline bool _should_divide_expression
(
	QGEdge **path,
	int idx,
	const QueryGraph *qg
) {
	QGEdge *e = path[idx];

	return (_should_populate_edge(e)                    ||  // this edge is populated
			_should_populate_edge(path[idx + 1])        ||  // the next edge is populated
			_highly_connected_node(qg, e->dest->alias)  ||  // destination node in+out degree > 2
			_referred_entity(e->dest->alias));              // destination node is referenced
}

// variable length expression must contain only a single operand: the edge being
// traversed, in cases such as (:labelA)-[e*]->(:labelB) both label A and B
// are applied via a label matrix operand, this function migrates A and B from a
// variable length expression to new expressions
// or completely discards them when possible
static AlgebraicExpression **_AlgebraicExpression_IsolateVariableLenExps
(
	const QueryGraph *qg,
	AlgebraicExpression **expressions
) {
	// return value is a new set of expressions
	// where each variable length expression
	// is guaranteed to have a single operand
	// as such in the worst case the number of
	// expressions doubles + 1
	size_t expCount = array_len(expressions);
	AlgebraicExpression **res = array_new(AlgebraicExpression*, expCount*2+1);

	// scan through each expression, locate expression which
	// have a variable length edge in them
	for(size_t expIdx = 0; expIdx < expCount; expIdx++) {
		AlgebraicExpression *exp = expressions[expIdx];
		if(!_AlgebraicExpression_ContainsVariableLengthEdge(qg, exp)) {
			array_append(res, exp);
			continue;
		}

		//----------------------------------------------------------------------
		// handle source
		//----------------------------------------------------------------------

		// expression contains a variable length edge
		const char *src_alias = AlgebraicExpression_Src(exp);
		QGNode *src = NULL;
		if(src_alias != NULL) {
			src = QueryGraph_GetNodeByAlias(qg, src_alias);
		}

		// a variable length expression with a labeled source node
		// we only care about the source label matrix, when it comes to
		// the first expression, as in the following expressions
		// src is the destination of the previous expression
		AlgebraicExpression *op = NULL;
		if(src != NULL && QGNode_Labeled(src)) {
			// remove src node matrix from expression
			op = AlgebraicExpression_RemoveSource(&exp);
		}

		if(op != NULL) {
			if(expIdx == 0) {
				array_append(res, op);
			} else {
				AlgebraicExpression_Free(op);
			}
		}

		array_append(res, exp);

		// if the expression has a labeled destination,
		// separate it into its own expression
		op = NULL;

		//----------------------------------------------------------------------
		// handle destination
		//----------------------------------------------------------------------

		const char *dest_alias = AlgebraicExpression_Dest(exp);
		if(dest_alias != NULL) {
			QGNode *dest = QueryGraph_GetNodeByAlias(qg, dest_alias);

			if(QGNode_Labeled(dest)) {
				// remove dest node matrix from expression
				op = AlgebraicExpression_RemoveDest(&exp);
			}
		}

		if(op != NULL) {
			// remove destination if following expression isn't a
			// variable length edge (src/dest sharing) otherwise introduce a new
			// label expression
			if(expIdx < expCount - 1 &&
			   !_AlgebraicExpression_ContainsVariableLengthEdge(qg, expressions[expIdx + 1])) {
				AlgebraicExpression_Free(op);
			} else {
				array_append(res, op);
			}
		}
	}

	array_free(expressions);
	return res;
}

// break down path into sub paths
// considering referenced intermidate nodes and edges
static QGEdge ***_Intermediate_Paths
(
	QGEdge **path,
	const QueryGraph *qg
) {
	ASSERT(path != NULL);
	ASSERT(qg != NULL);

	QGEdge *e = NULL;
	int pathLen = array_len(path);

	// allocating maximum number of expression possible
	QGEdge ***paths = array_new(QGEdge **, pathLen);
	QGEdge **intermediate_path = array_new(QGEdge *, pathLen);
	array_append(paths, intermediate_path);

	// scan path left to right
	// construct intermidate paths by "breaking" on referenced entities
	for(int i = 0; i < pathLen - 1; i++) {
		e = path[i];
		array_append(intermediate_path, e);
		if(_should_divide_expression(path, i, qg)) {
			// break! add current path to paths and create a new path
			intermediate_path = array_new(QGEdge *, pathLen);
			array_append(paths, intermediate_path);
		}
	}

	// handle last hop
	e = path[pathLen - 1];
	array_append(intermediate_path, e);

	return paths;
}

static AlgebraicExpression *_AlgebraicExpression_OperandFromNode
(
	QGNode *n
) {
	ASSERT(n != NULL);

	bool diag = true;
	const char *alias = n->alias;

	return AlgebraicExpression_NewOperand(NULL, diag, alias, alias, NULL, NULL);
}

static void _AlgebraicExpression_ExpandNodeOperand
(
	const QueryGraph *qg,
	AlgebraicExpression *exp
) {
	ASSERT(qg != NULL);
	ASSERT(exp != NULL);
	ASSERT(exp->type == AL_OPERAND);
	ASSERT(exp->operand.diagonal == true);

	const  char  *l      =  NULL;
	const  char  *alias  =  AlgebraicExpression_Src(exp);

	QGNode *n = QueryGraph_GetNodeByAlias(qg, alias);
	ASSERT(n != NULL);

	uint label_count = QGNode_LabelCount(n);
	if(label_count > 0) l = QGNode_GetLabel(n, 0);

	if(label_count < 2) {
		// set label
		exp->operand.label = l;
	} else {
		// two or more labels
		_InplaceRepurposeOperandToOperation(exp, AL_EXP_MUL);
		for(int i = 0; i < label_count; i++) {
			l = QGNode_GetLabel(n, i);
			AlgebraicExpression *op = AlgebraicExpression_NewOperand(NULL, true,
					alias, alias, NULL, l);
			AlgebraicExpression_AddChild(exp, op);
		}
	}
}

static void _AlgebraicExpression_ExpandNodeOperands
(
	const QueryGraph *qg,
	AlgebraicExpression *exp
) {
	ASSERT(qg  != NULL);
	ASSERT(exp != NULL);
	uint child_count;

	switch(exp->type) {
		case AL_OPERAND:
			if(exp->operand.diagonal) {
				_AlgebraicExpression_ExpandNodeOperand(qg, exp);
			}
			break;
		case AL_OPERATION:
			child_count = AlgebraicExpression_ChildCount(exp);
			for(uint i = 0; i < child_count; i++) {
				AlgebraicExpression *child = exp->operation.children[i];
				_AlgebraicExpression_ExpandNodeOperands(qg, child); 
			}
			break;
		default:
			ASSERT(false);
			break;
	}
}

static AlgebraicExpression *_AlgebraicExpression_OperandFromEdge
(
	QGEdge *e,
	bool transpose
) {
	ASSERT(e != NULL);

	uint reltype_id;
	QGNode              *src_node         = e->src;
	QGNode              *dest_node        = e->dest;
	AlgebraicExpression *add              = NULL;
	AlgebraicExpression *root             = NULL;
	AlgebraicExpression *src_filter       = NULL;

	// zero length edges should be treated as varaiable length traversal
	// by doing so a "var-len conditional traversal" operation will
	// be used matching only the source node of the pattern
	bool var_len_traversal = (QGEdge_VariableLength(e) || QGEdge_GhostEdge(e));

	// use original `src` and `dest` for algebraic operands
	const char *src  = (transpose)              ? dest_node->alias : src_node->alias;
	const char *dest = (transpose)              ? src_node->alias  : dest_node->alias;
	const char *edge = _should_populate_edge(e) ? e->alias         : NULL;

	// if src node has a label, multiply to the left by label matrix
	if(QGNode_LabelCount(src_node) > 0) {
		src_filter = _AlgebraicExpression_OperandFromNode(src_node);
	}

	uint reltype_count = array_len(e->reltypeIDs);
	switch(reltype_count) {
	case 0: // no relationship types specified; use the adjacency matrix
		root = AlgebraicExpression_NewOperand(NULL, false, src, dest, edge, NULL);
		break;
	case 1: // single relationship type
		root = AlgebraicExpression_NewOperand(NULL, false, src, dest, edge, e->reltypes[0]);
		break;
	default: // multiple edge type: -[:A|:B]->
		add = AlgebraicExpression_NewOperation(AL_EXP_ADD);
		for(uint i = 0; i < reltype_count; i++) {
			AlgebraicExpression *operand = AlgebraicExpression_NewOperand(
					NULL, false, src, dest, edge, e->reltypes[i]);
			AlgebraicExpression_AddChild(add, operand);
		}
		root = add;
		break;
	}

	if(e->bidirectional) {
		// ()-[]-()
		// Adj + Transpose(Adj)
		//
		// ()-[:R]-()
		// R + Transpose(R)
		//
		// ()-[:R0|R1]-()
		// (R0 + R1) + Transpose(R0 + R1)
		add = AlgebraicExpression_NewOperation(AL_EXP_ADD);
		AlgebraicExpression_AddChild(add, root);

		AlgebraicExpression *op_transpose = AlgebraicExpression_NewOperation(AL_EXP_TRANSPOSE);
		AlgebraicExpression_AddChild(op_transpose, AlgebraicExpression_Clone(root));
		AlgebraicExpression_AddChild(add, op_transpose);
		root = add;
	}

	// expand fixed variable length edge
	// -[A*2..2]->
	// A*A
	// -[A|B*2..2]->
	// (A+B) * (A+B)
	if(!var_len_traversal && e->minHops > 1) {
		AlgebraicExpression *mul = AlgebraicExpression_NewOperation(AL_EXP_MUL);
		AlgebraicExpression_AddChild(mul, root);
		for(int i = 1; i < e->minHops; i++) {
			// clone to avoid double free
			AlgebraicExpression_AddChild(mul, AlgebraicExpression_Clone(root));
		}
		root = mul;
	}

	// transpose entire expression
	if(transpose) {
		AlgebraicExpression *op_transpose = AlgebraicExpression_NewOperation(AL_EXP_TRANSPOSE);
		AlgebraicExpression_AddChild(op_transpose, root);
		root = op_transpose;
	}

	// apply source filter by multiplying to the left
	if(src_filter) {
		root = _AlgebraicExpression_MultiplyToTheLeft(src_filter, root);
	}

	return root;
}

/* In case edges `e0` and `e1` share a node:
 * (a)-[E0]->(b)<-[E1]-(c)
 * than the shared entity is returned
 * if edges are disjoint, NULL is returned. */
static QGNode *_SharedNode
(
	const QGEdge *e0,
	const QGEdge *e1
) {
	ASSERT(e0 != NULL);
	ASSERT(e1 != NULL);

	if(e0->dest == e1->src)  return e0->dest;  // (a)-[E0]->(b)-[E1]->(c)
	if(e0->dest == e1->dest) return e0->dest;  // (a)-[E0]->(b)<-[E1]-(c)
	if(e0->src == e1->dest)  return e0->src;   // (a)<-[E0]-(b)<-[E1]-(c)
	if(e0->src == e1->src)   return e0->src;   // (a)<-[E0]-(b)-[E1]->(c)

	return NULL;
}

static void _reversePath
(
	QGEdge **path,
	uint path_len,
	bool *transpositions
) {
	for(uint i = 0; i < path_len; i++) {
		/* A reversed path should have its transpositions flipped
		 * transpose(transpose(A)) = A */
		transpositions[i] = !transpositions[i];
	}

	/* Transpose(A*B) = Transpose(B) * Transpose(A)
	 * (a)<-[A]-(b)<-[B]-(c)-[C]->(d)
	 * At * Bt * C
	 * Transpose(At * Bt * C) =
	 * = Ct * B * A
	 * (d)-[Ct]->(c)-[B]->(b)-[A]->(a) */

	// Reverse the path array as well as the transposition array
	for(uint i = 0; i < path_len / 2; i++) {
		uint opposite = path_len - i - 1;
		QGEdge *tmp = path[opposite];
		path[opposite] = path[i];
		path[i] = tmp;

		bool transpose_tmp = transpositions[opposite];
		transpositions[opposite] = transpositions[i];
		transpositions[i] = transpose_tmp;
	}
}

static void _normalizePath
(
	QGEdge **path,          // path to normalize
	uint path_len,          // path length
	bool *transpositions    // specifies which edges need to be transposed
) {
	// initialize `transpositions` array
	memset(transpositions, 0, sizeof(bool) * path_len);

	// a single leg path
	if(path_len <= 1) return;

	uint transposeCount = 0;
	// for every edge except the last
	for(uint i = 0; i < path_len - 1; i++) {
		QGEdge *e = path[i];
		QGEdge *follow = path[i + 1];
		QGNode *shared = _SharedNode(e, follow);
		ASSERT(shared);

		// the edge should be transposed if its destination is not shared
		// (dest)<-[e]-(shared)-[follow]->()
		// (dest)<-[e]-(shared)<-[follow]-()
		if(e->dest != shared) {
			transpositions[i] = true;
			transposeCount++;
		}
	}

	// for the last edge, transpose if its source is not shared
	QGEdge *e = path[path_len - 1];
	QGNode *shared = _SharedNode(path[path_len - 2], e);
	if(e->src != shared) {
		transposeCount++;
		transpositions[path_len - 1] = true;
	}

	// reverse entire path if the majority of edges must be transposed
	if(transposeCount > (path_len - transposeCount)) {
		_reversePath(path, path_len, transpositions);
	}

	// apply transpose
	for(uint i = 0; i < path_len; i++) {
		QGEdge *e = path[i];
		if(transpositions[i]) QGEdge_Reverse(e);
	}
}

static AlgebraicExpression *_AlgebraicExpression_FromPath
(
	QGEdge **path,
	bool *transpositions
) {
	ASSERT(path);

	QGEdge               *e        =  NULL;
	AlgebraicExpression  *root     =  NULL;
	uint                 path_len  =  array_len(path);

	ASSERT(path_len > 0);

	// treating path as a chain
	// we're aligning all edges to "point right"
	// (A)-[E0]->(B)-[E0]->(C)-[E0]->(D).
	// e.g.
	// (A)-[E0]->(B)<-[E1]-(C)-[E2]->(D)
	// E1 will be transposed:
	// (A)-[E0]->(B)-[E1']->(C)-[E2]->(D) */

	// construct expression
	for(int i = 0; i < path_len; i++) {
		e = path[i];
		// add edge matrix
		AlgebraicExpression *op = _AlgebraicExpression_OperandFromEdge(e,
				transpositions[i]);

		if(root == NULL) {
			root = op;
		} else {
			// connect via a multiplication node
			root = _AlgebraicExpression_MultiplyToTheRight(root, op);
		}
	}   // end of path traversal

	// if last node on path has a label, multiply by label matrix
	if(QGNode_LabelCount(e->dest) > 0) {
		root = _AlgebraicExpression_MultiplyToTheRight(root,
				_AlgebraicExpression_OperandFromNode(e->dest));
	}

	return root;
}

//------------------------------------------------------------------------------
// AlgebraicExpression construction.
//------------------------------------------------------------------------------

// construct algebraic expression form query graph
AlgebraicExpression **AlgebraicExpression_FromQueryGraph
(
	const QueryGraph *qg    // Query-graph to process
) {
	ASSERT(qg != NULL);

	// construct algebraic expression(s) from query-graph
	// trying to take advantage of long multiplications with as few
	// transpose as possible we'll transform paths crossing the graph
	// "diameter", these are guarantee to be the longest, although
	// there might be situations in which these are not the most optimal paths
	// to explore
	//
	// once a path been transformed it's removed from the query-graph
	// and the process repeat itself

	// a graph with no edges implies an empty algebraic expression
	// the reasoning behind this decision is that the algebraic expression
	// represents graph traversals, no edges means no traversals
	AlgebraicExpression **exps = array_new(AlgebraicExpression *, 1);
	uint edge_count = QueryGraph_EdgeCount(qg);

	if(edge_count == 0) {
		ASSERT(QueryGraph_NodeCount(qg) == 1);
		QGNode *n = qg->nodes[0];
		AlgebraicExpression *exp = _AlgebraicExpression_OperandFromNode(n);
		_AlgebraicExpression_ExpandNodeOperands(qg, exp);
		array_append(exps, exp);
		return exps;
	}

	bool acyclic = IsAcyclicGraph(qg);
	QueryGraph *g = QueryGraph_Clone(qg);

	// as long as the query-graph isn't empty
	while(QueryGraph_EdgeCount(g) > 0) {
		// get leaf nodes at the deepest level
		int depth;
		QGNode *n;
		if(acyclic) n = LongestPathTree(g, &depth); // graph is a tree
		else n = LongestPathGraph(g, &depth);       // graph contains cycles

		// get a path of length level
		// allow closing a cycle if the graph is not acyclic
		QGEdge **path = DFS(n, depth, !acyclic);
		uint path_len = array_len(path);
		ASSERT(path_len == depth);

		// TODO:
		// in case path is a cycle, e.g. (b)-[]->(a)-[]->(b)
		// make sure the first node on the path is referenced
		// _should_divide_expression(path, 0) is true
		// if this is not the case we will unnecessarily break
		// the generated expression into 2 sub expressions
		// while what we can do is simply rotate the cycle, (a)-[]->(b)-[]->(a)
		// this is exactly the same only now we won't sub divide
		// checking if path is a cycle done by testing the start and end node

		// split path into sub paths
		bool transpositions[path_len];
		_normalizePath(path, path_len, transpositions);

		QGEdge ***paths = _Intermediate_Paths(path, qg);
		AlgebraicExpression **sub_exps = array_new(AlgebraicExpression *, 1);

		uint path_count = array_len(paths);
		uint edge_converted = 0;
		for(uint i = 0; i < path_count; i++) {
			// construct expression
			AlgebraicExpression *exp = _AlgebraicExpression_FromPath(paths[i],
					transpositions + edge_converted);

			edge_converted += array_len(paths[i]);
			array_append(sub_exps, exp);

			// remove exp[i] src label matrix (left most operand) as it's
			// being used by exp[i-1] dest label matrix
			// (:A)-[:X]->(:B)-[:Y]->(:C)
			// exp0: A * X * B
			// exp1: B * Y * C
			// should become
			// exp0: A * X * B
			// exp1: Y * C
			if(i > 0) {
				AlgebraicExpression *prev_exp = sub_exps[i-1];
				// make sure expression i follows previous expression
				const char *src_alias = AlgebraicExpression_Src(exp);
				const char *dest_alias = AlgebraicExpression_Dest(prev_exp);
				
				// Only check consistency if both aliases are valid
				if(src_alias != NULL && dest_alias != NULL) {
					QGNode *src = QueryGraph_GetNodeByAlias(qg, src_alias);
					QGNode *dest = QueryGraph_GetNodeByAlias(qg, dest_alias);
					ASSERT(src == dest);
				}

				// exp[i] shares a label matrix with exp[i-1]
				// remove redundancy
				//if(QGNode_Labeled(src)) {
				//	AlgebraicExpression *redundent = AlgebraicExpression_RemoveSource(&exp);
				//	AlgebraicExpression_Free(redundent);
				//}
			}
			// expression can not be empty
			ASSERT(AlgebraicExpression_OperandCount(exp) > 0);
		}

		sub_exps = _AlgebraicExpression_IsolateVariableLenExps(qg, sub_exps);

		uint sub_count = array_len(sub_exps);
		for(uint i = 0; i < sub_count; i++) {
			AlgebraicExpression *exp = sub_exps[i];
			_AlgebraicExpression_ExpandNodeOperands(qg, exp);
			// add constructed expression to return value
			array_append(exps, exp);
		}

		// remove path from graph
		_RemovePathFromGraph(g, path);

		// clean up
		for(uint i = 0; i < path_count; i++) array_free(paths[i]);
		array_free(path);
		array_free(paths);
		array_free(sub_exps);

		// if original graph contained a cycle
		// see now after we've removed a path if this is still the case
		if(!acyclic) acyclic = IsAcyclicGraph(g);
	}

	QueryGraph_Free(g);
	return exps;
}

