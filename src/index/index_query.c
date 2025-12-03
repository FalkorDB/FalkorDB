/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index.h"
#include "../util/arr.h"
#include "../index/index.h"
#include "../datatypes/point.h"
#include "../datatypes/array.h"
#include "../util/range/string_range.h"
#include "../util/range/numeric_range.h"
#include "../filter_tree/filter_tree_utils.h"
#include "../arithmetic/arithmetic_expression.h"

extern void Index_RangeFieldName
(
	char *type_aware_name,  // [out] type aware name
	const char *name,       // field name
	SIType *multi_val_type  // [optional] multi-val type
);

extern void Index_VectorFieldName
(
	char *type_aware_name,  // [out] type aware name
	const char *name        // field name
);

//------------------------------------------------------------------------------
// forward declarations
//------------------------------------------------------------------------------

// returns true if 'tree' been converted into an index query, false otherwise
static bool _FilterTreeToQueryNode
(
	RSQNode**root,             // array of query nodes to populate
	const FT_FilterNode *tree, // filter to convert into an index query
	RSIndex *idx               // queried index
);

//------------------------------------------------------------------------------
// To RediSearch query node
//------------------------------------------------------------------------------

// create a RediSearch query node out of a numeric range object
static RSQNode *_NumericRangeToQueryNode
(
	RSIndex *idx,              // queried index
	const char *field,         // queried field
	const NumericRange *range  // range to query
) {
	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, field, NULL);

	double max = (range->max == INFINITY) ? RSRANGE_INF : range->max;
	double min = (range->min == -INFINITY) ? RSRANGE_NEG_INF : range->min;
	return RediSearch_CreateNumericNode(idx, type_aware_field_name, max, min,
			range->include_max, range->include_min);
}

// create a RediSearch query node out of a string range object
static RSQNode *_StringRangeToQueryNode
(
	RSIndex *idx,             // queried index
	const char *field,        // queried field
	const StringRange *range  // range to query
) {
	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, field, NULL);
	RSQNode *root = RediSearch_CreateTagNode(idx, type_aware_field_name);
	RSQNode *child = NULL;
	const char *max = range->max;
	const char *min = range->min;

	if(max != NULL && min != NULL && strcmp(max, min) == 0) {
		// exact match
		child = RediSearch_CreateTagTokenNode(idx, max);
	} else {
		// range search
		max = (max == NULL) ? RSLECRANGE_INF     : max;
		min = (min == NULL) ? RSLEXRANGE_NEG_INF : min;
		child = RediSearch_CreateTagLexRangeNode(idx, min, max,
				range->include_min, range->include_max);
	}

	RediSearch_QueryNodeAddChild(root, child);

	return root;
}

// creates a RediSearch distance query from given filter
static RSQNode *_FilterTreeToDistanceQueryNode
(
	const FT_FilterNode *filter,  // filter to convert
	RSIndex *idx                  // queried index
) {
	char    *field = NULL;          // field being filtered
	SIValue origin = SI_NullVal();  // center of circle
	SIValue radius = SI_NullVal();  // circle radius

	extractOriginAndRadius(filter, &origin, &radius, &field);

	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, field, NULL);

	return RediSearch_CreateGeoNode(idx, type_aware_field_name,
			Point_lat(origin), Point_lon(origin), SI_GET_NUMERIC(radius),
			RS_GEO_DISTANCE_M);
}

// creates a RediSearch query node matching an entity with an attribute value
// which matches one of multiple values specified in an array
// e.g. WHERE n.v IN [1, 2, 3]
static bool _FilterTreeToMultiValQueryNode
(
	const FT_FilterNode *filter,  // filter to convert
	RSIndex *idx,                 // queried index
	RSQNode **root                // [output] RediSearch query node
) {
	ASSERT(idx    != NULL);
	ASSERT(root   != NULL);
	ASSERT(filter != NULL);
	ASSERT(isInFilter(filter));

	*root = NULL;

	// n.v IN [1,2,3]
	// a single union node should hold a number of token/numeric nodes
	// one for each element in the array.

	// extract both attribute name and list from expression
	AR_ExpNode *inOp = FilterTree_getExpression(filter);

	char *attr;
	bool attribute = AR_EXP_IsAttribute(AR_EXP_getChild(inOp, 0), &attr);
	ASSERT(attribute == true);

	SIValue list  = AR_EXP_getChild(inOp, 1)->operand.constant;
	uint list_len = SIArray_Length(list);

	if(list_len == 0) {
		// special case: "WHERE a.v in []"
		*root = RediSearch_CreateEmptyNode(idx);
		return true;
	}

	bool     res    = true;
	RSQNode *node   = NULL;
	RSQNode *parent = NULL;
	RSQNode *U      = RediSearch_CreateUnionNode(idx);

	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, attr, NULL);

	for(uint i = 0; res && i < list_len; i++) {
		double d;
		SIValue v = SIArray_Get(list, i);
		switch(SI_TYPE(v)) {
			case T_STRING:
			case T_INTERN_STRING:
				parent = RediSearch_CreateTagNode(idx, type_aware_field_name);
				node   = RediSearch_CreateTokenNode(idx, type_aware_field_name,
						v.stringval);
				RediSearch_QueryNodeAddChild(parent, node);
				node = parent;

				RediSearch_QueryNodeAddChild(U, node);
				break;

			case T_BOOL:
			case T_INT64:
			case T_DOUBLE:
				d = SI_GET_NUMERIC(v);
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						d, d, true, true);

				RediSearch_QueryNodeAddChild(U, node);
				break;

			default:
				ASSERT(false && "unexpected conditional operation");
				res = false;
				break;
		}
	}

	if(!res) {
		RediSearch_QueryNodeFree(U);
	} else {
		*root = U;
	}

	return res;
}

// convert an IN filter into a RediSearch query
// e.g. WHERE 8 IN n.v
static bool _FilterTreeToInQueryNode
(
	const FT_FilterNode *filter,  // filter to convert
	RSIndex *idx,                 // queried index
	RSQNode **node                // RediSearch query node
) {
	ASSERT(idx    != NULL);
	ASSERT(node   != NULL);
	ASSERT(filter != NULL);
	ASSERT(isInFilter(filter));

	*node = NULL;

	// MATCH (n:N) WHERE 8 IN n.v RETURN n

	//--------------------------------------------------------------------------
	// extract both attribute name and scalar from expression
	//--------------------------------------------------------------------------

	AR_ExpNode *exp = FilterTree_getExpression(filter);

	// extract scalar
	AR_ExpNode *scalar = AR_EXP_getChild(exp, 0);
	ASSERT(AR_EXP_IsConstant(scalar));

	SIValue v = scalar->operand.constant;
	SIType  t = SI_TYPE(v);

	// extract entity attribute
	char *attr;
	bool attribute = AR_EXP_IsAttribute(AR_EXP_getChild(exp, 1), &attr);
	ASSERT(attribute == true);

	//--------------------------------------------------------------------------
	// compose RediSearch query
	//--------------------------------------------------------------------------

	// only utilize index if scalar is either numeric, boolean or string
	if(!(t & (T_DOUBLE | T_INT64 | T_BOOL | T_STRING))) {
		return false;
	}

	double   d;
	bool     res    = false;
	RSQNode *child  = NULL;
	RSQNode *parent = NULL;

	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, attr, &t);

	switch(t) {
		case T_DOUBLE:
		case T_INT64:
		case T_BOOL:
			d = SI_GET_NUMERIC(v);
			*node = RediSearch_CreateNumericNode(idx, type_aware_field_name, d,
					d, true, true);
			res = true;
			break;
		case T_STRING:
		case T_INTERN_STRING:
				parent = RediSearch_CreateTagNode(idx, type_aware_field_name);
				child  = RediSearch_CreateTokenNode(idx, type_aware_field_name,
						v.stringval);
				RediSearch_QueryNodeAddChild(parent, child);
				*node = parent;
				res = true;
			break;
		default:
			ASSERT(false && "unsupported IN filter operand");
			break;
	}

	return res;
}

// reduce filter into a range object
// return true if filter was reduce, false otherwise
static bool _predicateTreeToRange
(
	const FT_FilterNode *tree,  // filter to convert
	rax *string_ranges,         // string ranges
	rax *numeric_ranges         // numerical ranges
) {
	ASSERT(tree           != NULL);
	ASSERT(string_ranges  != NULL);
	ASSERT(numeric_ranges != NULL);

	// handel filters of form: 'n.v op constant'
	char *prop = NULL;
	// TODO: we might not need this check
	if(!AR_EXP_IsAttribute(tree->pred.lhs, &prop)) return false;

	ASSERT(!AR_EXP_ContainsVariadic(tree->pred.rhs));
	SIValue c = AR_EXP_Evaluate(tree->pred.rhs, NULL);
	SIType  t = SI_TYPE(c);

	// make sure constant is an indexable type
	if(!(t & SI_INDEXABLE)) return false;

	int           op        =  tree->pred.op;
	StringRange   *sr       =  NULL;
	NumericRange  *nr       =  NULL;
	uint          prop_len  =  strlen(prop);

	// get or create range object for alias.prop
	// constant is either numeric or boolean
	if(t & SI_NUMERIC || t == T_BOOL) {
		// TODO: remove when RediSearch INT64 indexing bug fixed
		if(t == T_INT64 && c.longval & 0x7FF0000000000000) {
			return false;
		}
		nr = raxFind(numeric_ranges, (unsigned char *)prop, prop_len);
		// create if doesn't exists
		if(nr == raxNotFound) {
			nr = NumericRange_New();
			raxTryInsert(numeric_ranges, (unsigned char *)prop, prop_len,
					nr, NULL);
		}
		NumericRange_TightenRange(nr, op, SI_GET_NUMERIC(c));
	} else if(t & T_STRING) {
		sr = raxFind(string_ranges, (unsigned char *)prop, prop_len);
		// create if doesn't exists
		if(sr == raxNotFound) {
			sr = StringRange_New();
			raxTryInsert(string_ranges, (unsigned char *)prop, prop_len, sr, NULL);
		}
		StringRange_TightenRange(sr, op, c.stringval);
	}

	return true;
}

// connect all RediSearch query nodes
static RSQNode *_concat_query_nodes
(
	RSIndex *idx,     // queried index
	RSQNode **nodes,  // query nodes to concat
	uint count        // number of nodes
) {
	// no nodes, can not utilize the index
	if(count == 0) return NULL;

	// just a single filter
	if(count == 1) return nodes[0];

	// multiple filters, combine using AND
	RSQNode *root = RediSearch_CreateIntersectNode(idx, false);
	for(uint i = 0; i < count; i++) {
		RSQNode *qnode = nodes[i];
		RediSearch_QueryNodeAddChild(root, qnode);
	}

	return root;
}

// compose index query from ranges
static RSQNode *_ranges_to_query_nodes
(
	RSIndex *idx,        // index to query
	rax *string_ranges,  // string ranges
	rax *numeric_ranges  // numerical ranges
) {
	ASSERT(string_ranges  != NULL);
	ASSERT(numeric_ranges != NULL);

	// build RediSearch query tree
	// convert each range object to RediSearch query node
	raxIterator it;
	bool valid = true;  // false if there's a range conflict

	//--------------------------------------------------------------------------
	// validate ranges
	//--------------------------------------------------------------------------
	
	// validate string ranges
	raxStart(&it, string_ranges);
	raxSeek(&it, "^", NULL, 0);

	while(raxNext(&it)) {
		// make sure each property is bound to either numeric or string type
		// but not to both, e.g. a.v = 1 AND a.v = 'a'
		// in which case use an empty RSQueryNode
		char *field = (char *)it.key;
		if(raxFind(numeric_ranges, (unsigned char *)field, (int)it.key_len) != raxNotFound) {
			valid = false;
			break;
		}

		StringRange *sr = (StringRange *) it.data;
		if(!StringRange_IsValid(sr)) {
			valid = false;
			break;
		}
	}
	raxStop(&it);
	
	if(valid == false) return RediSearch_CreateEmptyNode(idx);

	// validate numeric ranges
	raxStart(&it, numeric_ranges);
	raxSeek(&it, "^", NULL, 0);

	while(raxNext(&it)) {
		NumericRange *nr = (NumericRange *) it.data;
		if(!NumericRange_IsValid(nr)) {
			valid = false;
			break;
		}
	}
	raxStop(&it);

	if(valid == false) return RediSearch_CreateEmptyNode(idx);

	//--------------------------------------------------------------------------
	// construct index range queries
	//--------------------------------------------------------------------------

	// detemine number of ranges
	uint i = 0;
	char query_field_name[1024];
	uint range_count = raxSize(numeric_ranges) + raxSize(string_ranges);
	RSQNode *rsqnodes[range_count];

	raxSeek(&it, "^", NULL, 0);
	while(raxNext(&it)) {
		char *field = (char *)it.key;
		ASSERT(it.key_len < sizeof(query_field_name));
		NumericRange *nr = (NumericRange *) it.data;

		snprintf(query_field_name, 1024, "%.*s", (int)it.key_len, field);
		RSQNode *rsqn = _NumericRangeToQueryNode(idx, query_field_name, nr);
		rsqnodes[i++] = rsqn;
	}
	raxStop(&it);

	raxStart(&it, string_ranges);
	raxSeek(&it, "^", NULL, 0);
	while(raxNext(&it)) {
		char *field = (char *)it.key;
		ASSERT(it.key_len < sizeof(query_field_name));
		StringRange *sr = (StringRange *) it.data;

		snprintf(query_field_name, 1024, "%.*s", (int)it.key_len, field);
		RSQNode *rsqn = _StringRangeToQueryNode(idx, query_field_name, sr);
		rsqnodes[i++] = rsqn;
	}
	raxStop(&it);

	RSQNode *root = _concat_query_nodes(idx, rsqnodes, range_count);
	return root;
}

// reduce filters into ranges
static void _compose_ranges
(
	const FT_FilterNode **trees,  // filters to convert into ranges
	rax *string_ranges,           // string ranges
	rax *numeric_ranges           // numerical reages
) {
	ASSERT(trees          != NULL);
	ASSERT(string_ranges  != NULL);
	ASSERT(numeric_ranges != NULL);

	uint count = array_len(trees);
	for(uint i = 0; i < count; i++) {
		const FT_FilterNode *tree = trees[i];
		if(tree->t == FT_N_PRED) {
			if(_predicateTreeToRange(tree, string_ranges, numeric_ranges)) {
				// managed to convert tree into range
				// discard tree and update loop index
				array_del_fast(trees, i);
				i--;
				count--;
			}
		}
	}
}

// tries to convert filter tree to a RediSearch query
// return true if tree was converted, false otherwise
// a conversion might fail if tree contains a none indexable type e.g. array
static bool _FilterTreeConditionToQueryNode
(
	RSQNode **root,            // [output] query node
	const FT_FilterNode *tree, // filter to convert
	RSIndex *idx               // queried index
) {
	ASSERT(idx     != NULL);
	ASSERT(root    != NULL);
	ASSERT(tree    != NULL);
	ASSERT(tree->t == FT_N_COND);

	*root = NULL; // initialize output to NULL

	// validate operator
	AST_Operator op = tree->cond.op;
	ASSERT(op == OP_OR || op == OP_AND);

	RSQNode *node  = NULL;
	RSQNode *left  = NULL;
	RSQNode *right = NULL;

	// create root node
	if(op == OP_OR) node = RediSearch_CreateUnionNode(idx);
	else node = RediSearch_CreateIntersectNode(idx, false);

	//--------------------------------------------------------------------------
	// convert left and right hand sides
	//--------------------------------------------------------------------------

	// process left branch
	bool res = _FilterTreeToQueryNode(&left, tree->cond.left, idx);

	// process right branch
	res &= _FilterTreeToQueryNode(&right, tree->cond.right, idx);

	ASSERT(left  != NULL);
	ASSERT(right != NULL);

	RediSearch_QueryNodeAddChild(node, left);
	RediSearch_QueryNodeAddChild(node, right);

	*root = node;
	return res;
}

// construct a RediSearch query to locate entities with an un-indexable
// field value
static void _NoneIndexableQuery
(
	RSQNode **root,
	RSIndex *idx,
	const char *field
) {
	// none indexable type, consult with the none indexed field
	RSQNode *parent = RediSearch_CreateTagNode(idx, INDEX_FIELD_NONE_INDEXED);
	RSQNode *child  = RediSearch_CreateTokenNode(idx, INDEX_FIELD_NONE_INDEXED,
			field);

	RediSearch_QueryNodeAddChild(parent, child);
	*root = parent;
}

// returns true if predicate filter been converted to an index query
static bool _FilterTreePredicateToQueryNode
(
	RSQNode **root,            // [output] query node
	const FT_FilterNode *tree, // filter to convert
	RSIndex *idx               // queried index
) {
	ASSERT(idx     != NULL);
	ASSERT(root    != NULL);
	ASSERT(tree    != NULL);
	ASSERT(tree->t == FT_N_PRED);

	// initialize root to NULL
	*root = NULL;

	// expecting left hand side to be an attribute access
	bool    res    = true;
	RSQNode *node  = NULL;
	char    *field = NULL;
	bool    attr   = AR_EXP_IsAttribute(tree->pred.lhs,  &field);

	ASSERT(attr == true);

	// validate const type
	SIValue v = AR_EXP_Evaluate(tree->pred.rhs, NULL);
	SIType  t = SI_TYPE(v);
	if(!(t & SI_INDEXABLE)) {
		// none indexable type, consult with the none indexed field
		_NoneIndexableQuery(root, idx, field);
		return false;
	}

	// validate operation, we can handle <, <=, =, >, >=
	AST_Operator op = tree->pred.op;
	ASSERT(op == OP_LT ||
		   op == OP_LE ||
		   op == OP_GT ||
		   op == OP_GE ||
		   op == OP_EQUAL);

	char type_aware_field_name[512];
	Index_RangeFieldName(type_aware_field_name, field, NULL);

	if(t & T_STRING) {
		switch(tree->pred.op) {
			case OP_LT:    // <
				node = RediSearch_CreateLexRangeNode(idx, type_aware_field_name,
						RSLEXRANGE_NEG_INF, v.stringval, 0, 0);
				break;
			case OP_LE:    // <=
				node = RediSearch_CreateLexRangeNode(idx, type_aware_field_name,
						RSLEXRANGE_NEG_INF, v.stringval, 0, 1);
				break;
			case OP_GT:    // >
				node = RediSearch_CreateLexRangeNode(idx, type_aware_field_name,
						v.stringval, RSLECRANGE_INF, 0, 0);
				break;
			case OP_GE:    // >=
				node = RediSearch_CreateLexRangeNode(idx, type_aware_field_name,
						v.stringval, RSLECRANGE_INF, 1, 0);
				break;
			case OP_EQUAL:  // ==
				node = RediSearch_CreateTokenNode(idx, type_aware_field_name,
						v.stringval);
				break;
			default:
				ASSERT(false && "unexpected operation");
		}

		RSQNode *parent = RediSearch_CreateTagNode(idx, type_aware_field_name);
		RediSearch_QueryNodeAddChild(parent, node);
		node = parent;
	} else {
		ASSERT(t & SI_NUMERIC || t == T_BOOL);
		double d = SI_GET_NUMERIC(v);
		switch(tree->pred.op) {
			case OP_LT:    // <
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						d, RSRANGE_NEG_INF, false, false);
				break;
			case OP_LE:    // <=
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						d, RSRANGE_NEG_INF, true, false);
				break;
			case OP_GT:    // >
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						RSRANGE_INF, d, false, false);
				break;
			case OP_GE:    // >=
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						RSRANGE_INF, d, false, true);
				break;
			case OP_EQUAL:  // ==
				node = RediSearch_CreateNumericNode(idx, type_aware_field_name,
						d, d, true, true);
				// TODO: remove when RediSearch INT64 indexing bug fixed
				if(t == T_INT64 && v.longval & 0x7FF0000000000000) {
					res = false;
				}
				break;
			default:
				ASSERT(false && "unexpected operation");
		}
	}

	*root = node;
	return res;
}

// returns true if 'tree' been converted into an index query, false otherwise
static bool _FilterTreeToQueryNode
(
	RSQNode **root,             // [output] query node
	const FT_FilterNode *tree,  // filter to convert into an index query
	RSIndex *idx                // queried index
) {
	ASSERT(idx  != NULL);
	ASSERT(root != NULL);
	ASSERT(tree != NULL);

	// initialize 'root' to NULL
	*root = NULL;

	// convert IN filter
	// 1. n.v IN [1,2,3]
	// 2. 8 IN n.v
	if(isInFilter(tree)) {
		AR_ExpNode *exp = FilterTree_getExpression(tree);
		AR_ExpNode *lhs = AR_EXP_getChild(exp, 0);
		AR_ExpNode *rhs = AR_EXP_getChild(exp, 1);

		// either lhs or rhs represent attribute access
		ASSERT(AR_EXP_IsAttribute(lhs, NULL) || AR_EXP_IsAttribute(rhs, NULL));

		bool res    = false;
		char *field = NULL;

		if(AR_EXP_IsAttribute(lhs, &field)) {
			// n.v IN [1,2,3]
			res = _FilterTreeToMultiValQueryNode(tree, idx, root);
		} else {
			// 8 IN n.v
			AR_EXP_IsAttribute(rhs, &field);
			res = _FilterTreeToInQueryNode(tree, idx, root);
		}

		// in case we were unable to convert the filter into a RediSearch query
		// e.g. [2] in n.v we'll use the 'non-indexable' RediSearch query
		if(!res) {
			_NoneIndexableQuery(root, idx, field);
		}

		return res;
	}

	if(isDistanceFilter(tree)) {
		*root = _FilterTreeToDistanceQueryNode(tree, idx);
		return true;
	}

	FT_FilterNodeType t = tree->t;

	if(t == FT_N_COND) {
		return _FilterTreeConditionToQueryNode(root, tree, idx);
	} else if(t == FT_N_PRED) {
		return _FilterTreePredicateToQueryNode(root, tree, idx);
	} else {
		ASSERT("unknown filter tree node type");
		return false;
	}
}

// construct a query tree from a filter tree
RSQNode *Index_BuildQueryTree
(
	FT_FilterNode **none_converted_filters,  // [out] none converted filters
	const Index idx,                         // index to query
	const FT_FilterNode *tree                // filter tree to convert
) {
	ASSERT(idx  != NULL);
	ASSERT(tree != NULL);
	ASSERT(none_converted_filters != NULL);

	RSIndex             *rsIdx  = Index_RSIndex(idx);
	RSQNode             **nodes = array_new(RSQNode*, 1);     // intermidate nodes
	const FT_FilterNode **trees = FilterTree_SubTrees(tree);  // individual subtrees

	//--------------------------------------------------------------------------
	// convert filters to numeric and string ranges
	//--------------------------------------------------------------------------

	rax *string_ranges  = raxNew();
	rax *numeric_ranges = raxNew();
	_compose_ranges(trees, string_ranges, numeric_ranges);
	if(raxSize(string_ranges) > 0 || raxSize(numeric_ranges) > 0) {
		RSQNode *ranges = _ranges_to_query_nodes(rsIdx, string_ranges, numeric_ranges);
		// TODO: check for empty node RediSearch_CreateEmptyNode
		array_append(nodes, ranges);
	}

	//--------------------------------------------------------------------------
	// convert remaining filters into RediSearch query nodes
	//--------------------------------------------------------------------------

	uint tree_count = array_len(trees);
	for(uint i = 0; i < tree_count; i++) {
		RSQNode *node = NULL;
		bool resolved_filter = _FilterTreeToQueryNode(&node, trees[i], rsIdx);

		if(node != NULL) {
			array_append(nodes, node);
		}

		if(resolved_filter) {
			// remove converted filter from filters array
			array_del_fast(trees, i);
			i--;
			tree_count--;
		}
	}

	//--------------------------------------------------------------------------
	// combine remaining filters
	//--------------------------------------------------------------------------

	// filters that can not be converted into an index query will be returned
	// to caller as a single filter tree
	// this might happen when the value compared against is a runtime value of
	// none indexable type e.g. array
	*none_converted_filters = FilterTree_Combine(trees, tree_count);

	RSQNode *root      = NULL;
	uint    node_count = array_len(nodes);

	// compose root query node by intersecting individual query nodes
	root = _concat_query_nodes(rsIdx, nodes, node_count);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	array_free(nodes);
	array_free(trees);
	raxFreeWithCallback(string_ranges, (void(*)(void *))StringRange_Free);
	raxFreeWithCallback(numeric_ranges, (void(*)(void *))NumericRange_Free);

	return root;
}

// construct a vector query tree
RSQNode *Index_BuildVectorQueryTree
(
	const Index idx,    // index to query
	const char *field,  // field to query
	const float *vec,   // query vector
	size_t nbytes,      // vector size in bytes
	int k			    // number of results to return
) {
	ASSERT(k      > 0);
	ASSERT(nbytes > 0);
	ASSERT(vec    != NULL);
	ASSERT(idx    != NULL);
	ASSERT(field  != NULL);

	// create a redisearch query node
	RSIndex *rsIdx = Index_RSIndex(idx);

	char type_aware_field_name[512];
	Index_VectorFieldName(type_aware_field_name, field);

	RSQNode *root = RediSearch_CreateVecSimNode(rsIdx, type_aware_field_name,
			(char*)vec, nbytes, k);
	
	return root;
}

// construct a unique constraint query tree
RSQNode *Index_BuildUniqueConstraintQuery
(
	const Index idx,           // index to query
	const SIValue *attr_vals,  // entity attributes to query
	AttributeID *attr_ids,     // constraint attribute ids
	uint8_t n                  // number of constraint attributes
) {
	ASSERT (n         > 0) ;
	ASSERT (idx       != NULL) ;
	ASSERT (attr_vals != NULL) ;
	ASSERT (attr_ids  != NULL) ;

	RSQNode *node;
	RSQNode *nodes[n];
	RSIndex *rsIdx = Index_RSIndex(idx);

	//--------------------------------------------------------------------------
	// construct query nodes
	//--------------------------------------------------------------------------

	for(uint8_t i = 0; i < n; i++) {
		AttributeID attr_id = attr_ids[i] ;
		const IndexField *f = Index_GetField (NULL, idx, attr_id) ;

		// field must exist and be of type INDEX_FLD_RANGE
		ASSERT (f != NULL) ;
		ASSERT (IndexField_GetType (f) & INDEX_FLD_RANGE) ;

		// use internal range field name
		const char *field = f->range_name ;

		// get current attribute from entity
		SIValue v = attr_vals[i] ;

		// create RediSearch query node according to entity attr type
		SIType t = SI_TYPE (v) ;
		ASSERT (t & (T_STRING | SI_NUMERIC | T_BOOL)) ;

		if (t & T_STRING) {
			node =
				RediSearch_CreateTagNode (rsIdx, field) ;
			RSQNode *child =
				RediSearch_CreateTagTokenNode (rsIdx, v.stringval) ;

			RediSearch_QueryNodeAddChild (node, child) ;
		} else {
			double d = SI_GET_NUMERIC (v) ;
			node =
				RediSearch_CreateNumericNode (rsIdx, field, d, d, true, true) ;
		}

		ASSERT (node != NULL) ;
		nodes[i] = node ;
	}

	//--------------------------------------------------------------------------
	// cancat filters
	//--------------------------------------------------------------------------

	RSQNode *root = node ;
	if (n > 1) {
		// intersection query node
		root = RediSearch_CreateIntersectNode (rsIdx, false) ;
		for (uint8_t i = 0; i < n; i++) {
			RediSearch_QueryNodeAddChild (root, nodes[i]) ;
		}
	}

	return root ;
}

