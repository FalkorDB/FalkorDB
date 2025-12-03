/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "entity_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../datatypes/map.h"
#include "../../errors/errors.h"
#include "../../datatypes/array.h"
#include "../../graph/graphcontext.h"
#include "../../datatypes/datatypes.h"
#include "../../graph/entities/node.h"
#include "../../graph/entities/edge.h"
#include "../../graph/entities/graph_entity.h"

// returns the id of a relationship or node
SIValue AR_ID(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	GraphEntity *graph_entity = (GraphEntity *)argv[0].ptrval;
	return SI_LongVal(ENTITY_GET_ID(graph_entity));
}

// returns an array of string representations of each label of a node
SIValue AR_LABELS(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();

	Node *node = argv[0].ptrval;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	// retrieve node labels
	uint label_count;
	NODE_GET_LABELS(gc->g, node, label_count);
	SIValue res = SI_Array(label_count);

	for(uint i = 0; i < label_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
		ASSERT(s != NULL);
		const char *name = Schema_GetName(s);
		SIArray_Append(&res, SI_ConstStringVal(name));
	}

	return res;
}

// returns true if input node contains all specified labels, otherwise false
SIValue AR_HAS_LABELS(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();

	bool         res    = true;
	Node         *node  = argv[0].ptrval;
	SIValue      labels = argv[1];
	EntityID     id     = ENTITY_GET_ID(node);
	GraphContext *gc    = QueryCtx_GetGraphCtx();
	Graph        *g     = gc->g;

	// iterate over given labels
	uint32_t labels_length = SIArray_Length(labels);
	for (uint32_t i = 0; i < labels_length; i++) {
		SIValue label_value = SIArray_Get(labels, i);
		if(!(SI_TYPE(label_value) & T_STRING)) {
			Error_SITypeMismatch(label_value, T_STRING);
			return SI_NullVal();
		}
		char *label = label_value.stringval;
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);

		// validate schema exists
		if(!s) {
			res = false;
			break;
		}

		// validate label is set
		Delta_Matrix M = Graph_GetLabelMatrix(g, Schema_GetID(s));
		ASSERT(M != NULL);

		if(Delta_Matrix_isStoredElement(M, id, id) == GrB_NO_VALUE) {
			res = false;
			break;
		}
	}
	
	return SI_BoolVal(res);
}

/* returns a string representation of the type of a relation. */
SIValue AR_TYPE(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	char *type = "";
	Edge *e = argv[0].ptrval;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	int id = Edge_GetRelationID(e);
	if(id != GRAPH_NO_RELATION) type = gc->relation_schemas[id]->name;
	return SI_ConstStringVal(type);
}

/* returns the start node of a relationship. */
SIValue AR_STARTNODE(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	Edge *e = argv[0].ptrval;
	NodeID start_id = Edge_GetSrcNodeID(e);
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Node *src = rm_malloc(sizeof(Node));
	*src = GE_NEW_NODE();
	// Retrieve the node from the graph.
	Graph_GetNode(gc->g, start_id, src);
	SIValue si_node = SI_Node(src);
	// Mark this value as a heap allocation so that it gets freed properly.
	SIValue_SetAllocationType(&si_node, M_SELF);
	return si_node;
}

/* returns the end node of a relationship. */
SIValue AR_ENDNODE(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	Edge *e = argv[0].ptrval;
	NodeID end_id = Edge_GetDestNodeID(e);
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Node *dest = rm_malloc(sizeof(Node));
	*dest = GE_NEW_NODE();
	// Retrieve the node from the graph.
	Graph_GetNode(gc->g, end_id, dest);
	SIValue si_node = SI_Node(dest);
	// Mark this value as a heap allocation so that it gets freed properly.
	SIValue_SetAllocationType(&si_node, M_SELF);
	return si_node;
}

/* returns true if the specified property exists in the node, or relationship. */
SIValue AR_EXISTS(SIValue *argv, int argc, void *private_data) {
	/* MATCH (n) WHERE EXISTS(n.name) RETURN n
	 * If property n.name does not exists
	 * SIValue representing NULL is returned.
	 * if n.name exists its value can not be NULL. */
	if(SIValue_IsNull(argv[0])) return SI_BoolVal(0);
	return SI_BoolVal(1);
}

// returns node incoming/outgoing degree
static SIValue _AR_NodeDegree
(
	SIValue *argv,
	int argc,
	GRAPH_EDGE_DIR dir  // edge direction
) {
	ASSERT(SI_TYPE(argv[0]) != T_NULL);

	Node          *n     = (Node*)argv[0].ptrval;
	uint64_t      count  = 0;
	GraphContext  *gc    = QueryCtx_GetGraphCtx();

	if(argc > 1) {
		// we're interested in specific relationship type(s)

		// get labels array from input arguments, but removing duplicates
		SIValue labels = SI_EmptyArray();
		if(SI_TYPE(argv[1]) & T_STRING) {
			// validate signature function(NODE, STR_0, STR_1, ... STR_N)
			for(int i = 1; i < argc; i++) {
				if(SI_TYPE(argv[i]) & T_STRING) {
					if(SIArray_ContainsValue(labels, argv[i], NULL) == false) {
						SIArray_Append(&labels, argv[i]);
					}
				} else {
					Error_SITypeMismatch(argv[i], T_STRING);
				}
			}
		} else if (SI_TYPE(argv[1]) == T_ARRAY) {
			if(argc > 2) {
				ErrorCtx_SetError(EMSG_NODE_DEGREE_ARGUMENTS, argc);
			}
			// validate signature function(NODE, ARRAY_OF_STRINGS)
			uint len = SIArray_Length(argv[1]);
			for(int j = 0; j < len; j++) {
				SIValue elem = SIArray_Get(argv[1], j);
				if(!(SI_TYPE(elem) & T_STRING)) {
					SIArray_Free(labels);
					Error_SITypeMismatch(elem, T_STRING);
					return SI_NullVal();
				}
				if(SIArray_ContainsValue(labels, elem, NULL) == false) {
					SIArray_Append(&labels, elem);
				}
			}
		}
		uint len = SIArray_Length(labels);
		for(int i = 0; i < len; i++) {
			SIValue elem = SIArray_Get(labels, i);
			const char *label = elem.stringval;
			// make sure relationship exists.
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
			if(s == NULL) {
				continue;
			}

			// count edges
			count += Graph_GetNodeDegree(gc->g, n, dir, s->id);
		}
		SIArray_Free(labels);
	} else {
		// get all relations, regardless of their type.
		count = Graph_GetNodeDegree(gc->g, n, dir, GRAPH_NO_RELATION);
	}

	SIValue res = SI_LongVal(count);
	return res;
}

// returns the number of incoming edges for given node
SIValue AR_INCOMEDEGREE
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) {
		return SI_NullVal();
	}

	return _AR_NodeDegree(argv, argc, GRAPH_EDGE_DIR_INCOMING);
}

// returns the number of outgoing edges for given node
SIValue AR_OUTGOINGDEGREE
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) {
		return SI_NullVal();
	}

	return _AR_NodeDegree(argv, argc, GRAPH_EDGE_DIR_OUTGOING);
}

SIValue AR_PROPERTY
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// return NULL for missing graph entity
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();

	// AR_PROPERTY may be invoked from AR_SUBSCRIPT in a case like:
	// WITH {val: 5} AS map RETURN map["val"]
	// As such, we need to validate the argument's type independently of the invocation validation.
	if(!(SI_TYPE(argv[1]) & T_STRING)) {
		// String indexes are only permitted on maps, not arrays.
		Error_SITypeMismatch(argv[1], T_STRING);
		return SI_NullVal();
	}

	// inputs:
	// argv[0] - node/edge/map/point
	// argv[1] - property string
	// argv[2] - property index

	//--------------------------------------------------------------------------
	// Process inputs
	//--------------------------------------------------------------------------

	SIValue val;
	SIValue *p_val;
	SIValue obj = argv[0];
	SIValue key = argv[1];
	const char *key_str = key.stringval;

	SIType t = SI_TYPE(obj);
	switch(t) {
		case T_NODE:
		case T_EDGE:
		{
			// retrieve entity property
			GraphEntity *graph_entity = (GraphEntity *)obj.ptrval;
			const char *prop_name     = argv[1].stringval;
			AttributeID prop_idx      = argv[2].longval;

			// we have the property string, attempt to look up the index now
			if(prop_idx == ATTRIBUTE_ID_NONE) {
				GraphContext *gc = QueryCtx_GetGraphCtx();
				prop_idx = GraphContext_GetAttributeID(gc, prop_name);
			}

			// retrieve the property
			p_val = GraphEntity_GetProperty(graph_entity, prop_idx);
			return SI_ConstValue(p_val);
		}

		case T_MAP:
			// retrieve map key
			Map_Get(obj, key, &val);

			// Return a volatile copy of the value, as it may be heap-allocated.
			return SI_ShareValue(val);

		case T_POINT:
			// retrieve property key 
			return Point_GetCoordinate(obj, key);

		case T_DATETIME:
		{
			// retrieve datetime component e.g. year, month, hour
			int comp;
			bool found = DateTime_getComponent(&obj, key_str, &comp);

			if(found == false) {
				ErrorCtx_SetError("unknown datetime component %s", key_str);
				return SI_NullVal();	
			}

			return SI_LongVal(comp);
		}

		case T_DATE:
		{
			// retrieve date component e.g. year, month, day
			int comp;
			bool found = Date_getComponent(&obj, key_str, &comp);

			if(found == false) {
				ErrorCtx_SetError("unknown date component %s", key_str);
				return SI_NullVal();
			}

			return SI_LongVal(comp);
		}

		case T_TIME:
		{
			// retrieve time component e.g. hour, minute, second
			int comp;
			bool found = Time_getComponent(&obj, key_str, &comp);

			if(found == false) {
				ErrorCtx_SetError("unknown time component %s", key_str);
				return SI_NullVal();
			}

			return SI_LongVal(comp);
		}

		case T_DURATION:
		{
			// retrieve duration component e.g. years, months, hours, minutes
			float comp;
			bool found = Duration_getComponent(&obj, key_str, &comp);

			if(found == false) {
				ErrorCtx_SetError("unknown duration component %s", key_str);
				return SI_NullVal();
			}

			return SI_DoubleVal(comp);
		}

		default:
			// unexpected type
			return SI_NullVal();
	}
}

SIValue AR_TYPEOF(SIValue *argv, int argc, void *private_data) {
	return SI_ConstStringVal(SIType_ToString(SI_TYPE(argv[0])));
}

void Register_EntityFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_NODE | T_EDGE);
	ret_type = T_NULL | T_INT64;
	func_desc = AR_FuncDescNew("id", AR_ID, 1, 1, types, ret_type, false, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_NODE);
	ret_type = T_NULL | T_ARRAY;
	func_desc = AR_FuncDescNew("labels", AR_LABELS, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_NODE);
	array_append(types, T_ARRAY);
	ret_type = T_NULL | T_BOOL;
	func_desc = AR_FuncDescNew("hasLabels", AR_HAS_LABELS, 2, 2, types,
			ret_type, false, false, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_EDGE);
	ret_type = T_NULL | T_STRING;
	func_desc = AR_FuncDescNew("type", AR_TYPE, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_EDGE);
	ret_type = T_NULL | T_NODE;
	func_desc = AR_FuncDescNew("startNode", AR_STARTNODE, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_EDGE);
	ret_type = T_NULL | T_NODE;
	func_desc = AR_FuncDescNew("endNode", AR_ENDNODE, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | SI_ALL);
	ret_type = T_NULL | T_BOOL;
	func_desc = AR_FuncDescNew("exists", AR_EXISTS, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_NODE);
	array_append(types, T_STRING | T_ARRAY);
	ret_type = T_NULL | T_INT64;
	func_desc = AR_FuncDescNew("indegree", AR_INCOMEDEGREE, 1, VAR_ARG_LEN,
			types, ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_NODE);
	array_append(types, T_STRING | T_ARRAY);
	ret_type = T_NULL | T_INT64;
	func_desc = AR_FuncDescNew("outdegree", AR_OUTGOINGDEGREE, 1, VAR_ARG_LEN,
			types, ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_NULL | T_NODE | T_EDGE | T_MAP | T_POINT | SI_TEMPORAL);
	array_append(types, T_STRING);
	array_append(types, T_INT64);
	ret_type = SI_ALL;
	func_desc = AR_FuncDescNew("property", AR_PROPERTY, 3, 3, types, ret_type,
			true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | SI_ALL);
	ret_type = T_STRING;
	func_desc = AR_FuncDescNew("typeof", AR_TYPEOF, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}

