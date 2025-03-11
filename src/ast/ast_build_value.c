
#include "RG.h"
#include "ast_build_value.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"

#include <sys/errno.h>

// convert an AST node into an integer SIValue
static void _SIValue_FromIntegerNode
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(v    != NULL);
	ASSERT(node != NULL);

	const char *value_str = cypher_ast_integer_get_valuestr(node);

	errno = 0;
	char *endptr = NULL;
	int64_t l = strtol(value_str, &endptr, 0);

	if(endptr[0] != 0) {
		// failed to convert integer value;
		// set compile-time error to be raised later.
		ErrorCtx_SetError(EMSG_INVALID_NUMERIC, value_str);
		*v = SI_NullVal();
		return;
	}

	if(errno == ERANGE) {
		ErrorCtx_SetError(EMSG_INTEGER_OVERFLOW, value_str);
		*v = SI_NullVal();
		return;
	}

	*v = SI_LongVal(l);
}

// convert an AST node into a float SIValue
static void _SIValue_FromFloatNode
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(v    != NULL);
	ASSERT(node != NULL);

	errno = 0; // reset before strtod
	char *endptr = NULL;
	const char *value_str = cypher_ast_float_get_valuestr(node);
	double d = strtod(value_str, &endptr);

	if(endptr[0] != 0) {
		// failed to convert integer value;
		// set compile-time error to be raised later
		ErrorCtx_SetError(EMSG_INVALID_NUMERIC, value_str);
		*v = SI_NullVal();
		return;
	}

	if(errno == ERANGE) {
		ErrorCtx_SetError(EMSG_FLOAT_OVERFLOW, value_str);
		*v = SI_NullVal();
		return;
	}

	*v = SI_DoubleVal(d);
}

// convert an AST node into a string SIValue
static void _SIValue_FromStringNode
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(v    != NULL);
	ASSERT(node != NULL);

	const char *value_str = cypher_ast_string_get_value(node);
	*v = SI_ConstStringVal((char *)value_str);
}

// convert an AST node into an array SIValue
static void _SIValue_FromCollectionNode
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(v    != NULL);
	ASSERT(node != NULL);

	uint n = cypher_ast_collection_length(node);
	SIValue collection = SI_Array(n);

	for(uint i = 0; i < n; i++) {
		const cypher_astnode_t *elem_node = cypher_ast_collection_get(node, i);
		SIValue _v;
		AST_ToSIValue(elem_node, &_v);
		SIArray_AppendAsOwner(&collection, &_v);
	}

	*v = collection;
}

// convert an AST node into a map SIValue
static void _SIValue_FromMapNode
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(v    != NULL);
	ASSERT(node != NULL);

	uint n = cypher_ast_map_nentries(node);
	SIValue map = SI_Map(n);

	// process each key value pair
	for(uint i = 0; i < n; i++) {
		const cypher_astnode_t *key_node = cypher_ast_map_get_key(node, i);
		const char *key = cypher_ast_prop_name_get_value(key_node);
		const cypher_astnode_t *val = cypher_ast_map_get_value(node, i);

		SIValue _key = SI_ConstStringVal((char *)key);
		SIValue _val;
		AST_ToSIValue(val, &_val);
		Map_AddNoClone(&map, _key, _val);
	}

	*v = map;
}

// convert an AST node into an SIValue
// e.g. -4.2
static void _SIValue_FromUnaryOp
(
	const cypher_astnode_t *node,  // AST node
	SIValue *v                     // [output]
) {
	ASSERT(node != NULL);

	const cypher_astnode_t *arg =
		cypher_ast_unary_operator_get_argument(node);
	const cypher_operator_t *operator =
		cypher_ast_unary_operator_get_operator(node);

	int a = 1;
	if(operator == CYPHER_OP_UNARY_MINUS) {
		a = -1;
	} else if(operator != CYPHER_OP_UNARY_PLUS) {
		// unsupported operator
		Error_UnsupportedASTNodeType(node);
	}

	AST_ToSIValue(arg, v);
	*v = SIValue_Multiply(SI_LongVal(a), *v);
}

// convert an AST node into an SIValue
void AST_ToSIValue
(
	const cypher_astnode_t *node,  // AST node to convert
	SIValue *v                     // [output]
) {
	ASSERT(node != NULL);

	const cypher_astnode_type_t t = cypher_astnode_type(node);

	// based on the AST node type convert to the appropriate type
	if(t == CYPHER_AST_INTEGER) {
		_SIValue_FromIntegerNode(node, v);
	} else if(t == CYPHER_AST_FLOAT) {
		_SIValue_FromFloatNode(node, v);
	} else if(t == CYPHER_AST_STRING) {
		_SIValue_FromStringNode(node, v);
	} else if(t == CYPHER_AST_TRUE) {
		*v = SI_BoolVal(true);
	} else if(t == CYPHER_AST_FALSE) {
		*v = SI_BoolVal(false);
	} else if(t == CYPHER_AST_NULL) {
		*v = SI_NullVal();
	} else if(t == CYPHER_AST_COLLECTION) {
		_SIValue_FromCollectionNode(node, v);
	} else if(t == CYPHER_AST_MAP) {
		_SIValue_FromMapNode(node, v);
	} else if(t == CYPHER_AST_UNARY_OPERATOR) {
		_SIValue_FromUnaryOp(node, v);
	} else {
		// unhandled types
		Error_UnsupportedASTNodeType(node);
		*v = SI_NullVal();
	}
}

