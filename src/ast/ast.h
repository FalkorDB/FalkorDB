/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"
#include "cypher-parser.h"
#include "../redismodule.h"
#include "ast_annotations_ctx_collection.h"
#include "../arithmetic/arithmetic_expression.h"

#define UNLIMITED UINT_MAX
#define IDENTIFIER_NOT_FOUND UINT_MAX

typedef enum {
	AST_VALID,
	AST_INVALID
} AST_Validation;

typedef struct {
	const cypher_astnode_t *root;                       // Root element of libcypher-parser AST
	rax *referenced_entities;                           // Mapping of the referenced entities.
	AST_AnnotationCtxCollection *anot_ctx_collection;   // Holds annotations contexts.
	bool free_root;                                     // The root should only be freed if this is a sub-AST we constructed
	uint *ref_count;                                    // A pointer to reference counter (for deletion).
	cypher_parse_result_t *parse_result;                // Query parsing output.
} AST;

// checks to see if libcypher-parser reported any errors
bool AST_ContainsErrors
(
	const cypher_parse_result_t *result
);

// reports first encountered error
// asserts if there are no errors!
void AST_ReportErrors
(
	const cypher_parse_result_t *result
);

AST_Validation AST_Validate_ParseResultRoot
(
	const cypher_parse_result_t *result,
	int *index
);

// make sure the parse result and the AST tree pass all validations
AST_Validation AST_Validate_Query
(
	const cypher_astnode_t *root
);

// validate query parameters parsing only
AST_Validation AST_Validate_QueryParams
(
	const cypher_parse_result_t *result
);

// checks if the parse result represents a read-only query
bool AST_ReadOnly
(
	const cypher_astnode_t *root
);

// checks if a query contains an ast-node corresponding to an eager operation
bool AST_IsEager
(
	const cypher_astnode_t *root
);

// checks to see if AST contains specified clause
bool AST_ContainsClause
(
	const AST *ast,
	cypher_astnode_type_t clause
);

// checks to see if an AST tree contains specified node type
bool AST_TreeContainsType
(
	const cypher_astnode_t *root,
	cypher_astnode_type_t clause
);

// returns all function (aggregated & none aggregated) mentioned in query
void AST_ReferredFunctions
(
	const cypher_astnode_t *root,
	rax *referred_funcs
);

// returns specified clause or NULL
// if 'clause_idx' is specified and requested clause type is found
// 'clause_idx' is set to the index of the returned clause
// otherwise 'clause_idx' isn't modified
const cypher_astnode_t *AST_GetClause
(
	const AST *ast,
	cypher_astnode_type_t clause_type,
	uint *clause_idx
);

// return clause at position 'i'
const cypher_astnode_t *AST_GetClauseByIdx
(
	const AST *ast,
	uint i
);

// returns the indexes into the AST of all instances of the given clause
uint *AST_GetClauseIndices
(
	const AST *ast,
	cypher_astnode_type_t clause_type
);

// returns the number of times the given clause appears in the AST
uint AST_GetClauseCount
(
	const AST *ast,
	cypher_astnode_type_t clause_type
);

// returns all instances of the given clause in the AST
const cypher_astnode_t **AST_GetClauses
(
	const AST *ast,
	cypher_astnode_type_t type
);

// returns an array (arr.h type) of all the nodes from a given type.
// note: array must be free after use
const cypher_astnode_t **AST_GetTypedNodes
(
	const cypher_astnode_t *ast,
	cypher_astnode_type_t type
);

// collect all aliases within `entity` subtree
void AST_CollectAliases
(
	const char ***aliases,
	const cypher_astnode_t *entity
);

AST *AST_Build
(
	cypher_parse_result_t *parse_result
);

AST *AST_NewSegment
(
	AST *master_ast,
	uint start_offset,
	uint end_offset
);

// returns a shallow copy of the original AST pointer with ref counter increased
AST *AST_ShallowCopy
(
	AST *orig
);

// populate the AST's map of all referenced aliases
void AST_BuildReferenceMap
(
	AST *ast,
	const cypher_astnode_t *project_clause
);

// annotate AST naming all anonymous graph entities
void AST_Enrich
(
	AST *ast
);

// returns true if the given alias is referenced within this AST segment
bool AST_AliasIsReferenced
(
	AST *ast,
	const char *alias
);

// returns true if the given identifier is used as an alias within this tree
bool AST_IdentifierIsAlias
(
	const cypher_astnode_t *root,
	const char *identifier
);

// convert an AST integer node (which is stored internally as a string) into an integer
long AST_ParseIntegerNode
(
	const cypher_astnode_t *int_node
);

// returns true if the given clause contains an aggregate function
bool AST_ClauseContainsAggregation
(
	const cypher_astnode_t *clause
);

// collect the aliases from a RETURN clause to populate ResultSet column names
const char **AST_BuildReturnColumnNames
(
	const cypher_astnode_t *return_clause
);

// collect the aliases from a CALL clause to populate ResultSet column names
const char **AST_BuildCallColumnNames
(
	const cypher_astnode_t *return_clause
);

// parse a query to construct an immutable AST
cypher_parse_result_t *parse_query
(
	const char *query  // query to parse
);

// parse a query parameter values only
// the remaining query string is set in the result body
cypher_parse_result_t *parse_params
(
	const char *query,
	const char **query_body
);

// free the immutable AST generated by the parser
void parse_result_free
(
	cypher_parse_result_t *parse_result
);

// returns the ast annotation context collection of the AST
AST_AnnotationCtxCollection *AST_GetAnnotationCtxCollection
(
	AST *ast
);

const char *AST_ToString
(
	const cypher_astnode_t *node
);

void AST_Free
(
	AST *ast
);
