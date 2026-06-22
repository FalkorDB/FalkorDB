# Fix for Issue #636: Crash found in fuzzer

/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "ast_validations.h"
#include "ast_shared.h"
#include "../errors/errors.h"
#include "../arithmetic/arithmetic_expression.h"
#include "RG.h"
#include "../util/arr.h"
#include "../util/strcmp.h"
#include "../procedures/procedure.h"
#include "../arithmetic/aggregate_funcs/agg_funcs.h"

// forward declarations
static void _AST_GetDefinedIdentifiers(const cypher_astnode_t *node, rax *identifiers);
static void _AST_GetReferredIdentifiers(const cypher_astnode_t *node, rax *identifiers);

// check if a string is contained in a rax tree
static bool _IdentifierInRax(rax *rax, const char *str) {
	return raxFind(rax, (unsigned char *)str, strlen(str)) != raxNotFound;
}

// adds a string to a rax tree
static void _IdentifierAddToRax(rax *rax, const char *str) {
	raxInsert(rax, (unsigned char *)str, strlen(str), NULL, NULL);
}

// validate that the same alias in a variable-length path pattern does not have
// contradictory labels on both endpoints
static AST_Validation _ValidateVarLenPathEndpoints(const cypher_astnode_t *path) {
	uint path_len = cypher_ast_pattern_path_nelements(path);
	
	// iterate through relationship patterns in the path
	for(uint i = 1; i < path_len; i += 2) {
		const cypher_astnode_t *rel = cypher_ast_pattern_path_get_element(path, i);
		
		// check if this is a variable-length relationship
		if(cypher_astnode_type(rel) != CYPHER_AST_REL_PATTERN) continue;
		
		const cypher_astnode_t *var_len = cypher_ast_rel_pattern_get_varlength(rel);
		if(var_len == NULL) continue;
		
		// get the nodes on both sides of the var-length relationship
		const cypher_astnode_t *left_node = cypher_ast_pattern_path_get_element(path, i - 1);
		const cypher_astnode_t *right_node = cypher_ast_pattern_path_get_element(path, i + 1);
		
		// get identifiers for both nodes
		const cypher_astnode_t *left_id = cypher_ast_node_pattern_get_identifier(left_node);
		const cypher_astnode_t *right_id = cypher_ast_node_pattern_get_identifier(right_node);
		
		if(left_id == NULL || right_id == NULL) continue;
		
		const char *left_alias = cypher_ast_identifier_get_name(left_id);
		const char *right_alias = cypher_ast_identifier_get_name(right_id);
		
		// if both endpoints have the same alias
		if(strcmp(left_alias, right_alias) == 0) {
			// get labels for both nodes
			uint left_nlabels = cypher_ast_node_pattern_nlabels(left_node);
			uint right_nlabels = cypher_ast_node_pattern_nlabels(right_node);
			
			// if both have labels, check for conflicts
			if(left_nlabels > 0 && right_nlabels > 0) {
				// collect left labels
				rax *left_labels = raxNew();
				for(uint j = 0; j < left_nlabels; j++) {
					const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(left_node, j);
					const char *label_name = cypher_ast_label_get_name(label);
					_IdentifierAddToRax(left_labels, label_name);
				}
				
				// check if right labels match left labels
				bool has_common_label = false;
				for(uint j = 0; j < right_nlabels; j++) {
					const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(right_node, j);
					const char *label_name = cypher_ast_label_get_name(label);
					if(_IdentifierInRax(left_labels, label_name)) {
						has_common_label = true;
						break;
					}
				}
				
				raxFree(left_labels);
				
				// if labels are different, this is a contradictory pattern
				// that can never match - return empty result set rather than crash
				if(!has_common_label) {
					ErrorCtx_SetError(EMSG_SAME_ALIAS_DIFFERENT_LABELS, left_alias);
					return AST_INVALID;
				}
			}
		}
	}
	
	return AST_VALID;
}

// validate pattern paths in MATCH clauses
static AST_Validation _ValidateMatchPatternPaths(const cypher_astnode_t *match_clause) {
	const cypher_astnode_t *pattern = cypher_ast_match_get_pattern(match_clause);
	uint npaths = cypher_ast_pattern_npaths(pattern);
	
	for(uint i = 0; i < npaths; i++) {
		const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, i);
		if(_ValidateVarLenPathEndpoints(path) != AST_VALID) {
			return AST_INVALID;
		}
	}
	
	return AST_VALID;
}

// wrapper validation function that checks for var-length path contradictions
AST_Validation AST_ValidateVarLenPathPatterns(const cypher_astnode_t *root) {
	if(root == NULL) return AST_VALID;
	
	cypher_astnode_type_t type = cycypher_astnode_type(root);
	
	// recursively check all MATCH clauses
	if(type == CYPHER_AST_MATCH) {
		return _ValidateMatchPatternPaths(root);
	}
	
	// recurse into child nodes
	uint nchildren = cypher_astnode_nchildren(root);
	for(uint i = 0; i < nchildren; i++) {
		const cypher_astnode_t *child = cypher_astnode_get_child(root, i);
		if(AST_ValidateVarLenPathPatterns(child) != AST_VALID) {
			return AST_INVALID;
		}
	}
	
	return AST_VALID;
}