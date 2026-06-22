/*
 * Copyright Redis Ltd. 2024 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or the Server Side Public License v1 (SSPLv1).
 */

#include "ast.h"
#include "../errors.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include <libcypher-parser.h>

// Visitor context for tracking node aliases and their labels
typedef struct {
	dict *node_labels;  // Maps alias -> label array
	int error;          // Flag if validation failed
	ErrorCtx *error_ctx; // Error context for reporting
} validation_context_t;

// Forward declaration
static VISITOR_STRATEGY _visit_node(const cypher_astnode_t *n, bool start,
                                     ast_visitor *visitor);

// Initialize validation mappings
bool AST_ValidationsMappingInit(void) {
	return true;
}

// Default visitor that processes all node types
VISITOR_STRATEGY _default_visit(const cypher_astnode_t *n, bool start,
                                 ast_visitor *visitor) {
	if (!start || !n) return VISITOR_CONTINUE;
	
	validation_context_t *ctx = (validation_context_t *)visitor->data;
	cypher_astnode_type_t node_type = cypher_astnode_type(n);
	
	// Check pattern paths for conflicting node aliases
	if (node_type == CYPHER_AST_PATTERN_PATH) {
		uint num_elements = cypher_ast_pattern_path_nrelationships(n);
		
		// Iterate through all node patterns in the path
		for (uint i = 0; i < num_elements; i++) {
			const cypher_astnode_t *rel = cypher_ast_pattern_path_get_relationship(n, i);
			if (!rel) continue;
			
			// Check both nodes in the relationship
			const cypher_astnode_t *start_node = cypher_ast_pattern_path_get_start_node(n, i);
			const cypher_astnode_t *end_node = cypher_ast_pattern_path_get_end_node(n, i);
			
			if (start_node && cypher_astnode_type(start_node) == CYPHER_AST_NODE_PATTERN) {
				if (_validate_node_pattern(start_node, ctx->node_labels, ctx->error_ctx) != 0) {
					ctx->error = 1;
					return VISITOR_BREAK;
				}
			}
			
			if (end_node && cypher_astnode_type(end_node) == CYPHER_AST_NODE_PATTERN) {
				if (_validate_node_pattern(end_node, ctx->node_labels, ctx->error_ctx) != 0) {
					ctx->error = 1;
					return VISITOR_BREAK;
				}
			}
		}
	}
	
	return VISITOR_CONTINUE;
}

// Validate a single node pattern for conflicting labels
static int _validate_node_pattern(const cypher_astnode_t *node_pattern,
                                   dict *node_labels, ErrorCtx *error_ctx) {
	if (!node_pattern) return 0;
	
	// Get the node identifier (alias)
	const cypher_astnode_t *identifier = cypher_ast_node_pattern_get_identifier(node_pattern);
	if (!identifier) return 0;
	
	const char *alias = cypher_ast_identifier_get_name(identifier);
	if (!alias || strlen(alias) == 0) return 0;
	
	// Get the labels for this node
	uint label_count = cypher_ast_node_pattern_nlabels(node_pattern);
	
	dictEntry *entry = dictFind(node_labels, (char *)alias);
	
	if (!entry) {
		// First occurrence - store the labels for this alias
		array labels = array_new(cypher_astnode_t *, label_count);
		if (!labels) return 0;
		
		for (uint i = 0; i < label_count; i++) {
			const cypher_astnode_t *label = cypher_ast_node_pattern_get_label(node_pattern, i);
			if (label) {
				array_append(labels, label);
			}
		}
		
		if (dictAdd(node_labels, (char *)alias, labels) != DICT_OK) {
			array_free(labels);
			return 1;
		}
	} else {
		// Alias already seen - check for label conflicts
		array prev_labels = (array)entry->v.val;
		
		if (_labels_conflict(prev_labels, node_pattern) != 0) {
			// Conflicting labels on same alias
			if (error_ctx) {
				ErrorCtx_SetError(error_ctx,
					"Node alias '%s' used with conflicting labels in pattern",
					alias);
			}
			return 1;
		}
	}
	
	return 0;
}

// Check if stored labels conflict with current node's labels (unordered set comparison)
static int _labels_conflict(array stored_labels, const cypher_astnode_t *node) {
	if (!node) return 0;
	
	uint stored_len = array_len(stored_labels);
	uint curr_len = cypher_ast_node_pattern_nlabels(node);
	
	// Different number of labels is a conflict
	if (stored_len != curr_len) return 1;
	
	// Check each label in stored_labels exists in current node's labels (set comparison)
	for (uint i = 0; i < stored_len; i++) {
		const cypher_astnode_t *stored_label = (const cypher_astnode_t *)array_index(stored_labels, i);
		if (!stored_label) return 1;
		
		const char *stored_name = cypher_ast_label_get_name(stored_label);
		if (!stored_name) return 1;
		
		bool found = false;
		for (uint j = 0; j < curr_len; j++) {
			const cypher_astnode_t *curr_label = cypher_ast_node_pattern_get_label(node, j);
			if (curr_label) {
				const char *curr_name = cypher_ast_label_get_name(curr_label);
				if (curr_name && strcmp(stored_name, curr_name) == 0) {
					found = true;
					break;
				}
			}
		}
		if (!found) return 1;  // Label from stored_labels not found in current labels
	}
	
	return 0;  // Labels are identical (order-independent)
}