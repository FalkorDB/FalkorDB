/*
 * Copyright Redis Ltd. 2024 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or the Server Side Public License v1 (SSPLv1).
 */

#include "ast.h"
#include "../errors.h"
#include "../util/arr.h"
#include "../util/dict.h"

// Validate AST structure for semantic correctness
// Returns non-zero if validation fails
int AST_Validate(AST *ast) {
	if (!ast) return 0;
	
	// Validate match patterns for duplicate aliases with conflicting labels
	int ret = AST_ValidateMatchPatterns(ast);
	if (ret != 0) return ret;
	
	return 0;
}

// Detect issues where the same node alias is used with different labels
// Issue #636: Crash found in fuzzer - same alias with conflicting labels
int AST_ValidateMatchPatterns(AST *ast) {
	if (!ast || !ast->root) return 0;
	
	// Dictionary to track node aliases and their labels
	dict *node_labels = dictCreate(&dictTypeHeapStringKey, NULL);
	
	int ret = _ValidatePatternNode(ast->root, node_labels);
	
	dictRelease(node_labels);
	return ret;
}

// Helper function to recursively validate pattern nodes
static int _ValidatePatternNode(AST_Node *node, dict *node_labels) {
	if (!node) return 0;
	
	// Check for conflicting label definitions on same alias
	if (node->type == N_PATTERN_PATH) {
		if (_ValidateNodeAliasLabels(node, node_labels) != 0) {
			return 1;
		}
	}
	
	// Recursively validate child nodes
	if (node->children) {
		uint child_count = array_len(node->children);
		for (uint i = 0; i < child_count; i++) {
			if (_ValidatePatternNode(node->children[i], node_labels) != 0) {
				return 1;
			}
		}
	}
	
	return 0;
}

// Validate that aliases don't have conflicting labels
static int _ValidateNodeAliasLabels(AST_Node *pattern, dict *node_labels) {
	if (!pattern) return 0;
	
	// Extract all node entities and their labels from the pattern
	uint node_count = array_len(pattern->children);
	for (uint i = 0; i < node_count; i++) {
		AST_Node *entity = pattern->children[i];
		if (entity->type == N_NODE_PATTERN) {
			const char *alias = entity->string_val;
			if (alias && strlen(alias) > 0) {
				// Get or create label set for this alias
				dictEntry *entry = dictFind(node_labels, (char *)alias);
				
				if (!entry) {
					// First occurrence of this alias - record its labels
					array labels = entity->vector;
					dictAdd(node_labels, (char *)alias, labels);
				} else {
					// Alias seen before - check for label conflicts
					array prev_labels = (array)entry->v.val;
					array curr_labels = entity->vector;
					
					if (_LabelsConflict(prev_labels, curr_labels)) {
						// Same alias used with different labels - error
						return 1;
					}
				}
			}
		}
	}
	
	return 0;
}

// Check if two label sets conflict (are non-identical)
static int _LabelsConflict(array labels1, array labels2) {
	uint len1 = array_len(labels1);
	uint len2 = array_len(labels2);
	
	// Different number of labels is a conflict
	if (len1 != len2) return 1;
	
	// Check each label matches
	for (uint i = 0; i < len1; i++) {
		AST_Node *label1 = labels1[i];
		AST_Node *label2 = labels2[i];
		
		if (!label1 || !label2) return 1;
		if (strcmp(label1->string_val, label2->string_val) != 0) {
			return 1; // Labels differ
		}
	}
	
	return 0; // Labels are identical
}
