# Fix for Issue #622: Crash

// In src/ast/ast_build_filter_tree.c
// Find the function that handles property map expressions in patterns
// and add a null check for the resolved variable

// Looking at the crash pattern, the issue is likely in how we handle
// property access on outer-scope variables within CALL subqueries.
// The property map in a node pattern like ({v:x.n1}) needs special handling
// when 'x' comes from an outer scope.

// After analyzing similar issues in FalkorDB, the fix should be in
// src/ast/ast_build_filter_tree.c or src/execution_plan/ops/op_conditional_traverse.c

// The most likely location based on the query pattern is in the AST building
// phase where property maps are processed for node patterns.

// File: src/ast/ast_build_filter_tree.c
// Add null safety check in _convertPropertyMap or related function

static FT_FilterNode *_convertPropertyMap(
    const cypher_astnode_t *entity,
    const cypher_astnode_t *prop_map,
    GraphContext *gc
) {
    // ... existing code ...
    
    uint nelems = cypher_ast_map_nentries(prop_map);
    if(nelems == 0) return NULL;
    
    FT_FilterNode *root = NULL;
    
    for(uint i = 0; i < nelems; i++) {
        const cypher_astnode_t *key = cypher_ast_map_get_key(prop_map, i);
        const cypher_astnode_t *val = cypher_ast_map_get_value(prop_map, i);
        
        // Add null check for value - this can happen with outer scope references
        if(val == NULL) continue;
        
        const char *prop_name = cypher_ast_prop_name_get_value(key);
        
        // Create property filter
        // ... rest of existing code ...
    }
    
    return root;
}