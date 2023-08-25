/* vi:set ts=4 sw=4 expandtab:
 *
 * Copyright 2016, Chris Leishman (http://github.com/cleishm)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CYPHER_PARSER_ASTNODE_H
#define CYPHER_PARSER_ASTNODE_H

#include "cypher-parser.h"
#include "annotation.h"
#include "ast.h"
#include "util.h"
#include <assert.h>
#include <limits.h>


struct cypher_astnode_vt
{
    const struct cypher_astnode_vt **parents;
    unsigned int nparents;
    const char *name;
    ssize_t (*detailstr)(const cypher_astnode_t *self, char *str, size_t size);
    void (*release)(cypher_astnode_t *self);
    cypher_astnode_t *(*clone)(const cypher_astnode_t *self,
            cypher_astnode_t **children);
};


struct cypher_astnode
{
    cypher_astnode_type_t type;
    cypher_astnode_t **children;
    unsigned int nchildren;
    struct cypher_input_range range;
    unsigned int ordinal;
    struct cypher_astnode_annotation *annotations;
};


int cypher_astnode_init(cypher_astnode_t *node, cypher_astnode_type_t type,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range);

void cypher_astnode_release(cypher_astnode_t *node);

ssize_t cypher_astnode_detailstr(const cypher_astnode_t *node, char *str,
        size_t size);


ssize_t snprint_sequence(char *str, size_t size,
        const cypher_astnode_t * const *elements, unsigned int nelements);


#define REQUIRE_TYPE(ptr, type, res) \
        REQUIRE(cypher_astnode_instanceof(ptr, type), res)

#define REQUIRE_TYPE_OPTIONAL(ptr, type, res) \
        REQUIRE(ptr == NULL || cypher_astnode_instanceof(ptr, type), res)

#define REQUIRE_TYPE_ALL(collection, size, type, res) \
    do { \
        REQUIRE((size == 0) || (collection != NULL), res); \
        unsigned int i = size; \
        while (i-- > 0) { \
            REQUIRE(cypher_astnode_instanceof(collection[i], type), res); \
        } \
    } while(0)

#define REQUIRE_CONTAINS(collection, size, val, res) \
    do { \
        REQUIRE((size > 0) && (collection != NULL), res); \
        unsigned int i = 0; \
        while (i < size && collection[i] != val) \
            i++; \
        REQUIRE(i < size, res); \
    } while(0)

#define REQUIRE_CONTAINS_OPTIONAL(collection, size, val, res) \
    do { \
        if (val != NULL) { \
            REQUIRE_CONTAINS(collection, size, val, res); \
        } \
    } while(0)

#define REQUIRE_CONTAINS_ALL(collection, size, vals, nvals, res) \
    do { \
        REQUIRE(size >= nvals, res); \
        unsigned int j = size; \
        while (j < size) { \
            REQUIRE_CONTAINS(collection, size, vals[j], res); \
        } \
    } while(0)

#define REQUIRE_CHILD(collection, size, val, type, res) \
    do { \
        REQUIRE_TYPE(val, type, res); \
        REQUIRE_CONTAINS(collection, size, val, res); \
    } while (0)

#define REQUIRE_CHILD_OPTIONAL(collection, size, val, type, res) \
    do { \
        if (val != NULL) { \
            REQUIRE_CHILD(collection, size, val, type, res); \
        } \
    } while (0)

#define REQUIRE_CHILD_ALL(collection, size, vals, nvals, type, res) \
    do { \
        REQUIRE_TYPE_ALL(vals, nvals, type, res); \
        REQUIRE_CONTAINS_ALL(collection, size, vals, nvals, res); \
    } while (0)


static inline int child_index(const cypher_astnode_t *node,
        const cypher_astnode_t *child)
{
    unsigned int i = 0;
    while (i < node->nchildren && node->children[i] != child)
        i ++;
    assert(i < node->nchildren);
    return i;
}


extern const struct cypher_astnode_vt cypher_statement_astnode_vt;
extern const struct cypher_astnode_vt cypher_statement_option_astnode_vt;
extern const struct cypher_astnode_vt cypher_cypher_option_astnode_vt;
extern const struct cypher_astnode_vt cypher_cypher_option_param_astnode_vt;
extern const struct cypher_astnode_vt cypher_explain_option_astnode_vt;
extern const struct cypher_astnode_vt cypher_profile_option_astnode_vt;
extern const struct cypher_astnode_vt cypher_schema_command_astnode_vt;
extern const struct cypher_astnode_vt cypher_create_node_props_index_astnode_vt;
extern const struct cypher_astnode_vt cypher_create_pattern_props_index_astnode_vt;
extern const struct cypher_astnode_vt cypher_drop_props_index_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_create_node_prop_constraint_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_drop_node_prop_constraint_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_create_rel_prop_constraint_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_drop_rel_prop_constraint_astnode_vt;
extern const struct cypher_astnode_vt cypher_query_astnode_vt;
extern const struct cypher_astnode_vt cypher_query_option_astnode_vt;
extern const struct cypher_astnode_vt cypher_using_periodic_commit_astnode_vt;
extern const struct cypher_astnode_vt cypher_query_clause_astnode_vt;
extern const struct cypher_astnode_vt cypher_load_csv_astnode_vt;
extern const struct cypher_astnode_vt cypher_start_astnode_vt;
extern const struct cypher_astnode_vt cypher_start_point_astnode_vt;
extern const struct cypher_astnode_vt cypher_node_index_lookup_astnode_vt;
extern const struct cypher_astnode_vt cypher_node_index_query_astnode_vt;
extern const struct cypher_astnode_vt cypher_node_id_lookup_astnode_vt;
extern const struct cypher_astnode_vt cypher_all_nodes_scan_astnode_vt;
extern const struct cypher_astnode_vt cypher_rel_index_lookup_astnode_vt;
extern const struct cypher_astnode_vt cypher_rel_index_query_astnode_vt;
extern const struct cypher_astnode_vt cypher_rel_id_lookup_astnode_vt;
extern const struct cypher_astnode_vt cypher_all_rels_scan_astnode_vt;
extern const struct cypher_astnode_vt cypher_match_astnode_vt;
extern const struct cypher_astnode_vt cypher_match_hint_astnode_vt;
extern const struct cypher_astnode_vt cypher_using_index_astnode_vt;
extern const struct cypher_astnode_vt cypher_using_join_astnode_vt;
extern const struct cypher_astnode_vt cypher_using_scan_astnode_vt;
extern const struct cypher_astnode_vt cypher_merge_astnode_vt;
extern const struct cypher_astnode_vt cypher_merge_action_astnode_vt;
extern const struct cypher_astnode_vt cypher_on_match_astnode_vt;
extern const struct cypher_astnode_vt cypher_on_create_astnode_vt;
extern const struct cypher_astnode_vt cypher_create_astnode_vt;
extern const struct cypher_astnode_vt cypher_set_astnode_vt;
extern const struct cypher_astnode_vt cypher_set_item_astnode_vt;
extern const struct cypher_astnode_vt cypher_set_property_astnode_vt;
extern const struct cypher_astnode_vt cypher_set_all_properties_astnode_vt;
extern const struct cypher_astnode_vt cypher_merge_properties_astnode_vt;
extern const struct cypher_astnode_vt cypher_set_labels_astnode_vt;
extern const struct cypher_astnode_vt cypher_delete_astnode_vt;
extern const struct cypher_astnode_vt cypher_remove_astnode_vt;
extern const struct cypher_astnode_vt cypher_remove_item_astnode_vt;
extern const struct cypher_astnode_vt cypher_remove_labels_astnode_vt;
extern const struct cypher_astnode_vt cypher_remove_property_astnode_vt;
extern const struct cypher_astnode_vt cypher_foreach_astnode_vt;
extern const struct cypher_astnode_vt cypher_with_astnode_vt;
extern const struct cypher_astnode_vt cypher_unwind_astnode_vt;
extern const struct cypher_astnode_vt cypher_call_astnode_vt;
extern const struct cypher_astnode_vt cypher_call_subquery_astnode_vt;
extern const struct cypher_astnode_vt cypher_return_astnode_vt;
extern const struct cypher_astnode_vt cypher_projection_astnode_vt;
extern const struct cypher_astnode_vt cypher_order_by_astnode_vt;
extern const struct cypher_astnode_vt cypher_sort_item_astnode_vt;
extern const struct cypher_astnode_vt cypher_union_astnode_vt;
extern const struct cypher_astnode_vt cypher_expression_astnode_vt;
extern const struct cypher_astnode_vt cypher_unary_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_binary_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_comparison_astnode_vt;
extern const struct cypher_astnode_vt cypher_apply_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_apply_all_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_property_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_subscript_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_slice_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_labels_operator_astnode_vt;
extern const struct cypher_astnode_vt cypher_list_comprehension_astnode_vt;
extern const struct cypher_astnode_vt cypher_pattern_comprehension_astnode_vt;
extern const struct cypher_astnode_vt cypher_case_astnode_vt;
extern const struct cypher_astnode_vt cypher_filter_astnode_vt;
extern const struct cypher_astnode_vt cypher_extract_astnode_vt;
extern const struct cypher_astnode_vt cypher_reduce_astnode_vt;
extern const struct cypher_astnode_vt cypher_all_astnode_vt;
extern const struct cypher_astnode_vt cypher_any_astnode_vt;
extern const struct cypher_astnode_vt cypher_single_astnode_vt;
extern const struct cypher_astnode_vt cypher_none_astnode_vt;
extern const struct cypher_astnode_vt cypher_collection_astnode_vt;
extern const struct cypher_astnode_vt cypher_map_astnode_vt;
extern const struct cypher_astnode_vt cypher_identifier_astnode_vt;
extern const struct cypher_astnode_vt cypher_parameter_astnode_vt;
extern const struct cypher_astnode_vt cypher_string_astnode_vt;
extern const struct cypher_astnode_vt cypher_integer_astnode_vt;
extern const struct cypher_astnode_vt cypher_float_astnode_vt;
extern const struct cypher_astnode_vt cypher_boolean_astnode_vt;
extern const struct cypher_astnode_vt cypher_true_astnode_vt;
extern const struct cypher_astnode_vt cypher_false_astnode_vt;
extern const struct cypher_astnode_vt cypher_null_astnode_vt;
extern const struct cypher_astnode_vt cypher_label_astnode_vt;
extern const struct cypher_astnode_vt cypher_reltype_astnode_vt;
extern const struct cypher_astnode_vt cypher_prop_name_astnode_vt;
extern const struct cypher_astnode_vt cypher_function_name_astnode_vt;
extern const struct cypher_astnode_vt cypher_index_name_astnode_vt;
extern const struct cypher_astnode_vt cypher_proc_name_astnode_vt;
extern const struct cypher_astnode_vt cypher_pattern_astnode_vt;
extern const struct cypher_astnode_vt cypher_pattern_path_astnode_vt;
extern const struct cypher_astnode_vt cypher_named_path_astnode_vt;
extern const struct cypher_astnode_vt cypher_shortest_path_astnode_vt;
extern const struct cypher_astnode_vt cypher_node_pattern_astnode_vt;
extern const struct cypher_astnode_vt cypher_rel_pattern_astnode_vt;
extern const struct cypher_astnode_vt cypher_range_astnode_vt;
extern const struct cypher_astnode_vt cypher_command_astnode_vt;
extern const struct cypher_astnode_vt cypher_comment_astnode_vt;
extern const struct cypher_astnode_vt cypher_line_comment_astnode_vt;
extern const struct cypher_astnode_vt cypher_block_comment_astnode_vt;
extern const struct cypher_astnode_vt cypher_error_astnode_vt;
extern const struct cypher_astnode_vt cypher_map_projection_astnode_vt;
extern const struct cypher_astnode_vt cypher_map_projection_selector_astnode_vt;
extern const struct cypher_astnode_vt cypher_map_projection_literal_astnode_vt;
extern const struct cypher_astnode_vt cypher_map_projection_property_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_map_projection_identifier_astnode_vt;
extern const struct cypher_astnode_vt
        cypher_map_projection_all_properties_astnode_vt;


typedef struct cypher_list_comprehension_astnode
        cypher_list_comprehension_astnode_t;

struct cypher_list_comprehension_astnode_vt
{
    const struct cypher_list_comprehension_astnode_vt **parents;
    unsigned int nparents;
    const cypher_astnode_t *(*get_identifier)(
            const cypher_list_comprehension_astnode_t *self);
    const cypher_astnode_t *(*get_expression)(
            const cypher_list_comprehension_astnode_t *self);
    const cypher_astnode_t *(*get_predicate)(
            const cypher_list_comprehension_astnode_t *self);
    const cypher_astnode_t *(*get_eval)(
            const cypher_list_comprehension_astnode_t *self);
};

struct cypher_list_comprehension_astnode
{
    struct cypher_astnode _astnode;
    const struct cypher_list_comprehension_astnode_vt *_vt;
};

int cypher_list_comprehension_astnode_init(
        cypher_list_comprehension_astnode_t *node,
        cypher_astnode_type_t type,
        const struct cypher_list_comprehension_astnode_vt *vt,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range);


typedef struct cypher_pattern_path_astnode cypher_pattern_path_astnode_t;

struct cypher_pattern_path_astnode_vt
{
    const struct cypher_pattern_path_astnode_vt **parents;
    unsigned int nparents;
    unsigned int (*nelements)(const cypher_pattern_path_astnode_t *self);
    const cypher_astnode_t *(*get_element)(
            const cypher_pattern_path_astnode_t *self, unsigned int index);
};

struct cypher_pattern_path_astnode
{
    struct cypher_astnode _astnode;
    const struct cypher_pattern_path_astnode_vt *_vt;
};


int cypher_pattern_path_astnode_init(cypher_pattern_path_astnode_t *node,
        cypher_astnode_type_t type,
        const struct cypher_pattern_path_astnode_vt *vt,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range);


#endif/*CYPHER_PARSER_ASTNODE_H*/
