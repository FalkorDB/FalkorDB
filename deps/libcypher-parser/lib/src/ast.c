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
#include "../../config.h"
#include "ast.h"
#include "astnode.h"
#include "util.h"
#include <assert.h>
#include <math.h>


struct cypher_astnode_vts
{
    const struct cypher_astnode_vt *statement;
    const struct cypher_astnode_vt *statement_option;
    const struct cypher_astnode_vt *cypher_option;
    const struct cypher_astnode_vt *cypher_option_param;
    const struct cypher_astnode_vt *explain_option;
    const struct cypher_astnode_vt *profile_option;
    const struct cypher_astnode_vt *schema_command;
    const struct cypher_astnode_vt *create_node_props_index;
    const struct cypher_astnode_vt *create_pattern_props_index;
    const struct cypher_astnode_vt *drop_props_index;
    const struct cypher_astnode_vt *create_node_prop_constraint;
    const struct cypher_astnode_vt *drop_node_prop_constraint;
    const struct cypher_astnode_vt *create_rel_prop_constraint;
    const struct cypher_astnode_vt *drop_rel_prop_constraint;
    const struct cypher_astnode_vt *query;
    const struct cypher_astnode_vt *query_option;
    const struct cypher_astnode_vt *using_periodic_commit;
    const struct cypher_astnode_vt *query_clause;
    const struct cypher_astnode_vt *load_csv;
    const struct cypher_astnode_vt *start;
    const struct cypher_astnode_vt *start_point;
    const struct cypher_astnode_vt *node_index_lookup;
    const struct cypher_astnode_vt *node_index_query;
    const struct cypher_astnode_vt *node_id_lookup;
    const struct cypher_astnode_vt *all_nodes_scan;
    const struct cypher_astnode_vt *rel_index_lookup;
    const struct cypher_astnode_vt *rel_index_query;
    const struct cypher_astnode_vt *rel_id_lookup;
    const struct cypher_astnode_vt *all_rels_scan;
    const struct cypher_astnode_vt *match;
    const struct cypher_astnode_vt *match_hint;
    const struct cypher_astnode_vt *using_index;
    const struct cypher_astnode_vt *using_join;
    const struct cypher_astnode_vt *using_scan;
    const struct cypher_astnode_vt *merge;
    const struct cypher_astnode_vt *merge_action;
    const struct cypher_astnode_vt *on_match;
    const struct cypher_astnode_vt *on_create;
    const struct cypher_astnode_vt *create;
    const struct cypher_astnode_vt *set;
    const struct cypher_astnode_vt *set_item;
    const struct cypher_astnode_vt *set_property;
    const struct cypher_astnode_vt *set_all_properties;
    const struct cypher_astnode_vt *merge_properties;
    const struct cypher_astnode_vt *set_labels;
    const struct cypher_astnode_vt *delete_clause;
    const struct cypher_astnode_vt *remove;
    const struct cypher_astnode_vt *remove_item;
    const struct cypher_astnode_vt *remove_labels;
    const struct cypher_astnode_vt *remove_property;
    const struct cypher_astnode_vt *foreach;
    const struct cypher_astnode_vt *with;
    const struct cypher_astnode_vt *unwind;
    const struct cypher_astnode_vt *call;
    const struct cypher_astnode_vt *call_subquery;
    const struct cypher_astnode_vt *return_clause;
    const struct cypher_astnode_vt *projection;
    const struct cypher_astnode_vt *order_by;
    const struct cypher_astnode_vt *sort_item;
    const struct cypher_astnode_vt *union_clause;
    const struct cypher_astnode_vt *expression;
    const struct cypher_astnode_vt *unary_operator;
    const struct cypher_astnode_vt *binary_operator;
    const struct cypher_astnode_vt *comparison;
    const struct cypher_astnode_vt *apply_operator;
    const struct cypher_astnode_vt *apply_all_operator;
    const struct cypher_astnode_vt *property_operator;
    const struct cypher_astnode_vt *subscript_operator;
    const struct cypher_astnode_vt *slice_operator;
    const struct cypher_astnode_vt *labels_operator;
    const struct cypher_astnode_vt *list_comprehension;
    const struct cypher_astnode_vt *pattern_comprehension;
    const struct cypher_astnode_vt *case_expression;
    const struct cypher_astnode_vt *filter;
    const struct cypher_astnode_vt *extract;
    const struct cypher_astnode_vt *reduce;
    const struct cypher_astnode_vt *all;
    const struct cypher_astnode_vt *any;
    const struct cypher_astnode_vt *single;
    const struct cypher_astnode_vt *none;
    const struct cypher_astnode_vt *collection;
    const struct cypher_astnode_vt *map;
    const struct cypher_astnode_vt *identifier;
    const struct cypher_astnode_vt *parameter;
    const struct cypher_astnode_vt *string;
    const struct cypher_astnode_vt *integer;
    const struct cypher_astnode_vt *float_literal;
    const struct cypher_astnode_vt *boolean_literal;
    const struct cypher_astnode_vt *true_literal;
    const struct cypher_astnode_vt *false_literal;
    const struct cypher_astnode_vt *null_literal;
    const struct cypher_astnode_vt *label;
    const struct cypher_astnode_vt *reltype;
    const struct cypher_astnode_vt *prop_name;
    const struct cypher_astnode_vt *function_name;
    const struct cypher_astnode_vt *index_name;
    const struct cypher_astnode_vt *proc_name;
    const struct cypher_astnode_vt *pattern;
    const struct cypher_astnode_vt *named_path;
    const struct cypher_astnode_vt *shortest_path;
    const struct cypher_astnode_vt *pattern_path;
    const struct cypher_astnode_vt *node_pattern;
    const struct cypher_astnode_vt *rel_pattern;
    const struct cypher_astnode_vt *range;
    const struct cypher_astnode_vt *command;
    const struct cypher_astnode_vt *comment;
    const struct cypher_astnode_vt *line_comment;
    const struct cypher_astnode_vt *block_comment;
    const struct cypher_astnode_vt *error;
    const struct cypher_astnode_vt *map_projection;
    const struct cypher_astnode_vt *map_projection_selector;
    const struct cypher_astnode_vt *map_projection_literal;
    const struct cypher_astnode_vt *map_projection_property;
    const struct cypher_astnode_vt *map_projection_identifier;
    const struct cypher_astnode_vt *map_projection_all_properties;
};
static const struct cypher_astnode_vts cypher_astnode_vts =
{
    .statement = &cypher_statement_astnode_vt,
    .statement_option = &cypher_statement_option_astnode_vt,
    .cypher_option = &cypher_cypher_option_astnode_vt,
    .cypher_option_param = &cypher_cypher_option_param_astnode_vt,
    .explain_option = &cypher_explain_option_astnode_vt,
    .profile_option = &cypher_profile_option_astnode_vt,
    .schema_command = &cypher_schema_command_astnode_vt,
    .create_node_props_index = &cypher_create_node_props_index_astnode_vt,
    .create_pattern_props_index = &cypher_create_pattern_props_index_astnode_vt,
    .drop_props_index = &cypher_drop_props_index_astnode_vt,
    .create_node_prop_constraint =
           &cypher_create_node_prop_constraint_astnode_vt,
    .drop_node_prop_constraint = &cypher_drop_node_prop_constraint_astnode_vt,
    .create_rel_prop_constraint = &cypher_create_rel_prop_constraint_astnode_vt,
    .drop_rel_prop_constraint = &cypher_drop_rel_prop_constraint_astnode_vt,
    .query = &cypher_query_astnode_vt,
    .query_option = &cypher_query_option_astnode_vt,
    .using_periodic_commit = &cypher_using_periodic_commit_astnode_vt,
    .query_clause = &cypher_query_clause_astnode_vt,
    .load_csv = &cypher_load_csv_astnode_vt,
    .start = &cypher_start_astnode_vt,
    .start_point = &cypher_start_point_astnode_vt,
    .node_index_lookup = &cypher_node_index_lookup_astnode_vt,
    .node_index_query = &cypher_node_index_query_astnode_vt,
    .node_id_lookup = &cypher_node_id_lookup_astnode_vt,
    .all_nodes_scan = &cypher_all_nodes_scan_astnode_vt,
    .rel_index_lookup = &cypher_rel_index_lookup_astnode_vt,
    .rel_index_query = &cypher_rel_index_query_astnode_vt,
    .rel_id_lookup = &cypher_rel_id_lookup_astnode_vt,
    .all_rels_scan = &cypher_all_rels_scan_astnode_vt,
    .match = &cypher_match_astnode_vt,
    .match_hint = &cypher_match_hint_astnode_vt,
    .using_index = &cypher_using_index_astnode_vt,
    .using_join = &cypher_using_join_astnode_vt,
    .using_scan = &cypher_using_scan_astnode_vt,
    .merge = &cypher_merge_astnode_vt,
    .merge_action = &cypher_merge_action_astnode_vt,
    .on_match = &cypher_on_match_astnode_vt,
    .on_create = &cypher_on_create_astnode_vt,
    .create = &cypher_create_astnode_vt,
    .set = &cypher_set_astnode_vt,
    .set_item = &cypher_set_item_astnode_vt,
    .set_property = &cypher_set_property_astnode_vt,
    .set_all_properties = &cypher_set_all_properties_astnode_vt,
    .merge_properties = &cypher_merge_properties_astnode_vt,
    .set_labels = &cypher_set_labels_astnode_vt,
    .delete_clause = &cypher_delete_astnode_vt,
    .remove = &cypher_remove_astnode_vt,
    .remove_item = &cypher_remove_item_astnode_vt,
    .remove_labels = &cypher_remove_labels_astnode_vt,
    .remove_property = &cypher_remove_property_astnode_vt,
    .foreach = &cypher_foreach_astnode_vt,
    .with = &cypher_with_astnode_vt,
    .unwind = &cypher_unwind_astnode_vt,
    .call = &cypher_call_astnode_vt,
    .call_subquery = &cypher_call_subquery_astnode_vt,
    .return_clause = &cypher_return_astnode_vt,
    .projection = &cypher_projection_astnode_vt,
    .order_by = &cypher_order_by_astnode_vt,
    .sort_item = &cypher_sort_item_astnode_vt,
    .union_clause = &cypher_union_astnode_vt,
    .expression = &cypher_expression_astnode_vt,
    .unary_operator = &cypher_unary_operator_astnode_vt,
    .binary_operator = &cypher_binary_operator_astnode_vt,
    .comparison = &cypher_comparison_astnode_vt,
    .apply_operator = &cypher_apply_operator_astnode_vt,
    .apply_all_operator = &cypher_apply_all_operator_astnode_vt,
    .property_operator = &cypher_property_operator_astnode_vt,
    .subscript_operator = &cypher_subscript_operator_astnode_vt,
    .slice_operator = &cypher_slice_operator_astnode_vt,
    .labels_operator = &cypher_labels_operator_astnode_vt,
    .list_comprehension = &cypher_list_comprehension_astnode_vt,
    .pattern_comprehension = &cypher_pattern_comprehension_astnode_vt,
    .case_expression = &cypher_case_astnode_vt,
    .filter = &cypher_filter_astnode_vt,
    .extract = &cypher_extract_astnode_vt,
    .reduce = &cypher_reduce_astnode_vt,
    .any = &cypher_any_astnode_vt,
    .all = &cypher_all_astnode_vt,
    .single = &cypher_single_astnode_vt,
    .none = &cypher_none_astnode_vt,
    .collection = &cypher_collection_astnode_vt,
    .map = &cypher_map_astnode_vt,
    .identifier = &cypher_identifier_astnode_vt,
    .parameter = &cypher_parameter_astnode_vt,
    .string = &cypher_string_astnode_vt,
    .integer = &cypher_integer_astnode_vt,
    .float_literal = &cypher_float_astnode_vt,
    .boolean_literal = &cypher_boolean_astnode_vt,
    .true_literal = &cypher_true_astnode_vt,
    .false_literal = &cypher_false_astnode_vt,
    .null_literal = &cypher_null_astnode_vt,
    .label = &cypher_label_astnode_vt,
    .reltype = &cypher_reltype_astnode_vt,
    .prop_name = &cypher_prop_name_astnode_vt,
    .function_name = &cypher_function_name_astnode_vt,
    .index_name = &cypher_index_name_astnode_vt,
    .proc_name = &cypher_proc_name_astnode_vt,
    .pattern = &cypher_pattern_astnode_vt,
    .named_path = &cypher_named_path_astnode_vt,
    .shortest_path = &cypher_shortest_path_astnode_vt,
    .pattern_path = &cypher_pattern_path_astnode_vt,
    .node_pattern = &cypher_node_pattern_astnode_vt,
    .rel_pattern = &cypher_rel_pattern_astnode_vt,
    .range = &cypher_range_astnode_vt,
    .command = &cypher_command_astnode_vt,
    .line_comment = &cypher_line_comment_astnode_vt,
    .block_comment = &cypher_block_comment_astnode_vt,
    .error = &cypher_error_astnode_vt,
    .map_projection = &cypher_map_projection_astnode_vt,
    .map_projection_selector = &cypher_map_projection_selector_astnode_vt,
    .map_projection_literal = &cypher_map_projection_literal_astnode_vt,
    .map_projection_property = &cypher_map_projection_property_astnode_vt,
    .map_projection_identifier = &cypher_map_projection_identifier_astnode_vt,
    .map_projection_all_properties =
            &cypher_map_projection_all_properties_astnode_vt
};

#define VT_OFFSET(name) offsetof(struct cypher_astnode_vts, name) \
    / sizeof(struct cypher_astnode_vt *)
#define VT_PTR(offset) *((struct cypher_astnode_vt * const *)(const void *)( \
            (const char *)&cypher_astnode_vts + \
            offset * sizeof(struct cypher_astnode_vt *)))

const uint8_t CYPHER_AST_STATEMENT = VT_OFFSET(statement);
const uint8_t CYPHER_AST_STATEMENT_OPTION = VT_OFFSET(statement_option);
const uint8_t CYPHER_AST_CYPHER_OPTION = VT_OFFSET(cypher_option);
const uint8_t CYPHER_AST_CYPHER_OPTION_PARAM = VT_OFFSET(cypher_option_param);
const uint8_t CYPHER_AST_EXPLAIN_OPTION = VT_OFFSET(explain_option);
const uint8_t CYPHER_AST_PROFILE_OPTION = VT_OFFSET(profile_option);
const uint8_t CYPHER_AST_SCHEMA_COMMAND = VT_OFFSET(schema_command);
const uint8_t CYPHER_AST_CREATE_NODE_PROPS_INDEX =
        VT_OFFSET(create_node_props_index);
const uint8_t CYPHER_AST_CREATE_PATTERN_PROPS_INDEX =
        VT_OFFSET(create_pattern_props_index);
const uint8_t CYPHER_AST_DROP_PROPS_INDEX =
        VT_OFFSET(drop_props_index);
const uint8_t CYPHER_AST_CREATE_NODE_PROP_CONSTRAINT =
        VT_OFFSET(create_node_prop_constraint);
const uint8_t CYPHER_AST_DROP_NODE_PROP_CONSTRAINT =
        VT_OFFSET(drop_node_prop_constraint);
const uint8_t CYPHER_AST_CREATE_REL_PROP_CONSTRAINT =
        VT_OFFSET(create_rel_prop_constraint);
const uint8_t CYPHER_AST_DROP_REL_PROP_CONSTRAINT =
        VT_OFFSET(drop_rel_prop_constraint);
const uint8_t CYPHER_AST_QUERY = VT_OFFSET(query);
const uint8_t CYPHER_AST_QUERY_OPTION = VT_OFFSET(query_option);
const uint8_t CYPHER_AST_USING_PERIODIC_COMMIT = VT_OFFSET(using_periodic_commit);
const uint8_t CYPHER_AST_QUERY_CLAUSE = VT_OFFSET(query_clause);
const uint8_t CYPHER_AST_LOAD_CSV = VT_OFFSET(load_csv);
const uint8_t CYPHER_AST_START = VT_OFFSET(start);
const uint8_t CYPHER_AST_START_POINT = VT_OFFSET(start_point);
const uint8_t CYPHER_AST_NODE_INDEX_LOOKUP = VT_OFFSET(node_index_lookup);
const uint8_t CYPHER_AST_NODE_INDEX_QUERY = VT_OFFSET(node_index_query);
const uint8_t CYPHER_AST_NODE_ID_LOOKUP = VT_OFFSET(node_id_lookup);
const uint8_t CYPHER_AST_ALL_NODES_SCAN = VT_OFFSET(all_nodes_scan);
const uint8_t CYPHER_AST_REL_INDEX_LOOKUP = VT_OFFSET(rel_index_lookup);
const uint8_t CYPHER_AST_REL_INDEX_QUERY = VT_OFFSET(rel_index_query);
const uint8_t CYPHER_AST_REL_ID_LOOKUP = VT_OFFSET(rel_id_lookup);
const uint8_t CYPHER_AST_ALL_RELS_SCAN = VT_OFFSET(all_rels_scan);
const uint8_t CYPHER_AST_MATCH = VT_OFFSET(match);
const uint8_t CYPHER_AST_MATCH_HINT = VT_OFFSET(match_hint);
const uint8_t CYPHER_AST_USING_INDEX = VT_OFFSET(using_index);
const uint8_t CYPHER_AST_USING_JOIN = VT_OFFSET(using_join);
const uint8_t CYPHER_AST_USING_SCAN = VT_OFFSET(using_scan);
const uint8_t CYPHER_AST_MERGE = VT_OFFSET(merge);
const uint8_t CYPHER_AST_MERGE_ACTION = VT_OFFSET(merge_action);
const uint8_t CYPHER_AST_ON_MATCH = VT_OFFSET(on_match);
const uint8_t CYPHER_AST_ON_CREATE = VT_OFFSET(on_create);
const uint8_t CYPHER_AST_CREATE = VT_OFFSET(create);
const uint8_t CYPHER_AST_SET = VT_OFFSET(set);
const uint8_t CYPHER_AST_SET_ITEM = VT_OFFSET(set_item);
const uint8_t CYPHER_AST_SET_PROPERTY = VT_OFFSET(set_property);
const uint8_t CYPHER_AST_SET_ALL_PROPERTIES = VT_OFFSET(set_all_properties);
const uint8_t CYPHER_AST_MERGE_PROPERTIES = VT_OFFSET(merge_properties);
const uint8_t CYPHER_AST_SET_LABELS = VT_OFFSET(set_labels);
const uint8_t CYPHER_AST_DELETE = VT_OFFSET(delete_clause);
const uint8_t CYPHER_AST_REMOVE = VT_OFFSET(remove);
const uint8_t CYPHER_AST_REMOVE_ITEM = VT_OFFSET(remove_item);
const uint8_t CYPHER_AST_REMOVE_LABELS = VT_OFFSET(remove_labels);
const uint8_t CYPHER_AST_REMOVE_PROPERTY = VT_OFFSET(remove_property);
const uint8_t CYPHER_AST_FOREACH = VT_OFFSET(foreach);
const uint8_t CYPHER_AST_WITH = VT_OFFSET(with);
const uint8_t CYPHER_AST_UNWIND = VT_OFFSET(unwind);
const uint8_t CYPHER_AST_CALL = VT_OFFSET(call);
const uint8_t CYPHER_AST_CALL_SUBQUERY = VT_OFFSET(call_subquery);
const uint8_t CYPHER_AST_RETURN = VT_OFFSET(return_clause);
const uint8_t CYPHER_AST_PROJECTION = VT_OFFSET(projection);
const uint8_t CYPHER_AST_ORDER_BY = VT_OFFSET(order_by);
const uint8_t CYPHER_AST_SORT_ITEM = VT_OFFSET(sort_item);
const uint8_t CYPHER_AST_UNION = VT_OFFSET(union_clause);
const uint8_t CYPHER_AST_EXPRESSION = VT_OFFSET(expression);
const uint8_t CYPHER_AST_UNARY_OPERATOR = VT_OFFSET(unary_operator);
const uint8_t CYPHER_AST_BINARY_OPERATOR = VT_OFFSET(binary_operator);
const uint8_t CYPHER_AST_COMPARISON = VT_OFFSET(comparison);
const uint8_t CYPHER_AST_APPLY_OPERATOR = VT_OFFSET(apply_operator);
const uint8_t CYPHER_AST_APPLY_ALL_OPERATOR = VT_OFFSET(apply_all_operator);
const uint8_t CYPHER_AST_PROPERTY_OPERATOR = VT_OFFSET(property_operator);
const uint8_t CYPHER_AST_SUBSCRIPT_OPERATOR = VT_OFFSET(subscript_operator);
const uint8_t CYPHER_AST_SLICE_OPERATOR = VT_OFFSET(slice_operator);
const uint8_t CYPHER_AST_LABELS_OPERATOR = VT_OFFSET(labels_operator);
const uint8_t CYPHER_AST_LIST_COMPREHENSION = VT_OFFSET(list_comprehension);
const uint8_t CYPHER_AST_PATTERN_COMPREHENSION = VT_OFFSET(pattern_comprehension);
const uint8_t CYPHER_AST_CASE = VT_OFFSET(case_expression);
const uint8_t CYPHER_AST_FILTER = VT_OFFSET(filter);
const uint8_t CYPHER_AST_EXTRACT = VT_OFFSET(extract);
const uint8_t CYPHER_AST_REDUCE = VT_OFFSET(reduce);
const uint8_t CYPHER_AST_ALL = VT_OFFSET(all);
const uint8_t CYPHER_AST_ANY = VT_OFFSET(any);
const uint8_t CYPHER_AST_SINGLE = VT_OFFSET(single);
const uint8_t CYPHER_AST_NONE = VT_OFFSET(none);
const uint8_t CYPHER_AST_COLLECTION = VT_OFFSET(collection);
const uint8_t CYPHER_AST_MAP = VT_OFFSET(map);
const uint8_t CYPHER_AST_IDENTIFIER = VT_OFFSET(identifier);
const uint8_t CYPHER_AST_PARAMETER = VT_OFFSET(parameter);
const uint8_t CYPHER_AST_STRING = VT_OFFSET(string);
const uint8_t CYPHER_AST_INTEGER = VT_OFFSET(integer);
const uint8_t CYPHER_AST_FLOAT = VT_OFFSET(float_literal);
const uint8_t CYPHER_AST_BOOLEAN = VT_OFFSET(boolean_literal);
const uint8_t CYPHER_AST_TRUE = VT_OFFSET(true_literal);
const uint8_t CYPHER_AST_FALSE = VT_OFFSET(false_literal);
const uint8_t CYPHER_AST_NULL = VT_OFFSET(null_literal);
const uint8_t CYPHER_AST_LABEL = VT_OFFSET(label);
const uint8_t CYPHER_AST_RELTYPE = VT_OFFSET(reltype);
const uint8_t CYPHER_AST_PROP_NAME = VT_OFFSET(prop_name);
const uint8_t CYPHER_AST_FUNCTION_NAME = VT_OFFSET(function_name);
const uint8_t CYPHER_AST_INDEX_NAME = VT_OFFSET(index_name);
const uint8_t CYPHER_AST_PROC_NAME = VT_OFFSET(proc_name);
const uint8_t CYPHER_AST_PATTERN = VT_OFFSET(pattern);
const uint8_t CYPHER_AST_NAMED_PATH = VT_OFFSET(named_path);
const uint8_t CYPHER_AST_SHORTEST_PATH = VT_OFFSET(shortest_path);
const uint8_t CYPHER_AST_PATTERN_PATH = VT_OFFSET(pattern_path);
const uint8_t CYPHER_AST_NODE_PATTERN = VT_OFFSET(node_pattern);
const uint8_t CYPHER_AST_REL_PATTERN = VT_OFFSET(rel_pattern);
const uint8_t CYPHER_AST_RANGE = VT_OFFSET(range);
const uint8_t CYPHER_AST_COMMAND = VT_OFFSET(command);
const uint8_t CYPHER_AST_COMMENT = VT_OFFSET(comment);
const uint8_t CYPHER_AST_LINE_COMMENT = VT_OFFSET(line_comment);
const uint8_t CYPHER_AST_BLOCK_COMMENT = VT_OFFSET(block_comment);
const uint8_t CYPHER_AST_ERROR = VT_OFFSET(error);
const uint8_t CYPHER_AST_MAP_PROJECTION = VT_OFFSET(map_projection);
const uint8_t CYPHER_AST_MAP_PROJECTION_SELECTOR = VT_OFFSET(map_projection_selector);
const uint8_t CYPHER_AST_MAP_PROJECTION_LITERAL =
        VT_OFFSET(map_projection_literal);
const uint8_t CYPHER_AST_MAP_PROJECTION_PROPERTY =
        VT_OFFSET(map_projection_property);
const uint8_t CYPHER_AST_MAP_PROJECTION_IDENTIFIER =
        VT_OFFSET(map_projection_identifier);
const uint8_t CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES =
        VT_OFFSET(map_projection_all_properties);
static const uint8_t _MAX_VT_OFF =
    (sizeof(struct cypher_astnode_vts) / sizeof(struct cypher_astnode_vt *));

static_assert(
    (sizeof(struct cypher_astnode_vts) / sizeof(struct cypher_astnode_vt *)) <= UINT8_MAX,
    "cannot have more than 2^8 AST node types");


cypher_astnode_type_t cypher_astnode_type(const cypher_astnode_t *node)
{
    REQUIRE(node != NULL, _MAX_VT_OFF);
    return node->type;
}


static bool cypher_astnode_vt_instanceof(const struct cypher_astnode_vt *vt,
        const struct cypher_astnode_vt *target)
{
    if (vt == target)
    {
        return true;
    }
    for (unsigned int i = 0; i < vt->nparents; ++i)
    {
        if (cypher_astnode_vt_instanceof(vt->parents[i], target))
        {
            return true;
        }
    }
    return false;
}


bool cypher_astnode_instanceof(const cypher_astnode_t *node,
        cypher_astnode_type_t type)
{
    if (node == NULL)
    {
        return false;
    }
    REQUIRE(node->type < _MAX_VT_OFF, false);
    REQUIRE(type < _MAX_VT_OFF, false);
    const struct cypher_astnode_vt *vt = VT_PTR(node->type);
    const struct cypher_astnode_vt *target = VT_PTR(type);
    return cypher_astnode_vt_instanceof(vt, target);
}


const char *cypher_astnode_typestr(cypher_astnode_type_t type)
{
    REQUIRE(type < _MAX_VT_OFF, NULL);
    const struct cypher_astnode_vt *vt = VT_PTR(type);
    return vt->name;
}


struct cypher_input_range cypher_astnode_range(const cypher_astnode_t *node)
{
    return node->range;
}


void cypher_ast_vfree(cypher_astnode_t * const *ast, unsigned int n)
{
    for (; n > 0; ++ast, --n)
    {
        cypher_ast_free(*ast);
    }
}


cypher_astnode_t **cypher_ast_vclone(cypher_astnode_t * const *ast,
        unsigned int n)
{
    if (n == 0)
    {
        return NULL;
    }

    cypher_astnode_t **clones = calloc(n, sizeof(cypher_astnode_t *));
    if (clones == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < n; ++i)
    {
        clones[i] = cypher_ast_clone(ast[i]);
        if (clones[i] == NULL)
        {
            goto failure;
        }
    }
    return clones;

    int errsv;
failure:
    errsv = errno;
    cypher_ast_vfree(clones, n);
    free(clones);
    errno = errsv;
    return NULL;
}


void cypher_ast_free(cypher_astnode_t *ast)
{
    if (ast == NULL)
    {
        return;
    }

    while (ast->annotations != NULL)
    {
        cp_release_annotation(ast->annotations);
    }

    cypher_astnode_t **children = ast->children;
    unsigned int nchildren = ast->nchildren;

    assert(ast->type < _MAX_VT_OFF);
    const struct cypher_astnode_vt *vt = VT_PTR(ast->type);
    vt->release(ast);

    cypher_ast_vfree(children, nchildren);
    free(children);
}


void cypher_astnode_free(cypher_astnode_t *ast)
{
    if (ast == NULL)
    {
        return;
    }

    while (ast->annotations != NULL)
    {
        cp_release_annotation(ast->annotations);
    }

    cypher_astnode_t **children = ast->children;

    assert(ast->type < _MAX_VT_OFF);
    const struct cypher_astnode_vt *vt = VT_PTR(ast->type);
    vt->release(ast);

    free(children);
}


cypher_astnode_t *cypher_ast_clone(const cypher_astnode_t *ast)
{
    if (ast == NULL)
    {
        return NULL;
    }

    cypher_astnode_t **children = NULL;
    if (ast->nchildren != 0)
    {
        children = cypher_ast_vclone(ast->children, ast->nchildren);
        if (children == NULL)
        {
            return NULL;
        }
    }

    assert(ast->type < _MAX_VT_OFF);
    const struct cypher_astnode_vt *vt = VT_PTR(ast->type);
    cypher_astnode_t *clone = vt->clone(ast, children);
    if (clone == NULL)
    {
        goto failure;
    }
    free(children);
    return clone;

    int errsv;
failure:
    errsv = errno;
    cypher_ast_vfree(children, ast->nchildren);
    free(children);
    errno = errsv;
    return NULL;
}


unsigned int cypher_ast_depth(const cypher_astnode_t *ast)
{
    unsigned int depth = 0;
    for (unsigned int i = 0; i < ast->nchildren; ++i)
    {
        depth = maxu(depth, cypher_ast_depth(ast->children[i]));
    }
    return 1 + depth;
}


unsigned int cypher_ast_set_ordinals(cypher_astnode_t *ast, unsigned int n)
{
    if (ast == NULL)
    {
        return n;
    }
    ast->ordinal = n++;
    for (unsigned int i = 0; i < ast->nchildren; ++i)
    {
        n = cypher_ast_set_ordinals(ast->children[i], n);
    }
    return n;
}


static ssize_t cypher_astnode_detailstr_realloc(const cypher_astnode_t *node,
        char **buf, size_t *bufcap)
{
    ssize_t width = cypher_astnode_detailstr(node, *buf, *bufcap);
    if (width < 0)
    {
        return -1;
    }
    if ((size_t)width > *bufcap)
    {
        char *newbuf = realloc(*buf, (size_t)width + 1);
        if (newbuf == NULL)
        {
            return -1;
        }
        *buf = newbuf;
        *bufcap = (size_t)width + 1;
        width = cypher_astnode_detailstr(node, *buf, *bufcap);
        if (width < 0)
        {
            return -1;
        }
    }
    return width;
}


static void ast_fprint_field_widths(const cypher_astnode_t *ast,
        unsigned int *max_ordinal, size_t *max_start, size_t *max_end,
        unsigned int *name_width, unsigned int depth)
{
    assert(ast != NULL);

    *max_ordinal = maxu(*max_ordinal, ast->ordinal);

    *max_start = maxzu(*max_start, ast->range.start.offset);
    *max_end = maxzu(*max_end, ast->range.end.offset);

    const char *typestr = cypher_astnode_typestr(cypher_astnode_type(ast));
    *name_width = maxu(*name_width, strlen(typestr) + (depth * 2));

    for (unsigned int i = 0; i < ast->nchildren; ++i)
    {
        ast_fprint_field_widths(ast->children[i], max_ordinal,
                max_start, max_end, name_width, depth + 1);
    }
}


#define CYPHER_AST_FPRINT_MIN_DETAIL_WIDTH 10

static int ast_fprint_detail(FILE *stream, const cypher_astnode_t *node,
        const char *detail, size_t len, unsigned int render_width,
        unsigned int offset,
        const struct cypher_parser_colorization *colorization)
{
    assert(colorization != NULL);
    assert(len > 0);

    if (fputs(colorization->ast_desc[0], stream) == EOF)
    {
        return -1;
    }

    size_t width = (render_width == 0)? 0 :
        (offset < render_width)? render_width - offset
            : CYPHER_AST_FPRINT_MIN_DETAIL_WIDTH;

    for (size_t n = 0, w = width; n < (size_t)len; ++n, ++detail)
    {
        char desc[3] = { '\\', '\0', '\0' };
        switch (*detail)
        {
            case '\a': desc[1] = 'a'; break;
            case '\b': desc[1] = 'b'; break;
            case '\f': desc[1] = 'f'; break;
            case '\n': desc[1] = 'n'; break;
            case '\r': desc[1] = 'r'; break;
            case '\t': desc[1] = 't'; break;
            case '\v': desc[1] = 'v'; break;
            default:
                desc[0] = *detail;
        }

        for (unsigned int i = 0; i < 2 && desc[i] != '\0'; ++i)
        {
            if (fputc(desc[i], stream) == EOF)
            {
                return -1;
            }
            if (render_width > 0 && --w == 0 && (n+1) < (size_t)len)
            {
                if (fputc('\n', stream) == EOF)
                {
                    return -1;
                }
                for (unsigned int j = offset; j > 0; --j)
                {
                    if (fputc(' ', stream) == EOF)
                    {
                        return -1;
                    }
                }
                w = width;
            }
        }
    }

    if (fputc('\n', stream) == EOF ||
        fputs(colorization->ast_desc[1], stream) == EOF)
    {
        return -1;
    }

    return 0;
}


static int _cypher_ast_fprint(const cypher_astnode_t *ast, FILE *stream,
        const struct cypher_parser_colorization *colorization,
        char **buf, size_t *bufcap, unsigned int render_width,
        unsigned int ordinal_width, unsigned int start_width,
        unsigned int end_width, unsigned int name_width, unsigned int depth)
{
    if ((snprintf_realloc(buf, bufcap, "@%u", ast->ordinal) < 0) ||
        (fprintf(stream, "%s%*s%s  ", colorization->ast_ordinal[0],
                 ordinal_width, *buf, colorization->ast_ordinal[1]) < 0))
    {
        return -1;
    }

#ifdef WIN32
    const char* format = "%s%*Iu..%-*Iu%s  %s";
#else
    const char* format = "%s%*zu..%-*zu%s  %s";
#endif
    if (fprintf(stream, format,
                colorization->ast_range[0],
                start_width, ast->range.start.offset,
                end_width, ast->range.end.offset,
                colorization->ast_range[1], colorization->ast_indent[0]) < 0)
    {
        return -1;
    }

    for (unsigned int i = 0; i < depth; ++i)
    {
        if (fputs("> ", stream) == EOF)
        {
            return -1;
        }
    }

    const char *typestr = cypher_astnode_typestr(cypher_astnode_type(ast));
    if (fprintf(stream, "%s%s%s%s", colorization->ast_indent[1],
                colorization->ast_type[0], typestr,
                colorization->ast_type[1]) < 0)
    {
        return -1;
    }

    ssize_t len = cypher_astnode_detailstr_realloc(ast, buf, bufcap);
    if (len < 0)
    {
        return -1;
    }

    if (len > 0)
    {
        unsigned int consumed = depth * 2 + strlen(typestr);
        assert(consumed <= name_width);
        unsigned int pad = name_width - consumed + 2;
        if (fprintf(stream, "%*s", pad, "") < 0)
        {
            return -1;
        }

        unsigned int detail_offset = name_width + start_width + end_width +
            ordinal_width + 8;
        if (ast_fprint_detail(stream, ast, *buf, len, render_width,
                    detail_offset, colorization) < 0)
        {
            return -1;
        }
    }
    else
    {
        if (fputc('\n', stream) == EOF)
        {
            return -1;
        }
    }

    for (unsigned int i = 0; i < cypher_astnode_nchildren(ast); ++i)
    {
        const cypher_astnode_t *child = cypher_astnode_get_child(ast, i);
        if (_cypher_ast_fprint(child, stream, colorization, buf, bufcap,
                    render_width, ordinal_width, start_width, end_width,
                    name_width, depth+1) < 0)
        {
            return -1;
        }
    }
    return 0;
}


int cypher_ast_fprint(const cypher_astnode_t *ast, FILE *stream,
        unsigned int width,
        const struct cypher_parser_colorization *colorization,
        uint_fast32_t flags)
{
    REQUIRE(ast != NULL, -1);
    if (colorization == NULL)
    {
        colorization = cypher_parser_no_colorization;
    }

    unsigned int max_ordinal = 0, name_width = 0;
    size_t max_start = 0, max_end = 0;
    ast_fprint_field_widths(ast, &max_ordinal, &max_start, &max_end,
            &name_width, 0);
    unsigned int ordinal_width = (unsigned int)log10(max_ordinal)+2;
    unsigned int start_width = (unsigned int)log10(max_start)+1;
    unsigned int end_width = (unsigned int)log10(max_end)+1;

    size_t bufcap = 1024;
    char *buf = malloc(bufcap);
    if (buf == NULL)
    {
        return -1;
    }
    int r = _cypher_ast_fprint(ast, stream, colorization, &buf, &bufcap, width,
            ordinal_width, start_width, end_width, name_width, 0);
    free(buf);
    return r;
}


int cypher_ast_fprintv(cypher_astnode_t * const *asts, unsigned int n,
        FILE *stream, unsigned int width,
        const struct cypher_parser_colorization *colorization,
        uint_fast32_t flags)
{
    REQUIRE(n == 0 || asts != NULL, -1);
    if (colorization == NULL)
    {
        colorization = cypher_parser_no_colorization;
    }

    unsigned int max_ordinal = 0, name_width = 0;
    size_t max_start = 0, max_end = 0;
    for (unsigned int i = 0; i < n; ++i)
    {
        ast_fprint_field_widths(asts[i], &max_ordinal, &max_start, &max_end,
                &name_width, 0);
    }
    unsigned int ordinal_width = (unsigned int)log10(max_ordinal)+2;
    unsigned int start_width = (unsigned int)log10(max_start)+1;
    unsigned int end_width = (unsigned int)log10(max_end)+1;

    size_t bufcap = 1024;
    char *buf = malloc(bufcap);
    if (buf == NULL)
    {
        return -1;
    }

    int result = -1;

    for (unsigned int i = 0; i < n; ++i)
    {
        if (_cypher_ast_fprint(asts[i], stream, colorization,
                    &buf, &bufcap, width, ordinal_width, start_width,
                    end_width, name_width, 0) < 0)
        {
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    free(buf);
    return result;
}


int cypher_astnode_init(cypher_astnode_t *node,
        cypher_astnode_type_t type, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    assert(node != NULL);
    assert(nchildren == 0 || children != NULL);

    node->type = type;
    node->range = range;
    if (nchildren > 0)
    {
        node->children = mdup(children, nchildren * sizeof(cypher_astnode_t *));
        if (node->children == NULL)
        {
            return -1;
        }
        node->nchildren = nchildren;
    }
    else
    {
        node->children = NULL;
        node->nchildren = 0;
    }
    return 0;
}


void cypher_astnode_release(cypher_astnode_t *node)
{
    free(node);
}


ssize_t cypher_astnode_detailstr(const cypher_astnode_t *node, char *str,
        size_t size)
{
    assert(node->type < _MAX_VT_OFF);
    const struct cypher_astnode_vt *vt = VT_PTR(node->type);
    return vt->detailstr(node, str, size);
}


unsigned int cypher_astnode_nchildren(const cypher_astnode_t *node)
{
    return node->nchildren;
}


const cypher_astnode_t *cypher_astnode_get_child(const cypher_astnode_t *node,
        unsigned int index)
{
    if (index >= node->nchildren)
    {
        return NULL;
    }
    return node->children[index];
}


void cypher_astnode_set_child(cypher_astnode_t *parent,
        cypher_astnode_t *child, unsigned int index)
{
    assert(index < parent->nchildren);
    parent->children[index] = child;
}


ssize_t snprint_sequence(char *str, size_t size,
        const cypher_astnode_t * const *elements, unsigned int nelements)
{
    size_t n = 0;
    if (n < size)
    {
        str[n] = '[';
    }
    n++;
    for (unsigned int i = 0; i < nelements; )
    {
        ssize_t r = snprintf(str+n, (n < size)? size-n : 0,
                "@%u", elements[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
        if (++i < nelements)
        {
            if (n < size)
            {
                str[n] = ',';
            }
            n++;
            if (n < size)
            {
                str[n] = ' ';
            }
            n++;
        }
    }
    if (n < size)
    {
        str[n] = ']';
    }
    n++;
    return n;
}
