lib_LTLIBRARIES = libcypher-parser.la

include_HEADERS = cypher-parser.h
libcypher_parser_la_SOURCES = \
	annotation.c \
	annotation.h \
	ast.c \
	ast.h \
	astnode.h \
	ast_all.c \
	ast_all_nodes_scan.c \
	ast_all_rels_scan.c \
	ast_any.c \
	ast_apply_operator.c \
	ast_apply_all_operator.c \
	ast_boolean.c \
	ast_binary_operator.c \
	ast_block_comment.c \
	ast_call.c \
	ast_call_subquery.c \
	ast_case.c \
	ast_collection.c \
	ast_command.c \
	ast_comment.c \
	ast_comparison.c \
	ast_create.c \
	ast_create_pattern_props_index.c \
	ast_create_node_props_index.c \
	ast_create_node_prop_constraint.c \
	ast_create_rel_prop_constraint.c \
	ast_cypher_option.c \
	ast_cypher_option_param.c \
	ast_delete.c \
	ast_drop_props_index.c \
	ast_drop_pattern_props_index.c \
	ast_drop_node_prop_constraint.c \
	ast_drop_rel_prop_constraint.c \
	ast_error.c \
	ast_explain_option.c \
	ast_expression.c \
	ast_extract.c \
	ast_false.c \
	ast_filter.c \
	ast_float.c \
	ast_foreach.c \
	ast_function_name.c \
	ast_identifier.c \
	ast_index_name.c \
	ast_integer.c \
	ast_label.c \
	ast_labels_operator.c \
	ast_line_comment.c \
	ast_list_comprehension.c \
	ast_load_csv.c \
	ast_match.c \
	ast_match_hint.c \
	ast_map.c \
	ast_merge.c \
	ast_merge_action.c \
	ast_merge_properties.c \
	ast_named_path.c \
	ast_node_id_lookup.c \
	ast_node_index_lookup.c \
	ast_node_index_query.c \
	ast_node_pattern.c \
	ast_none.c \
	ast_null.c \
	ast_on_create.c \
	ast_on_match.c \
	ast_order_by.c \
	ast_parameter.c \
	ast_pattern.c \
	ast_pattern_comprehension.c \
	ast_pattern_path.c \
	ast_proc_name.c \
	ast_profile_option.c \
	ast_projection.c \
	ast_prop_name.c \
	ast_property_operator.c \
	ast_query.c \
	ast_query_clause.c \
	ast_query_option.c \
	ast_range.c \
	ast_reduce.c \
	ast_rel_id_lookup.c \
	ast_rel_index_lookup.c \
	ast_rel_index_query.c \
	ast_remove.c \
	ast_remove_item.c \
	ast_remove_labels.c \
	ast_remove_property.c \
	ast_rel_pattern.c \
	ast_reltype.c \
	ast_return.c \
	ast_schema_command.c \
	ast_set.c \
	ast_set_all_properties.c \
	ast_set_item.c \
	ast_set_labels.c \
	ast_set_property.c \
	ast_shortest_path.c \
	ast_single.c \
	ast_slice_operator.c \
	ast_sort_item.c \
	ast_start.c \
	ast_start_point.c \
	ast_statement.c \
	ast_statement_option.c \
	ast_string.c \
	ast_subscript_operator.c \
	ast_map_projection.c \
	ast_map_projection_selector.c \
	ast_map_projection_literal.c \
	ast_map_projection_property.c \
	ast_map_projection_identifier.c \
	ast_map_projection_all_properties.c \
	ast_true.c \
	ast_unary_operator.c \
	ast_union.c \
	ast_unwind.c \
	ast_using_index.c \
	ast_using_join.c \
	ast_using_periodic_commit.c \
	ast_using_scan.c \
	ast_with.c \
	errors.c \
	errors.h \
	operators.c \
	operators.h \
	parser.c \
	parser.leg \
	parser_config.c \
	parser_config.h \
	quick_parser.c \
	quick_parser.leg \
	result.c \
	result.h \
	segment.c \
	segment.h \
	string_buffer.c \
	string_buffer.h \
	util.c \
	util.h \
	vector.c \
	vector.h

libcypher_parser_la_CFLAGS = $(PTHREAD_CFLAGS) $(OPENSSL_CFLAGS) -Wpedantic -Wvla
libcypher_parser_la_LIBADD = $(PTHREAD_LIBS) $(OPENSSL_LIBS) $(LIBM)
# Updating the version-info:
# 1. If the library source code has changed at all since the last update, then
#    increment revision (‘c:r:a’ becomes ‘c:r+1:a’).
# 2. If any interfaces have been added, removed, or changed since the last
#    update, increment current, and set revision to 0.
# 3. If any interfaces have been added since the last public release, then
#    increment age.
# 4. If any interfaces have been removed or changed since the last public
#    release, then set age to 0.
libcypher_parser_la_LDFLAGS = -version-info 10:2:2

parser.c: parser_leg.c
quick_parser.c: quick_parser_leg.c

if HAVE_LEG
%_leg.c: %.leg
	$(LEG) -o $@ $<
else
%_leg.c: %.leg
	$(error Cannot compile .leg source file, as the leg compiler was not found by the configure script: please install http://piumarta.com/software/peg/ and re-run the configure script)
endif

EXTRA_DIST = parser_leg.c quick_parser_leg.c
MAINTAINERCLEANFILES = parser_leg.c quick_parser_leg.c
