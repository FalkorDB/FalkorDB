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
#include "cypher-parser.h"
#include "ast.h"
#include "errors.h"
#include "operators.h"
#include "parser_config.h"
#include "result.h"
#include "segment.h"
#include "string_buffer.h"
#include "util.h"
#include "vector.h"
#include <assert.h>
#include <ctype.h>
#include <setjmp.h>

DECLARE_VECTOR(offsets, unsigned int, 0);
DECLARE_VECTOR(precedences, unsigned int, 0);
DECLARE_VECTOR(operators, const cypher_operator_t *, NULL);
DECLARE_VECTOR(astnodes, cypher_astnode_t *, NULL);

struct block
{
    size_t buffer_start;
    size_t buffer_end;
    struct cypher_input_range range;
    astnodes_t sequence;
    astnodes_t children;
};

DECLARE_VECTOR(blocks, struct block *, NULL);

typedef struct _yycontext yycontext;
typedef int (*yyrule)(yycontext *yy);

/**
 * Return the appropriate yyrule according to the given flags.
 * 
 * @param  flags: Libcypher parser entry point flags.
 * @return yyrule which fit the flag entry point.
 */
yyrule cypher_yyrule_from_flags(int flags);

typedef int (*source_cb_t)(void *data, char *buf, int n);

static int parse_each(yyrule rule, source_cb_t source, void *sourcedata,
        cypher_parser_segment_callback_t callback, void *userdata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags);
static cypher_parse_result_t *parse(yyrule rule, source_cb_t source,
        void *sourcedata, struct cypher_input_position *last,
        cypher_parser_config_t *config, uint_fast32_t flags);
static int parse_one(yycontext *yy, yyrule rule);
static void source(yycontext *yy, char *buf, int *result, int max_size);


struct source_from_buffer_data
{
    const char *buffer;
    size_t length;
};


static int source_from_buffer(void *data, char *buf, int n)
{
    struct source_from_buffer_data *input = data;
    int len = min(input->length, n);
    input->length -= len;
    if (len == 0)
    {
        return len;
    }
    memcpy(buf, input->buffer, len);
    return len;
}


static int uparse_each(yyrule rule, const char *s, size_t n,
        cypher_parser_segment_callback_t callback, void *userdata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    REQUIRE(s != NULL, -1);
    REQUIRE(callback != NULL, -1);
    struct source_from_buffer_data sourcedata = { .buffer = s, .length = n };
    return parse_each(rule, source_from_buffer, &sourcedata, callback,
            userdata, last, config, flags);
}


static cypher_parse_result_t *uparse(yyrule rule, const char *s, size_t n,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    REQUIRE(s != NULL, NULL);
    struct source_from_buffer_data sourcedata = { .buffer = s, .length = n };
    return parse(rule, source_from_buffer, &sourcedata, last, config, flags);
}


static int source_from_stream(void *data, char *buf, int n)
{
    FILE *stream = data;
    int c = getc(stream);
    if (c == EOF)
    {
        return 0;
    }
    *buf = c;
    return 1;
}


static int fparse_each(yyrule rule, FILE *stream,
        cypher_parser_segment_callback_t callback, void *userdata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    REQUIRE(stream != NULL, -1);
    REQUIRE(callback != NULL, -1);
    return parse_each(rule, source_from_stream, stream, callback, userdata,
            last, config, flags);
}


static cypher_parse_result_t *fparse(yyrule rule, FILE *stream,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    REQUIRE(stream != NULL, NULL);
    return parse(rule, source_from_stream, stream, last, config, flags);
}


static void *abort_malloc(yycontext *yy, size_t size);
static void *abort_realloc(yycontext *yy, void *ptr, size_t size);
static void finished(yycontext *yy);
static void line_start(yycontext *yy);
static void block_start_action(yycontext *yy, char *text, int count);
static struct block *block_start(yycontext *yy, size_t offset,
        struct cypher_input_position position);
static void block_end_action(yycontext *yy, char *text, int count);
static void block_replace_action(yycontext *yy, char *text, int pos);
static void block_merge_action(yycontext *yy, char *text, int pos);
static struct block *block_end(yycontext *yy, size_t offset,
        struct cypher_input_position position);

#define ERR(label) _err(yy, label)
static void _err(yycontext *yy, const char *msg);
static void record_error(yycontext *yy);

#define strbuf_reset() cp_sb_reset(&(yy->string_buffer))
#define strbuf_append(s, n) _strbuf_append(yy, s, n)
static void _strbuf_append(yycontext *yy, const char *s, size_t n);
#define strbuf_append_block() _strbuf_append_block(yy)
static void _strbuf_append_block(yycontext *yy);

#define sequence_add(node) _sequence_add(yy, node)
static void _sequence_add(yycontext *yy, cypher_astnode_t *node);
#define collection_literal() _collection_literal(yy)
static cypher_astnode_t *_collection_literal(yycontext *yy);

#define OP(n) (yy->op = CYPHER_OP_##n, 1)
#define op_push(n) _op_push(yy, CYPHER_OP_##n)
static void _op_push(yycontext *yy, const cypher_operator_t *op);
#define op_pop() operators_pop(&(yy->operators))

#define PREC_PUSH() (_prec_push(yy), 1)
static void _prec_push(yycontext *yy);
#define PREC_PUSH_TOP() (_prec_push_top(yy), 1)
static void _prec_push_top(yycontext *yy);
#define PREC_CHK() \
    ((yy->op->precedence >= precedences_last(&(yy->precedences)))? 1 : 0)
#define PREC_POP() (precedences_pop(&(yy->precedences)), 1)

#define statement(b) _statement(yy, b)
static cypher_astnode_t *_statement(yycontext *yy, cypher_astnode_t *body);
#define cypher_option(b) _cypher_option(yy, b)
static cypher_astnode_t *_cypher_option(yycontext *yy,
        cypher_astnode_t *version);
#define cypher_option_param(n, v) _cypher_option_param(yy, n, v)
static cypher_astnode_t *_cypher_option_param(yycontext *yy,
        cypher_astnode_t *name, cypher_astnode_t *value);
#define explain_option() _explain_option(yy)
static cypher_astnode_t *_explain_option(yycontext *yy);
#define profile_option() _profile_option(yy)
static cypher_astnode_t *_profile_option(yycontext *yy);
#define create_index(l) _create_index(yy, l)
static cypher_astnode_t *_create_index(yycontext *yy, cypher_astnode_t *label);
#define create_pattern_index(i, t, it, o, r) _create_pattern_index(yy, i, t, it, o, r)
static cypher_astnode_t *_create_pattern_index(yycontext *yy,
         cypher_astnode_t *identifier, cypher_astnode_t *label,
         enum cypher_ast_index_type index_type, cypher_astnode_t *options,
         bool is_relation);
#define drop_index(l) _drop_index(yy, l)
static cypher_astnode_t *_drop_index(yycontext *yy, cypher_astnode_t *label);
#define drop_pattern_index(i, t, it, r) _drop_pattern_index(yy, i, t, it, r)
static cypher_astnode_t *_drop_pattern_index(yycontext *yy,
         cypher_astnode_t *identifier, cypher_astnode_t *label,
         enum cypher_ast_index_type index_type, bool is_relation);
#define create_node_prop_constraint(i, l, e, u) \
        _create_node_prop_constraint(yy, i, l, e, u)
static cypher_astnode_t *_create_node_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique);
#define drop_node_prop_constraint(i, l, e, u) \
        _drop_node_prop_constraint(yy, i, l, e, u)
static cypher_astnode_t *_drop_node_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique);
#define create_rel_prop_constraint(i, l, e, u) \
        _create_rel_prop_constraint(yy, i, l, e, u)
static cypher_astnode_t *_create_rel_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique);
#define drop_rel_prop_constraint(i, l, e, u) \
        _drop_rel_prop_constraint(yy, i, l, e, u)
static cypher_astnode_t *_drop_rel_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique);
#define query() _query(yy)
static cypher_astnode_t *_query(yycontext *yy);
#define using_periodic_commit(l) _using_periodic_commit(yy, l)
static cypher_astnode_t *_using_periodic_commit(yycontext *yy,
        cypher_astnode_t *limit);
#define load_csv(wh, url, id, ft) _load_csv(yy, wh, url, id, ft)
static cypher_astnode_t *_load_csv(yycontext *yy, bool with_headers,
        cypher_astnode_t *url, cypher_astnode_t *identifier,
        cypher_astnode_t *field_terminator);
#define start_clause(c) _start_clause(yy, c)
static cypher_astnode_t *_start_clause(yycontext *yy, 
        cypher_astnode_t *predicate);
#define node_index_lookup(i, x, p, l) _node_index_lookup(yy, i, x, p, l)
static cypher_astnode_t *_node_index_lookup(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *prop_name, cypher_astnode_t *lookup);
#define node_index_query(i, x, q) _node_index_query(yy, i, x, q)
static cypher_astnode_t *_node_index_query(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *query);
#define node_id_lookup(i) _node_id_lookup(yy, i)
static cypher_astnode_t *_node_id_lookup(yycontext *yy,
        cypher_astnode_t *identifier);
#define all_nodes_scan(i) _all_nodes_scan(yy, i)
static cypher_astnode_t *_all_nodes_scan(yycontext *yy,
        cypher_astnode_t *identifier);
#define rel_index_lookup(i, x, p, l) _rel_index_lookup(yy, i, x, p, l)
static cypher_astnode_t *_rel_index_lookup(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *prop_name, cypher_astnode_t *lookup);
#define rel_index_query(i, x, q) _rel_index_query(yy, i, x, q)
static cypher_astnode_t *_rel_index_query(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *query);
#define rel_id_lookup(i) _rel_id_lookup(yy, i)
static cypher_astnode_t *_rel_id_lookup(yycontext *yy,
        cypher_astnode_t *identifier);
#define all_rels_scan(i) _all_rels_scan(yy, i)
static cypher_astnode_t *_all_rels_scan(yycontext *yy,
        cypher_astnode_t *identifier);
#define match_clause(o, p, c) _match_clause(yy, o, p, c)
static cypher_astnode_t *_match_clause(yycontext *yy, bool optional,
        cypher_astnode_t *pattern, cypher_astnode_t *predicate);
#define using_index(i, l, p) _using_index(yy, i, l, p)
static cypher_astnode_t *_using_index(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *prop_name);
#define using_join() _using_join(yy)
static cypher_astnode_t *_using_join(yycontext *yy);
#define using_scan(i, l) _using_scan(yy, i, l)
static cypher_astnode_t *_using_scan(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label);
#define merge_clause(p) _merge_clause(yy, p)
static cypher_astnode_t *_merge_clause(yycontext *yy,
        cypher_astnode_t *pattern_part);
#define on_match() _on_match(yy)
static cypher_astnode_t *_on_match(yycontext *yy);
#define on_create() _on_create(yy)
static cypher_astnode_t *_on_create(yycontext *yy);
#define create_clause(u, p) _create_clause(yy, u, p)
static cypher_astnode_t *_create_clause(yycontext *yy, bool unique,
        cypher_astnode_t *pattern);
#define set_clause() _set_clause(yy)
static cypher_astnode_t *_set_clause(yycontext *yy);
#define set_property(p, e) _set_property(yy, p, e)
static cypher_astnode_t *_set_property(yycontext *yy,
        cypher_astnode_t *prop_name, cypher_astnode_t *expression);
#define set_all_properties(i, e) _set_all_properties(yy, i, e)
static cypher_astnode_t *_set_all_properties(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression);
#define merge_properties(i, e) _merge_properties(yy, i, e)
static cypher_astnode_t *_merge_properties(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression);
#define set_labels(i) _set_labels(yy, i)
static cypher_astnode_t *_set_labels(yycontext *yy,
        cypher_astnode_t *identifier);
#define delete(m) _delete(yy, m)
static cypher_astnode_t *_delete(yycontext *yy, cypher_ast_delete_mode_t mode);
#define remove_clause() _remove_clause(yy)
static cypher_astnode_t *_remove_clause(yycontext *yy);
#define remove_property(p) _remove_property(yy, p)
static cypher_astnode_t *_remove_property(yycontext *yy,
        cypher_astnode_t *prop_name);
#define remove_labels(i) _remove_labels(yy, i)
static cypher_astnode_t *_remove_labels(yycontext *yy,
        cypher_astnode_t *identifier);
#define foreach_clause(i, e) _foreach_clause(yy, i, e)
static cypher_astnode_t *_foreach_clause(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression);
#define with_clause(d, a, o, s, l, p) _with_clause(yy, d, a, o, s, l, p)
static cypher_astnode_t *_with_clause(yycontext *yy, bool distinct,
        bool include_existing, cypher_astnode_t *order_by,
        cypher_astnode_t *skip, cypher_astnode_t *limit,
        cypher_astnode_t *predicate);
#define unwind_clause(e, i) _unwind_clause(yy, e, i)
static cypher_astnode_t *_unwind_clause(yycontext *yy,
        cypher_astnode_t *expression, cypher_astnode_t *identifier);
#define call_clause(p, w) _call_clause(yy, p, w)
static cypher_astnode_t *_call_clause(yycontext *yy,
        cypher_astnode_t *proc_name, cypher_astnode_t *predicate);
#define call_subquery() _call_subquery(yy)
static cypher_astnode_t *_call_subquery(yycontext *yy);
#define return_clause(d, a, o, s, l) _return_clause(yy, d, a, o, s, l)
static cypher_astnode_t *_return_clause(yycontext *yy, bool distinct,
        bool include_existing, cypher_astnode_t *order_by,
        cypher_astnode_t *skip, cypher_astnode_t *limit);
#define projection(e, a) _projection(yy, e, a)
static cypher_astnode_t *_projection(yycontext *yy,
        cypher_astnode_t *expression, cypher_astnode_t *alias);
#define order_by() _order_by(yy)
static cypher_astnode_t *_order_by(yycontext *yy);
#define sort_item(e, a) _sort_item(yy, e, a)
static cypher_astnode_t *_sort_item(yycontext *yy, cypher_astnode_t *expression,
        bool ascending);
#define union_clause(a) _union_clause(yy, a)
static cypher_astnode_t *_union_clause(yycontext *yy, bool all);
#define unary_operator(o, a) _unary_operator(yy, o, a)
static cypher_astnode_t *_unary_operator(yycontext *yy,
        const cypher_operator_t *op, cypher_astnode_t *arg);
#define binary_operator(o, l, r) _binary_operator(yy, o, l, r)
static cypher_astnode_t *_binary_operator(yycontext *yy,
        const cypher_operator_t *op, cypher_astnode_t *left,
        cypher_astnode_t *right);
#define comparison_operator() _comparison_operator(yy)
static cypher_astnode_t *_comparison_operator(yycontext *yy);
#define apply_operator(l, d) _apply_operator(yy, l, d)
static cypher_astnode_t *_apply_operator(yycontext *yy, cypher_astnode_t *left,
        bool distinct);
#define apply_all_operator(l, d) _apply_all_operator(yy, l, d)
static cypher_astnode_t *_apply_all_operator(yycontext *yy,
        cypher_astnode_t *left, bool distinct);
#define property_operator(l, r) _property_operator(yy, l, r)
static cypher_astnode_t *_property_operator(yycontext *yy,
        cypher_astnode_t *map, cypher_astnode_t *prop_name);
#define subscript_operator(l, r) _subscript_operator(yy, l, r)
static cypher_astnode_t *_subscript_operator(yycontext *yy,
        cypher_astnode_t *arg, cypher_astnode_t *subscript);
#define slice_operator(l, s, e) _slice_operator(yy, l, s, e)
static cypher_astnode_t *_slice_operator(yycontext *yy,
        cypher_astnode_t *expression, cypher_astnode_t *start,
        cypher_astnode_t* end);
#define map_projection(l) _map_projection(yy, l)
static cypher_astnode_t *_map_projection(yycontext *yy,
        cypher_astnode_t *expression);
#define map_projection_literal(p, e) _map_projection_literal(yy, p, e)
static cypher_astnode_t *_map_projection_literal(yycontext *yy,
        cypher_astnode_t *prop_name, cypher_astnode_t *expression);
#define map_projection_property(p) _map_projection_property(yy, p)
static cypher_astnode_t *_map_projection_property(yycontext *yy,
        cypher_astnode_t *prop_name);
#define map_projection_identifier(p) _map_projection_identifier(yy, p)
static cypher_astnode_t *_map_projection_identifier(yycontext *yy,
        cypher_astnode_t *identifier);
#define map_projection_all_properties() _map_projection_all_properties(yy)
static cypher_astnode_t *_map_projection_all_properties(yycontext *yy);
#define labels_operator(l) _labels_operator(yy, l)
static cypher_astnode_t *_labels_operator(yycontext *yy,
        cypher_astnode_t *left);
#define list_comprehension(i,e,p,v) _list_comprehension(yy, i, e, p, v)
static cypher_astnode_t *_list_comprehension(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate, cypher_astnode_t *eval);
#define pattern_comprehension(i,r,p,v) _pattern_comprehension(yy, i, r, p, v)
static cypher_astnode_t *_pattern_comprehension(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *pattern,
        cypher_astnode_t *predicate, cypher_astnode_t *eval);
#define case_expression(e,d) _case_expression(yy, e, d)
static cypher_astnode_t *_case_expression(yycontext *yy,
        cypher_astnode_t *expression, cypher_astnode_t *deflt);
#define filter(i,e,p) _filter(yy, i, e, p)
static cypher_astnode_t *_filter(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate);
#define extract(i,e,v) _extract(yy, i, e, v)
static cypher_astnode_t *_extract(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *eval);
#define reduce(a,n,i,e,v) _reduce(yy, a, n, i, e, v)
static cypher_astnode_t *_reduce(yycontext *yy, cypher_astnode_t *accumulator,
        cypher_astnode_t *init, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *eval);
#define all_predicate(i,e,p) _all_predicate(yy, i, e, p)
static cypher_astnode_t *_all_predicate(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate);
#define any_predicate(i,e,p) _any_predicate(yy, i, e, p)
static cypher_astnode_t *_any_predicate(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate);
#define single_predicate(i,e,p) _single_predicate(yy, i, e, p)
static cypher_astnode_t *_single_predicate(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate);
#define none_predicate(i,e,p) _none_predicate(yy, i, e, p)
static cypher_astnode_t *_none_predicate(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate);
#define map_literal() _map_literal(yy)
static cypher_astnode_t *_map_literal(yycontext *yy);
#define strbuf_identifier() _strbuf_identifier(yy)
static cypher_astnode_t *_strbuf_identifier(yycontext *yy);
#define block_identifier() _block_identifier(yy)
static cypher_astnode_t *_block_identifier(yycontext *yy);
#define strbuf_parameter() _strbuf_parameter(yy)
static cypher_astnode_t *_strbuf_parameter(yycontext *yy);
#define strbuf_integer() _strbuf_integer(yy)
static cypher_astnode_t *_strbuf_integer(yycontext *yy);
#define strbuf_float() _strbuf_float(yy)
static cypher_astnode_t *_strbuf_float(yycontext *yy);
#define true_literal() _true_literal(yy)
static cypher_astnode_t *_true_literal(yycontext *yy);
#define false_literal() _false_literal(yy)
static cypher_astnode_t *_false_literal(yycontext *yy);
#define null_literal() _null_literal(yy)
static cypher_astnode_t *_null_literal(yycontext *yy);
#define strbuf_label() _strbuf_label(yy)
static cypher_astnode_t *_strbuf_label(yycontext *yy);
#define strbuf_reltype() _strbuf_reltype(yy)
static cypher_astnode_t *_strbuf_reltype(yycontext *yy);
#define strbuf_prop_name() _strbuf_prop_name(yy)
static cypher_astnode_t *_strbuf_prop_name(yycontext *yy);
#define strbuf_function_name() _strbuf_function_name(yy)
static cypher_astnode_t *_strbuf_function_name(yycontext *yy);
#define strbuf_index_name() _strbuf_index_name(yy)
static cypher_astnode_t *_strbuf_index_name(yycontext *yy);
#define strbuf_proc_name() _strbuf_proc_name(yy)
static cypher_astnode_t *_strbuf_proc_name(yycontext *yy);
#define pattern() _pattern(yy)
static cypher_astnode_t *_pattern(yycontext *yy);
#define named_path(s, p) _named_path(yy, s, p)
static cypher_astnode_t *_named_path(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *path);
#define shortest_path(s, p) _shortest_path(yy, s, p)
static cypher_astnode_t *_shortest_path(yycontext *yy, bool single,
        cypher_astnode_t *path);
#define pattern_path() _pattern_path(yy)
static cypher_astnode_t *_pattern_path(yycontext *yy);
#define node_pattern(i, p) _node_pattern(yy, i, p)
static cypher_astnode_t *_node_pattern(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *properties);
#define simple_rel_pattern(d) \
    _rel_pattern(yy, CYPHER_REL_##d, NULL, NULL, NULL)
#define rel_pattern(d, i, r, p) _rel_pattern(yy, CYPHER_REL_##d, i, r, p)
static cypher_astnode_t *_rel_pattern(yycontext *yy,
        enum cypher_rel_direction direction, cypher_astnode_t *identifier,
        cypher_astnode_t *varlength, cypher_astnode_t *properties);
#define range(s, e) _range(yy, s, e)
static cypher_astnode_t *_range(yycontext *yy, cypher_astnode_t *start,
        cypher_astnode_t *end);
#define command(name) _command(yy, name)
static cypher_astnode_t *_command(yycontext *yy, cypher_astnode_t *name);
#define string(s, n) _string(yy, s, n)
static cypher_astnode_t *_string(yycontext *yy, const char *s, size_t n);
#define block_string() _block_string(yy)
static cypher_astnode_t *_block_string(yycontext *yy);
#define strbuf_string() _strbuf_string(yy)
static cypher_astnode_t *_strbuf_string(yycontext *yy);
#define line_comment() _line_comment(yy)
static cypher_astnode_t *_line_comment(yycontext *yy);
#define block_comment() _block_comment(yy)
static cypher_astnode_t *_block_comment(yycontext *yy);
#define skip() _skip(yy)
static cypher_astnode_t *_skip(yycontext *yy);


#define YY_CTX_MEMBERS \
    cypher_parser_config_t *config; \
    sigjmp_buf abort_env; \
    struct cypher_input_position position_offset; \
    offsets_t line_start_offsets; \
    blocks_t blocks; \
    struct block *prev_block; /* last "closed" block */ \
    struct cp_string_buffer string_buffer; \
    const cypher_operator_t *op; \
    operators_t operators; \
    precedences_t precedences; \
    source_cb_t source; \
    void *source_data; \
    cypher_astnode_t *result; \
    bool eof; \
    cp_error_tracking_t error_tracking; \
    unsigned int consumed;

#define YYSTYPE cypher_astnode_t *

#define YY_MALLOC abort_malloc
#define YY_REALLOC abort_realloc

#define YY_BEGIN \
    (yy->__begin = yy->__pos, yyDo(yy, block_start_action, yy->__pos, 0), 1)
#define YY_END \
    (yy->__end = 0, yyDo(yy, block_end_action, yy->__pos, 0), 1)

#define YY_CTX_LOCAL
#define YY_PARSE(T) static T

#define YY_INPUT(yy, buf, result, max_size) \
    source((yy), (buf), &(result), (max_size))

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-value"
#include "parser_leg.c"


#define abort_parse(yy) \
    do { assert(errno != 0); siglongjmp(yy->abort_env, errno); } while (0)
static int safe_yyparsefrom(yycontext *yy, yyrule rule);
static unsigned int backtrack_lines(yycontext *yy, unsigned int pos);
static struct cypher_input_position input_position(yycontext *yy,
        unsigned int pos);
static void block_free(struct block *block);
static cypher_astnode_t *add_terminal(yycontext *yy, cypher_astnode_t *node);
static cypher_astnode_t *add_child(yycontext *yy, cypher_astnode_t *node);


void source(yycontext *yy, char *buf, int *result, int max_size)
{
    if (buf == NULL)
    {
        *result = 0;
        return;
    }
    assert(yy != NULL && yy->source != NULL);
    *result = yy->source(yy->source_data, buf, max_size);
}


int cypher_uparse_each(const char *s, size_t n,
        cypher_parser_segment_callback_t callback, void *userdata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{

    yyrule rule = cypher_yyrule_from_flags(flags);
    
    return uparse_each(rule, s, n, callback, userdata, last, config, flags);
}


cypher_parse_result_t *cypher_uparse(const char *s, size_t n,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    yyrule rule = cypher_yyrule_from_flags(flags);
    return uparse(rule, s, n, last, config, flags);
}


int cypher_fparse_each(FILE *stream, cypher_parser_segment_callback_t callback,
        void *userdata, struct cypher_input_position *last,
        cypher_parser_config_t *config, uint_fast32_t flags)
{
    yyrule rule = cypher_yyrule_from_flags(flags);
    return fparse_each(rule, stream, callback, userdata, last, config, flags);
}


cypher_parse_result_t *cypher_fparse(FILE *stream,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    yyrule rule = cypher_yyrule_from_flags(flags);
    return fparse(rule, stream, last, config, flags);
}


int parse_each(yyrule rule, source_cb_t source, void *sourcedata,
        cypher_parser_segment_callback_t callback, void *userdata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    int result = -1;

    yycontext yy;
    memset(&yy, 0, sizeof(yycontext));
    yy.config = (config != NULL)? config : &cypher_parser_std_config;
    yy.position_offset = yy.config->initial_position;
    offsets_init(&(yy.line_start_offsets));
    blocks_init(&(yy.blocks));
    operators_init(&(yy.operators));
    precedences_init(&(yy.precedences));
    yy.source = source;
    yy.source_data = sourcedata;
    cp_et_init(&(yy.error_tracking), yy.config->error_colorization);

    struct block *top_block = NULL;

    if (offsets_push(&(yy.line_start_offsets), 0))
    {
        goto cleanup;
    }
    top_block = block_start(&yy, 0, input_position(&yy, 0));
    if (top_block == NULL)
    {
        goto cleanup;
    }

    unsigned int ordinal = yy.config->initial_ordinal;

    for (;;)
    {
        if (parse_one(&yy, rule))
        {
            goto cleanup;
        }

        if (yy.consumed == 0)
        {
            assert(yy.result == NULL);
            assert(cp_et_nerrors(&(yy.error_tracking)) == 0);
            assert(yy.eof);
            break;
        }

        // TODO: last should be set even on parse failure
        if (last != NULL)
        {
            *last = input_position(&yy, yy.consumed);
        }

        struct cypher_input_range range =
            { .start = yy.position_offset,
              .end = input_position(&yy, yy.consumed) };

        cypher_parse_error_t *errors = cp_et_errors(&(yy.error_tracking));
        unsigned int nerrors = cp_et_nerrors(&(yy.error_tracking));
        cypher_astnode_t **roots = astnodes_elements(&(top_block->children));
        unsigned int nroots = astnodes_size(&(top_block->children));

        cypher_parse_segment_t *segment = cypher_parse_segment(ordinal,
                range, errors, nerrors, roots, nroots, yy.result, yy.eof);
        if (segment == NULL)
        {
            goto cleanup;
        }

        cp_et_clear_errors(&(yy.error_tracking));
        astnodes_clear(&(top_block->children));
        ordinal += segment->nnodes;

        int err = callback(userdata, segment);
        cypher_parse_segment_release(segment);
        if (err > 0)
        {
            break;
        }
        else if (err)
        {
            result = err;
            goto cleanup;
        }

        if (yy.eof || flags & CYPHER_PARSE_SINGLE)
        {
            break;
        }

        yy.position_offset = range.end;

        offsets_clear(&(yy.line_start_offsets));
        if (offsets_push(&(yy.line_start_offsets), 0))
        {
            goto cleanup;
        }
    }

    result = 0;

    int errsv;
cleanup:
    errsv = errno;
    offsets_cleanup(&(yy.line_start_offsets));
    block_free(top_block);
    blocks_cleanup(&(yy.blocks));
    operators_cleanup(&(yy.operators));
    precedences_cleanup(&(yy.precedences));
    cp_et_cleanup(&(yy.error_tracking));
    cp_sb_cleanup(&(yy.string_buffer));
    yyrelease(&yy);
    errno = errsv;
    return result;
}


static int parse_all_callback(void *data, cypher_parse_segment_t *segment)
{
    cypher_parse_result_t *result = (cypher_parse_result_t *)data;
    return cp_result_merge_segment(result, segment);
}


cypher_parse_result_t *parse(yyrule rule, source_cb_t source, void *sourcedata,
        struct cypher_input_position *last, cypher_parser_config_t *config,
        uint_fast32_t flags)
{
    cypher_parse_result_t *result = calloc(1, sizeof(cypher_parse_result_t));
    if (result == NULL)
    {
        return NULL;
    }

    if (parse_each(rule, source, sourcedata, parse_all_callback, result,
                last, config, flags))
    {
        cypher_parse_result_free(result);
        return NULL;
    }

    return result;
}


int parse_one(yycontext *yy, yyrule rule)
{
#ifndef NDEBUG
    struct block *top_block = blocks_last(&(yy->blocks));
#endif

    yy->result = NULL;
    yy->eof = false;
    if (safe_yyparsefrom(yy, rule) <= 0)
    {
        goto failure;
    }

    assert(blocks_size(&(yy->blocks)) == 1 &&
            "Mismatched use of `<` and `>` in grammar");
    assert(blocks_last(&(yy->blocks)) == top_block &&
            "Top-most block has been replaced");
    assert(astnodes_size(&(top_block->sequence)) == 0 &&
            "AST nodes left in top-most block sequence");
    assert(yy->prev_block == NULL &&
            "Block closed but not cleared - "
            "no AST node created after final `>` in grammar");
    assert(operators_size(&(yy->operators)) == 0 &&
            "Operator stack not emptied");
    assert(precedences_size(&(yy->precedences)) == 0 &&
            "Precedence stack not emptied");

    cp_et_clear_potentials(&(yy->error_tracking));

    return 0;

    int errsv;
failure:
    errsv = errno;
    offsets_clear(&(yy->line_start_offsets));
    struct block *block;
    while ((block = blocks_pop(&(yy->blocks))) != NULL)
    {
        block_free(block);
    }
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    operators_clear(&(yy->operators));
    precedences_clear(&(yy->precedences));
    cp_et_clear_potentials(&(yy->error_tracking));
    errno = errsv;
    return -1;
}


int safe_yyparsefrom(yycontext *yy, yyrule rule)
{
    int err;
    if ((err = sigsetjmp(yy->abort_env, 0)) != 0)
    {
        errno = err;
        return -1;
    }

    int result = yyparsefrom(yy, rule);
    memset(yy->abort_env, 0, sizeof(sigjmp_buf));
    return result;
}


void *abort_malloc(yycontext *yy, size_t size)
{
    void *m = malloc(size);
    if (m == NULL)
    {
        abort_parse(yy);
    }
    return m;
}


void *abort_realloc(yycontext *yy, void *ptr, size_t size)
{
    void *m = realloc(ptr, size);
    if (m == NULL)
    {
        abort_parse(yy);
    }
    return m;
}


void finished(yycontext *yy)
{
    yy->consumed = yy->__pos;

    for (unsigned int i = cp_et_nerrors(&(yy->error_tracking)); i-- > 0; )
    {
        cypher_parse_error_t *err = yy->error_tracking.errors + i;
        if (err->context != NULL)
        {
            break;
        }

        size_t ctx_offset = err->position.offset - yy->position_offset.offset;
        char *ctx = line_context(yy->__buf, yy->__limit, &ctx_offset, 80);
        if (ctx == NULL)
        {
            abort_parse(yy);
        }
        err->context = ctx;
        err->context_offset = ctx_offset;
    }
}


void line_start(yycontext *yy)
{
    assert(yy->__pos >= 0);
    unsigned int pos = (unsigned int)yy->__pos;
    unsigned int line_start_pos = backtrack_lines(yy, pos);
    if (line_start_pos == pos)
    {
        return;
    }
    if (offsets_push(&(yy->line_start_offsets), pos))
    {
        abort_parse(yy);
    }
}


unsigned int backtrack_lines(yycontext *yy, unsigned int pos)
{
    unsigned int top;
    while ((top = offsets_last(&(yy->line_start_offsets))) > pos)
    {
        offsets_pop(&(yy->line_start_offsets));
    }
    return top;
}


struct cypher_input_position input_position(yycontext *yy, unsigned int pos)
{
    assert(offsets_size(&(yy->line_start_offsets)) > 0);

    unsigned int depth = offsets_size(&(yy->line_start_offsets));
    unsigned int line_start_pos;
    do
    {
        line_start_pos = offsets_get(&(yy->line_start_offsets), depth-1);
    } while (line_start_pos > pos && depth > 0 && depth--);

    assert(depth > 0);
    assert(line_start_pos <= pos);

    struct cypher_input_position position =
        { .line = (depth-1) + yy->position_offset.line,
          .column = pos - line_start_pos +
              ((depth == 1)? yy->position_offset.column : 1),
          .offset = pos + yy->position_offset.offset };
    return position;
}


void block_start_action(yycontext *yy, char *text, int pos)
{
    assert(pos >= 0);
    struct cypher_input_position position = input_position(yy, pos);
    if (block_start(yy, pos, position) == NULL)
    {
        abort_parse(yy);
    }
    block_free(yy->prev_block);
    yy->prev_block = NULL;
}


struct block *block_start(yycontext *yy, size_t offset,
        struct cypher_input_position position)
{
    struct block *block = malloc(sizeof(struct block));
    if (block == NULL)
    {
        return NULL;
    }
    block->buffer_start = offset;
    block->buffer_end = offset;
    block->range.start = position;
    block->range.end = position;
    astnodes_init(&(block->sequence));
    astnodes_init(&(block->children));
    if (blocks_push(&(yy->blocks), block))
    {
        free(block);
        return NULL;
    }
    return block;
}


// close a block (and move to yy->prev_block)
void block_end_action(yycontext *yy, char *text, int pos)
{
    assert(pos >= 0);
    struct cypher_input_position position = input_position(yy, pos);
    struct block *block = block_end(yy, pos, position);
    assert(block != NULL);
    assert(yy->prev_block == NULL || astnodes_size(&(yy->prev_block->children)) == 0);
    block_free(yy->prev_block);
    yy->prev_block = block;
}


// close a block, and start a new block with the same starting input position
void block_replace_action(yycontext *yy, char *text, int pos)
{
    assert(pos >= 0);
    struct cypher_input_position position = input_position(yy, pos);
    struct block *block = block_end(yy, pos, position);
    assert(block != NULL);
    assert(yy->prev_block == NULL || astnodes_size(&(yy->prev_block->children)) == 0);
    block_free(yy->prev_block);
    yy->prev_block = block;
    if (block_start(yy, pos, block->range.start) == NULL)
    {
        abort_parse(yy);
    }
}


// close a block, but move all the children to the parent
void block_merge_action(yycontext *yy, char *text, int pos)
{
    assert(pos >= 0);
    struct cypher_input_position position = input_position(yy, pos);
    struct block *block = block_end(yy, pos, position);
    assert(block != NULL);
    assert(yy->prev_block == NULL || astnodes_size(&(yy->prev_block->children)) == 0);
    block_free(yy->prev_block);
    yy->prev_block = block;

    unsigned int nchildren = astnodes_size(&(block->children));
    if (nchildren > 0)
    {
        struct block *pblock = blocks_last(&(yy->blocks));
        assert(pblock != NULL);
        for (unsigned int i = 0; i < nchildren; ++i)
        {
            if (astnodes_push(&(pblock->children), astnodes_get(&(block->children), i)))
            {
                abort_parse(yy);
            }
        }
        astnodes_clear(&(block->children));
    }
}


struct block *block_end(yycontext *yy, size_t offset,
        struct cypher_input_position position)
{
    assert(blocks_size(&(yy->blocks)) > 0);
    struct block *block = blocks_pop(&(yy->blocks));
    assert(block != NULL);
    block->buffer_end = offset;
    block->range.end = position;
    assert(block->buffer_start <= block->buffer_end);
    assert(block->range.start.offset <= block->range.end.offset);
    return block;
}


void block_free(struct block *block)
{
    if (block == NULL)
    {
        return;
    }
    cypher_astnode_t *child;
    while ((child = astnodes_pop(&(block->children))) != NULL)
    {
        cypher_ast_free(child);
    }
    astnodes_cleanup(&(block->sequence));
    astnodes_cleanup(&(block->children));
    free(block);
}


void _err(yycontext *yy, const char *label)
{
    assert(yy->__pos >= 0);
    unsigned int pos = (unsigned int)yy->__pos;
    backtrack_lines(yy, pos);

    struct cypher_input_position position = input_position(yy, pos);
    char c = (yy->__pos < yy->__limit)? yy->__buf[pos] : '\0';
    if (cp_et_note_potential_error(&(yy->error_tracking), position, c, label))
    {
        abort_parse(yy);
    }
}


void record_error(yycontext *yy)
{
    if (cp_et_reify_potentials(&(yy->error_tracking)))
    {
        abort_parse(yy);
    }
}


void _strbuf_append(yycontext *yy, const char *s, size_t n)
{
    if (cp_sb_append(&(yy->string_buffer), s, n))
    {
        abort_parse(yy);
    }
}


void _strbuf_append_block(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "Block is only available immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    _strbuf_append(yy, s, n);
}


void _sequence_add(yycontext *yy, cypher_astnode_t *node)
{
    struct block *block = blocks_last(&(yy->blocks));
    assert(block != NULL);
    if (astnodes_push(&(block->sequence), node))
    {
        int errsv = errno;
        cypher_ast_free(node);
        errno = errsv;
        abort_parse(yy);
    }
}


void _op_push(yycontext *yy, const cypher_operator_t *op)
{
    if (operators_push(&(yy->operators), op))
    {
        abort_parse(yy);
    }
}


void _prec_push(yycontext *yy)
{
    assert(yy->op != NULL);
    unsigned int next_prec = (yy->op->associativity == LEFT_ASSOC)?
            yy->op->precedence + 1 : yy->op->precedence;
    if (precedences_push(&(yy->precedences), next_prec))
    {
        abort_parse(yy);
    }
}


void _prec_push_top(yycontext *yy)
{
    if (precedences_push(&(yy->precedences), 0))
    {
        abort_parse(yy);
    }
}


cypher_astnode_t *_statement(yycontext *yy, cypher_astnode_t *body)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_statement(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), body,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_cypher_option(yycontext *yy, cypher_astnode_t *version)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_cypher_option(version,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_cypher_option_param(yycontext *yy, cypher_astnode_t *name,
        cypher_astnode_t *value)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_cypher_option_param(name, value,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_explain_option(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_explain_option(range));
}


cypher_astnode_t *_profile_option(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_profile_option(range));
}


cypher_astnode_t *_create_index(yycontext *yy, cypher_astnode_t *label)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_create_node_props_index(label,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_create_pattern_index(yycontext *yy,
      cypher_astnode_t *identifier, cypher_astnode_t *label,
      enum cypher_ast_index_type index_type, cypher_astnode_t *options,
      bool is_relation)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_create_pattern_props_index(identifier,
            label, index_type, options, is_relation, astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_drop_index(yycontext *yy, cypher_astnode_t *label)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_drop_props_index(label,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}

static cypher_astnode_t *_drop_pattern_index(yycontext *yy,
         cypher_astnode_t *identifier, cypher_astnode_t *label,
         enum cypher_ast_index_type index_type, bool is_relation) {
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_drop_pattern_props_index(identifier,
            label,
            index_type,
            is_relation,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_create_node_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_create_node_prop_constraint(
            identifier, label, expression, unique,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_drop_node_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_drop_node_prop_constraint(
            identifier, label, expression, unique,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_create_rel_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_create_rel_prop_constraint(
            identifier, label, expression, unique,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_drop_rel_prop_constraint(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *label,
        cypher_astnode_t *expression, bool unique)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_drop_rel_prop_constraint(
            identifier, label, expression, unique,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_query(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");

    cypher_astnode_t **seq = astnodes_elements(&(yy->prev_block->sequence));
    unsigned int nseq = astnodes_size(&(yy->prev_block->sequence));

    unsigned int nopts = 0;
    while (nopts < nseq &&
            cypher_astnode_instanceof(seq[nopts], CYPHER_AST_QUERY_OPTION))
    {
        nopts++;
    }

    cypher_astnode_t *node = cypher_ast_query(
            seq, nopts, seq + nopts, nseq - nopts,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_using_periodic_commit(yycontext *yy, cypher_astnode_t *limit)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_using_periodic_commit(limit,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_load_csv(yycontext *yy, bool with_headers,
        cypher_astnode_t *url, cypher_astnode_t *identifier,
        cypher_astnode_t *field_terminator)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_load_csv(with_headers, url,
            identifier, field_terminator,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_start_clause(yycontext *yy, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_start(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), predicate,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_node_index_lookup(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *prop_name, cypher_astnode_t *lookup)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_node_index_lookup(identifier, index,
            prop_name, lookup, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_node_index_query(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *query)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_node_index_query(identifier, index,
            query, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_node_id_lookup(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_node_id_lookup(identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_all_nodes_scan(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_all_nodes_scan(identifier,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_rel_index_lookup(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *prop_name, cypher_astnode_t *lookup)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_rel_index_lookup(identifier, index,
            prop_name, lookup, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_rel_index_query(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *index,
        cypher_astnode_t *query)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_rel_index_query(identifier, index,
            query, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_rel_id_lookup(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_rel_id_lookup(identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_all_rels_scan(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_all_rels_scan(identifier,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_match_clause(yycontext *yy, bool optional,
        cypher_astnode_t *pattern, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_match(optional, pattern,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), predicate,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_using_index(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *label, cypher_astnode_t *prop_name)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_using_index(identifier, label,
            prop_name, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_using_join(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_using_join(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_using_scan(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *label)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_using_scan(identifier, label,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_merge_clause(yycontext *yy, cypher_astnode_t *pattern_part)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_merge(pattern_part,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_on_match(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_on_match(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_on_create(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_on_create(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_create_clause(yycontext *yy, bool unique,
        cypher_astnode_t *pattern)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_create(unique, pattern,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_set_clause(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_set(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_set_property(yycontext *yy, cypher_astnode_t *prop_name,
        cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_set_property(prop_name, expression,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_set_all_properties(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_set_all_properties(identifier,
            expression, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_merge_properties(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_merge_properties(identifier,
            expression, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_set_labels(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_set_labels(identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_delete(yycontext *yy, cypher_ast_delete_mode_t mode)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_delete(mode,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_remove_clause(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_remove(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_remove_property(yycontext *yy, cypher_astnode_t *prop_name)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_remove_property(prop_name,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_remove_labels(yycontext *yy, cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_remove_labels(identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_foreach_clause(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_foreach(identifier, expression,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_with_clause(yycontext *yy, bool distinct,
        bool include_existing, cypher_astnode_t *order_by,
        cypher_astnode_t *skip, cypher_astnode_t *limit,
        cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_with(distinct, include_existing,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            order_by, skip, limit, predicate,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_unwind_clause(yycontext *yy, cypher_astnode_t *expression,
        cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_unwind(expression, identifier,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_call_clause(yycontext *yy, cypher_astnode_t *proc_name,
        cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");

    cypher_astnode_t **seq = astnodes_elements(&(yy->prev_block->sequence));
    unsigned int nseq = astnodes_size(&(yy->prev_block->sequence));

    unsigned int nargs = 0;
    while (nargs < nseq &&
            !cypher_astnode_instanceof(seq[nargs], CYPHER_AST_PROJECTION))
    {
        nargs++;
    }

    cypher_astnode_t *node = cypher_ast_call(proc_name,
            seq, nargs, seq + nargs, nseq - nargs, predicate,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_call_subquery(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");

    cypher_astnode_t **seq = astnodes_elements(&(yy->prev_block->sequence));
    unsigned int nseq = astnodes_size(&(yy->prev_block->sequence));

    cypher_astnode_t *query = cypher_ast_query(
        NULL, 0, seq, nseq, astnodes_elements(&(yy->prev_block->children)),
        astnodes_size(&(yy->prev_block->children)), yy->prev_block->range);

    cypher_astnode_t *node = cypher_ast_call_subquery(
        query,
        cypher_astnode_range(query));
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_return_clause(yycontext *yy, bool distinct,
        bool include_existing, cypher_astnode_t *order_by,
        cypher_astnode_t *skip, cypher_astnode_t *limit)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_return(distinct, include_existing,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), order_by, skip, limit,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_projection(yycontext *yy, cypher_astnode_t *expression,
        cypher_astnode_t *alias)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_projection(expression, alias,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_order_by(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_order_by(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_sort_item(yycontext *yy, cypher_astnode_t *expression,
        bool ascending)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_sort_item(expression, ascending,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_union_clause(yycontext *yy, bool all)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_union(all,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_unary_operator(yycontext *yy, const cypher_operator_t *op,
        cypher_astnode_t *arg)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_unary_operator(op, arg,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_binary_operator(yycontext *yy, const cypher_operator_t *op,
        cypher_astnode_t *left, cypher_astnode_t *right)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_binary_operator(op, left, right,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_comparison_operator(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");

    unsigned int nargs = astnodes_size(&(yy->prev_block->sequence));
    assert(nargs > 1);
    unsigned int chain_length = nargs - 1;

    unsigned int ops_depth = operators_size(&(yy->operators));
    assert(ops_depth >= chain_length);
    const cypher_operator_t * const *ops = operators_elements(&(yy->operators))
        + (ops_depth - chain_length);

    cypher_astnode_t *node = cypher_ast_comparison(chain_length,
            ops, astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)), yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    operators_npop(&(yy->operators), chain_length);
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_apply_operator(yycontext *yy, cypher_astnode_t *left,
        bool distinct)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_apply_operator(left, distinct,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_apply_all_operator(yycontext *yy, cypher_astnode_t *left,
        bool distinct)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_apply_all_operator(left, distinct,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_subscript_operator(yycontext *yy,
        cypher_astnode_t *expression, cypher_astnode_t *subscript)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_subscript_operator(expression,
            subscript, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_property_operator(yycontext *yy, cypher_astnode_t *map,
        cypher_astnode_t *prop_name)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_property_operator(map, prop_name,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_slice_operator(yycontext *yy, cypher_astnode_t *expression,
        cypher_astnode_t *start, cypher_astnode_t* end)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_slice_operator(expression, start, end,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_projection(yycontext *yy, cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_map_projection(expression,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_projection_literal(yycontext *yy,
        cypher_astnode_t *prop_name, cypher_astnode_t *expression)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_map_projection_literal(
            prop_name, expression,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_projection_property(yycontext *yy,
        cypher_astnode_t *prop_name)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_map_projection_property(prop_name,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_projection_identifier(yycontext *yy,
        cypher_astnode_t *identifier)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_map_projection_identifier(identifier,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_projection_all_properties(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_map_projection_all_properties(
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_labels_operator(yycontext *yy, cypher_astnode_t *left)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_labels_operator(left,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_list_comprehension(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *expression,
        cypher_astnode_t *predicate, cypher_astnode_t *eval)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_list_comprehension(identifier,
            expression, predicate, eval,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_pattern_comprehension(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *pattern,
        cypher_astnode_t *predicate, cypher_astnode_t *eval)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_pattern_comprehension(identifier,
            pattern, predicate, eval,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_case_expression(yycontext *yy, cypher_astnode_t *expression,
        cypher_astnode_t *deflt)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_case(expression,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)) / 2, deflt,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_filter(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_filter(identifier, expression,
            predicate, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_extract(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *eval)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_extract(identifier, expression,
            eval, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_reduce(yycontext *yy, cypher_astnode_t *accumulator,
        cypher_astnode_t *init, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *eval)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_reduce(accumulator, init, identifier,
            expression, eval, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_all_predicate(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_all(identifier, expression,
            predicate, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_any_predicate(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_any(identifier, expression,
            predicate, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_single_predicate(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_single(identifier, expression,
            predicate, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_none_predicate(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *expression, cypher_astnode_t *predicate)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_none(identifier, expression,
            predicate, astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_collection_literal(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_collection(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_map_literal(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    assert(astnodes_size(&(yy->prev_block->sequence)) % 2 == 0);
    cypher_astnode_t *node = cypher_ast_pair_map(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)) / 2,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_strbuf_identifier(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_identifier(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_block_identifier(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    struct cypher_input_range range = yy->prev_block->range;
    for (; n > 0 && isspace(*s); ++s, --n)
        ;
    for (; n > 0 && isspace(s[n-1]); --n)
        ;
    return add_terminal(yy, cypher_ast_identifier(s, n, range));
}


cypher_astnode_t *_strbuf_parameter(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_parameter(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_integer(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_integer(cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_float(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_float(cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_true_literal(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_true(range));
}


cypher_astnode_t *_false_literal(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_false(range));
}


cypher_astnode_t *_null_literal(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_null(range));
}


cypher_astnode_t *_strbuf_label(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_label(cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_reltype(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_reltype(cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_prop_name(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_prop_name(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_function_name(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_function_name(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_index_name(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_index_name(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_strbuf_proc_name(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_proc_name(
                cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_pattern(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_pattern(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_named_path(yycontext *yy,
        cypher_astnode_t *identifier, cypher_astnode_t *path)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_named_path(identifier, path,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_shortest_path(yycontext *yy, bool single,
        cypher_astnode_t *path)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_shortest_path(single, path,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_pattern_path(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_pattern_path(
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_node_pattern(yycontext *yy, cypher_astnode_t *identifier,
        cypher_astnode_t *properties)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_node_pattern(identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), properties,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_rel_pattern(yycontext *yy,
        enum cypher_rel_direction direction, cypher_astnode_t *identifier,
        cypher_astnode_t *varlength, cypher_astnode_t *properties)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_rel_pattern(direction, identifier,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)), properties, varlength,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_range(yycontext *yy, cypher_astnode_t *start,
        cypher_astnode_t *end)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_range(start, end,
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_command(yycontext *yy, cypher_astnode_t *name)
{
    assert(cypher_astnode_instanceof(name, CYPHER_AST_STRING));

    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    cypher_astnode_t *node = cypher_ast_command(name,
            astnodes_elements(&(yy->prev_block->sequence)),
            astnodes_size(&(yy->prev_block->sequence)),
            astnodes_elements(&(yy->prev_block->children)),
            astnodes_size(&(yy->prev_block->children)),
            yy->prev_block->range);
    if (node == NULL)
    {
        abort_parse(yy);
    }
    astnodes_clear(&(yy->prev_block->sequence));
    astnodes_clear(&(yy->prev_block->children));
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *_string(yycontext *yy, const char *s, size_t n)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_string(s, n, range));
}


cypher_astnode_t *_block_string(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_string(s, n, range));
}


cypher_astnode_t *_strbuf_string(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_string(cp_sb_data(&(yy->string_buffer)),
                cp_sb_length(&(yy->string_buffer)), range));
}


cypher_astnode_t *_line_comment(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_line_comment(s, n, range));
}


cypher_astnode_t *_block_comment(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_block_comment(s, n, range));
}


cypher_astnode_t *_skip(yycontext *yy)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    char *s = yy->__buf + yy->prev_block->buffer_start;
    size_t n = yy->prev_block->buffer_end - yy->prev_block->buffer_start;
    struct cypher_input_range range = yy->prev_block->range;
    return add_terminal(yy, cypher_ast_error(s, n, range));
}


cypher_astnode_t *add_terminal(yycontext *yy, cypher_astnode_t *node)
{
    assert(yy->prev_block != NULL &&
            "An AST node can only be created immediately after a `>` in the grammar");
    assert(astnodes_size(&(yy->prev_block->children)) == 0 &&
            "terminal AST nodes should have no children created in the "
            "preceeding block");
    block_free(yy->prev_block);
    yy->prev_block = NULL;
    return add_child(yy, node);
}


cypher_astnode_t *add_child(yycontext *yy, cypher_astnode_t *node)
{
    if (node == NULL)
    {
        abort_parse(yy);
    }
    struct block *block = blocks_last(&(yy->blocks));
    assert(block != NULL);
    if (astnodes_push(&(block->children), node))
    {
        int errsv = errno;
        cypher_ast_free(node);
        errno = errsv;
        abort_parse(yy);
    }
    return node;
}

yyrule cypher_yyrule_from_flags(int flags) {
    if(flags & CYPHER_PARSE_ONLY_STATEMENTS) return yy_statement;
    if(flags & CYPHER_PARSE_ONLY_PARAMETERS) return yy_params;
    return yy_directive;
}
