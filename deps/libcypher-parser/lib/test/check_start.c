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
#include "../../lib/src/cypher-parser.h"
#include "memstream.h"
#include <check.h>
#include <errno.h>
#include <unistd.h>


static cypher_parse_result_t *result;
static char *memstream_buffer;
static size_t memstream_size;
static FILE *memstream;


static void setup(void)
{
    result = NULL;
    memstream = open_memstream(&memstream_buffer, &memstream_size);
    fputc('\n', memstream);
}


static void teardown(void)
{
    cypher_parse_result_free(result);
    fclose(memstream);
    free(memstream_buffer);
}


START_TEST (parse_node_index_lookup)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=node:index(foo = 'bar');",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 32);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..32  statement                body=@1\n"
"@1   0..32  > query                  clauses=[@2]\n"
"@2   0..31  > > START                points=[@3]\n"
"@3   6..31  > > > node index lookup  @4 = node:@5(@6 = @7)\n"
"@4   6..7   > > > > identifier       `n`\n"
"@5  13..18  > > > > index name       `index`\n"
"@6  19..22  > > > > prop name        `foo`\n"
"@7  25..30  > > > > string           \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *lookup = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(lookup), CYPHER_AST_NODE_INDEX_LOOKUP);

    const cypher_astnode_t *id =
            cypher_ast_node_index_lookup_get_identifier(lookup);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *index_name =
            cypher_ast_node_index_lookup_get_index_name(lookup);
    ck_assert_int_eq(cypher_astnode_type(index_name), CYPHER_AST_INDEX_NAME);
    ck_assert_str_eq(cypher_ast_index_name_get_value(index_name), "index");

    const cypher_astnode_t *prop_name =
            cypher_ast_node_index_lookup_get_prop_name(lookup);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "foo");

    const cypher_astnode_t *str =
            cypher_ast_node_index_lookup_get_lookup(lookup);
    ck_assert_int_eq(cypher_astnode_type(str), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(str), "bar");
}
END_TEST


START_TEST (parse_node_index_query)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=node:index('bar');",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 26);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..26  statement               body=@1\n"
"@1   0..26  > query                 clauses=[@2]\n"
"@2   0..25  > > START               points=[@3]\n"
"@3   6..25  > > > node index query  @4 = node:@5(@6)\n"
"@4   6..7   > > > > identifier      `n`\n"
"@5  13..18  > > > > index name      `index`\n"
"@6  19..24  > > > > string          \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *iquery = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(iquery), CYPHER_AST_NODE_INDEX_QUERY);

    const cypher_astnode_t *id =
            cypher_ast_node_index_query_get_identifier(iquery);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *index_name =
            cypher_ast_node_index_query_get_index_name(iquery);
    ck_assert_int_eq(cypher_astnode_type(index_name), CYPHER_AST_INDEX_NAME);
    ck_assert_str_eq(cypher_ast_index_name_get_value(index_name), "index");

    const cypher_astnode_t *str =
            cypher_ast_node_index_query_get_query(iquery);
    ck_assert_int_eq(cypher_astnode_type(str), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(str), "bar");
}
END_TEST


START_TEST (parse_node_id_lookup)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=node(65, 78, 3, 0) // find nodes\nRETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..50  statement             body=@1\n"
" @1   0..50  > query               clauses=[@2, @10]\n"
" @2   0..41  > > START             points=[@3]\n"
" @3   6..26  > > > node id lookup  @4 = node(@5, @6, @7, @8)\n"
" @4   6..7   > > > > identifier    `n`\n"
" @5  13..15  > > > > integer       65\n"
" @6  17..19  > > > > integer       78\n"
" @7  21..22  > > > > integer       3\n"
" @8  24..25  > > > > integer       0\n"
" @9  29..40  > > > line_comment    // find nodes\n"
"@10  41..49  > > RETURN            projections=[@11]\n"
"@11  48..49  > > > projection      expression=@12\n"
"@12  48..49  > > > > identifier    `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *lookup = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(lookup), CYPHER_AST_NODE_ID_LOOKUP);

    const cypher_astnode_t *identifier =
            cypher_ast_node_id_lookup_get_identifier(lookup);
    ck_assert_int_eq(cypher_astnode_type(identifier), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(identifier), "n");

    ck_assert_int_eq(cypher_ast_node_id_lookup_nids(lookup), 4);
    const cypher_astnode_t *id = cypher_ast_node_id_lookup_get_id(lookup, 0);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "65");

    id = cypher_ast_node_id_lookup_get_id(lookup, 1);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "78");

    id = cypher_ast_node_id_lookup_get_id(lookup, 2);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "3");

    id = cypher_ast_node_id_lookup_get_id(lookup, 3);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "0");

    ck_assert_ptr_eq(cypher_ast_node_id_lookup_get_id(lookup, 4), NULL);
}
END_TEST


START_TEST (parse_all_nodes_scan)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n = node(*)\nRETURN /* all nodes */ n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 43);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..43  statement             body=@1\n"
"@1   0..43  > query               clauses=[@2, @5]\n"
"@2   0..18  > > START             points=[@3]\n"
"@3   6..17  > > > all nodes scan  identifier=@4\n"
"@4   6..7   > > > > identifier    `n`\n"
"@5  18..42  > > RETURN            projections=[@7]\n"
"@6  27..38  > > > block_comment   /* all nodes */\n"
"@7  41..42  > > > projection      expression=@8\n"
"@8  41..42  > > > > identifier    `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *scan = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(scan), CYPHER_AST_ALL_NODES_SCAN);

    const cypher_astnode_t *identifier =
            cypher_ast_all_nodes_scan_get_identifier(scan);
    ck_assert_int_eq(cypher_astnode_type(identifier), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(identifier), "n");
}
END_TEST


START_TEST (parse_rel_index_lookup)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=rel:index(foo = 'bar');",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 31);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..31  statement               body=@1\n"
"@1   0..31  > query                 clauses=[@2]\n"
"@2   0..30  > > START               points=[@3]\n"
"@3   6..30  > > > rel index lookup  @4 = rel:@5(@6 = @7)\n"
"@4   6..7   > > > > identifier      `n`\n"
"@5  12..17  > > > > index name      `index`\n"
"@6  18..21  > > > > prop name       `foo`\n"
"@7  24..29  > > > > string          \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *lookup = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(lookup), CYPHER_AST_REL_INDEX_LOOKUP);

    const cypher_astnode_t *id =
            cypher_ast_rel_index_lookup_get_identifier(lookup);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *index_name =
            cypher_ast_rel_index_lookup_get_index_name(lookup);
    ck_assert_int_eq(cypher_astnode_type(index_name), CYPHER_AST_INDEX_NAME);
    ck_assert_str_eq(cypher_ast_index_name_get_value(index_name), "index");

    const cypher_astnode_t *prop_name =
            cypher_ast_rel_index_lookup_get_prop_name(lookup);
    ck_assert_int_eq(cypher_astnode_type(prop_name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(prop_name), "foo");

    const cypher_astnode_t *str =
            cypher_ast_rel_index_lookup_get_lookup(lookup);
    ck_assert_int_eq(cypher_astnode_type(str), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(str), "bar");
}
END_TEST


START_TEST (parse_rel_index_query)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=rel:index('bar');",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 25);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..25  statement              body=@1\n"
"@1   0..25  > query                clauses=[@2]\n"
"@2   0..24  > > START              points=[@3]\n"
"@3   6..24  > > > rel index query  @4 = rel:@5(@6)\n"
"@4   6..7   > > > > identifier     `n`\n"
"@5  12..17  > > > > index name     `index`\n"
"@6  18..23  > > > > string         \"bar\"\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *iquery = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(iquery), CYPHER_AST_REL_INDEX_QUERY);

    const cypher_astnode_t *id =
            cypher_ast_rel_index_query_get_identifier(iquery);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "n");

    const cypher_astnode_t *index_name =
            cypher_ast_rel_index_query_get_index_name(iquery);
    ck_assert_int_eq(cypher_astnode_type(index_name), CYPHER_AST_INDEX_NAME);
    ck_assert_str_eq(cypher_ast_index_name_get_value(index_name), "index");

    const cypher_astnode_t *str =
            cypher_ast_rel_index_query_get_query(iquery);
    ck_assert_int_eq(cypher_astnode_type(str), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(str), "bar");
}
END_TEST


START_TEST (parse_rel_id_lookup)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n=rel(65, 78, 3, 0) // find nodes\nRETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 49);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..49  statement            body=@1\n"
" @1   0..49  > query              clauses=[@2, @10]\n"
" @2   0..40  > > START            points=[@3]\n"
" @3   6..25  > > > rel id lookup  @4 = rel(@5, @6, @7, @8)\n"
" @4   6..7   > > > > identifier   `n`\n"
" @5  12..14  > > > > integer      65\n"
" @6  16..18  > > > > integer      78\n"
" @7  20..21  > > > > integer      3\n"
" @8  23..24  > > > > integer      0\n"
" @9  28..39  > > > line_comment   // find nodes\n"
"@10  40..48  > > RETURN           projections=[@11]\n"
"@11  47..48  > > > projection     expression=@12\n"
"@12  47..48  > > > > identifier   `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *lookup = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(lookup), CYPHER_AST_REL_ID_LOOKUP);

    const cypher_astnode_t *identifier =
            cypher_ast_rel_id_lookup_get_identifier(lookup);
    ck_assert_int_eq(cypher_astnode_type(identifier), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(identifier), "n");

    ck_assert_int_eq(cypher_ast_rel_id_lookup_nids(lookup), 4);
    const cypher_astnode_t *id = cypher_ast_rel_id_lookup_get_id(lookup, 0);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "65");

    id = cypher_ast_rel_id_lookup_get_id(lookup, 1);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "78");

    id = cypher_ast_rel_id_lookup_get_id(lookup, 2);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "3");

    id = cypher_ast_rel_id_lookup_get_id(lookup, 3);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(id), "0");

    ck_assert_ptr_eq(cypher_ast_rel_id_lookup_get_id(lookup, 4), NULL);
}
END_TEST


START_TEST (parse_all_rels_scan)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n = rel(*)\nRETURN /* all rels */ n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 41);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..41  statement            body=@1\n"
"@1   0..41  > query              clauses=[@2, @5]\n"
"@2   0..17  > > START            points=[@3]\n"
"@3   6..16  > > > all rels scan  identifier=@4\n"
"@4   6..7   > > > > identifier   `n`\n"
"@5  17..40  > > RETURN           projections=[@7]\n"
"@6  26..36  > > > block_comment  /* all rels */\n"
"@7  39..40  > > > projection     expression=@8\n"
"@8  39..40  > > > > identifier   `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    ck_assert_ptr_eq(cypher_ast_start_get_predicate(start), NULL);

    ck_assert_int_eq(cypher_ast_start_npoints(start), 1);
    const cypher_astnode_t *scan = cypher_ast_start_get_point(start, 0);
    ck_assert_int_eq(cypher_astnode_type(scan), CYPHER_AST_ALL_RELS_SCAN);

    const cypher_astnode_t *identifier =
            cypher_ast_all_rels_scan_get_identifier(scan);
    ck_assert_int_eq(cypher_astnode_type(identifier), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(identifier), "n");
}
END_TEST


START_TEST (parse_start_with_predicate)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "START n = node(*) /* predicate */ WHERE n.foo > 1 RETURN n;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 59);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..59  statement             body=@1\n"
" @1   0..59  > query               clauses=[@2, @11]\n"
" @2   0..50  > > START             points=[@3], WHERE=@6\n"
" @3   6..17  > > > all nodes scan  identifier=@4\n"
" @4   6..7   > > > > identifier    `n`\n"
" @5  20..31  > > > block_comment   /* predicate */\n"
" @6  40..50  > > > comparison      @7 > @10\n"
" @7  40..46  > > > > property      @8.@9\n"
" @8  40..41  > > > > > identifier  `n`\n"
" @9  42..45  > > > > > prop name   `foo`\n"
"@10  48..49  > > > > integer       1\n"
"@11  50..58  > > RETURN            projections=[@12]\n"
"@12  57..58  > > > projection      expression=@13\n"
"@13  57..58  > > > > identifier    `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *start = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(start), CYPHER_AST_START);

    const cypher_astnode_t *pred = cypher_ast_start_get_predicate(start);
    ck_assert(cypher_astnode_instanceof(pred, CYPHER_AST_EXPRESSION));
    ck_assert_int_eq(cypher_astnode_type(pred), CYPHER_AST_COMPARISON);

    ck_assert_int_eq(cypher_ast_comparison_get_length(pred), 1);

    const cypher_operator_t *o = cypher_ast_comparison_get_operator(pred, 0);
    ck_assert_ptr_eq(o, CYPHER_OP_GT);

    const cypher_astnode_t *l = cypher_ast_comparison_get_argument(pred, 0);
    ck_assert_int_eq(cypher_astnode_type(l), CYPHER_AST_PROPERTY_OPERATOR);

    const cypher_astnode_t *r = cypher_ast_comparison_get_argument(pred, 1);
    ck_assert_int_eq(cypher_astnode_type(r), CYPHER_AST_INTEGER);
}
END_TEST


TCase* start_tcase(void)
{
    TCase *tc = tcase_create("start");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_node_index_lookup);
    tcase_add_test(tc, parse_node_index_query);
    tcase_add_test(tc, parse_node_id_lookup);
    tcase_add_test(tc, parse_all_nodes_scan);
    tcase_add_test(tc, parse_rel_index_lookup);
    tcase_add_test(tc, parse_rel_index_query);
    tcase_add_test(tc, parse_rel_id_lookup);
    tcase_add_test(tc, parse_all_rels_scan);
    tcase_add_test(tc, parse_start_with_predicate);
    return tc;
}
