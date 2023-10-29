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


START_TEST (parse_map_project_none)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN a{} AS x", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 15);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..15  statement               body=@1\n"
"@1   0..15  > query                 clauses=[@2]\n"
"@2   0..15  > > RETURN              projections=[@3]\n"
"@3   7..15  > > > projection        expression=@4, alias=@6\n"
"@4   7..11  > > > > map projection  @5{}\n"
"@5   7..8   > > > > > identifier    `a`\n"
"@6  14..15  > > > > identifier      `x`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 1), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);

    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_MAP_PROJECTION);

    const cypher_astnode_t *id = cypher_ast_map_projection_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "a");

    ck_assert_int_eq(cypher_ast_map_projection_nselectors(exp), 0);
    ck_assert_ptr_eq(cypher_ast_map_projection_get_selector(exp, 0), NULL);
}
END_TEST


START_TEST (parse_map_project_multiple_selectors)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN map{x: 1, .y, z, .*}", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
//    ck_assert_int_eq(last.offset, 15);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..27  statement                            body=@1\n"
" @1   0..27  > query                              clauses=[@2]\n"
" @2   0..27  > > RETURN                           projections=[@3]\n"
" @3   7..27  > > > projection                     expression=@4, alias=@14\n"
" @4   7..27  > > > > map projection               @5{@6, @9, @11, @13}\n"
" @5   7..10  > > > > > identifier                 `map`\n"
" @6  11..15  > > > > > literal projection         @7: @8\n"
" @7  11..12  > > > > > > prop name                `x`\n"
" @8  14..15  > > > > > > integer                  1\n"
" @9  17..19  > > > > > property projection        .@10\n"
"@10  18..19  > > > > > > prop name                `y`\n"
"@11  21..22  > > > > > identifier projection      @12\n"
"@12  21..22  > > > > > > identifier               `z`\n"
"@13  24..26  > > > > > all properties projection  .*\n"
"@14   7..27  > > > > identifier                   `map{x: 1, .y, z, .*}`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);
    ck_assert_ptr_eq(cypher_ast_return_get_projection(clause, 1), NULL);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);

    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_MAP_PROJECTION);

    const cypher_astnode_t *id = cypher_ast_map_projection_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "map");

    ck_assert_int_eq(cypher_ast_map_projection_nselectors(exp), 4);
    ck_assert_ptr_eq(cypher_ast_map_projection_get_selector(exp, 4), NULL);

    const cypher_astnode_t *sel = cypher_ast_map_projection_get_selector(exp, 0);
    ck_assert_int_eq(cypher_astnode_type(sel), CYPHER_AST_MAP_PROJECTION_LITERAL);
    const cypher_astnode_t *pname = cypher_ast_map_projection_literal_get_prop_name(sel);
    ck_assert_int_eq(cypher_astnode_type(pname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(pname), "x");
    const cypher_astnode_t *lexp = cypher_ast_map_projection_literal_get_expression(sel);
    ck_assert_int_eq(cypher_astnode_type(lexp), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(lexp), "1");

    sel = cypher_ast_map_projection_get_selector(exp, 1);
    ck_assert_int_eq(cypher_astnode_type(sel), CYPHER_AST_MAP_PROJECTION_PROPERTY);
    pname = cypher_ast_map_projection_property_get_prop_name(sel);
    ck_assert_int_eq(cypher_astnode_type(pname), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(pname), "y");

    sel = cypher_ast_map_projection_get_selector(exp, 2);
    ck_assert_int_eq(cypher_astnode_type(sel), CYPHER_AST_MAP_PROJECTION_IDENTIFIER);
    id = cypher_ast_map_projection_identifier_get_identifier(sel);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "z");

    sel = cypher_ast_map_projection_get_selector(exp, 3);
    ck_assert_int_eq(cypher_astnode_type(sel), CYPHER_AST_MAP_PROJECTION_ALL_PROPERTIES);
}
END_TEST


TCase* map_projection_tcase(void)
{
    TCase *tc = tcase_create("map projection");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_map_project_none);
    tcase_add_test(tc, parse_map_project_multiple_selectors);
    return tc;
}
