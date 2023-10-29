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


START_TEST (parse_simple_pattern_comprehension)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN [ (a)-->(b) | b.name ];",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 30);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..30  statement                      body=@1\n"
" @1   0..30  > query                        clauses=[@2]\n"
" @2   0..29  > > RETURN                     projections=[@3]\n"
" @3   7..29  > > > projection               expression=@4, alias=@14\n"
" @4   7..29  > > > > pattern comprehension  [@5 | @11]\n"
" @5   9..18  > > > > > pattern path         (@6)-[@8]-(@9)\n"
" @6   9..12  > > > > > > node pattern       (@7)\n"
" @7  10..11  > > > > > > > identifier       `a`\n"
" @8  12..15  > > > > > > rel pattern        -[]->\n"
" @9  15..18  > > > > > > node pattern       (@10)\n"
"@10  16..17  > > > > > > > identifier       `b`\n"
"@11  21..28  > > > > > property             @12.@13\n"
"@12  21..22  > > > > > > identifier         `b`\n"
"@13  23..27  > > > > > > prop name          `name`\n"
"@14   7..29  > > > > identifier             `[ (a)-->(b) | b.name ]`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_PATTERN_COMPREHENSION);

    ck_assert_ptr_eq(cypher_ast_pattern_comprehension_get_identifier(exp), NULL);

    const cypher_astnode_t *path = cypher_ast_pattern_comprehension_get_pattern(exp);
    ck_assert_int_eq(cypher_astnode_type(path), CYPHER_AST_PATTERN_PATH);
    ck_assert_int_eq(cypher_ast_pattern_path_nelements(path), 3);
    ck_assert_ptr_eq(cypher_ast_pattern_path_get_element(path, 3), NULL);

    const cypher_astnode_t *node = cypher_ast_pattern_path_get_element(path, 0);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
    const cypher_astnode_t *id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "a");

    node = cypher_ast_pattern_path_get_element(path, 1);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_REL_PATTERN);

    node = cypher_ast_pattern_path_get_element(path, 2);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_NODE_PATTERN);
    id = cypher_ast_node_pattern_get_identifier(node);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "b");

    ck_assert_ptr_eq(cypher_ast_pattern_comprehension_get_predicate(exp), NULL);

    const cypher_astnode_t *eval = cypher_ast_pattern_comprehension_get_eval(exp);
    ck_assert_int_eq(cypher_astnode_type(eval), CYPHER_AST_PROPERTY_OPERATOR);
    id = cypher_ast_property_operator_get_expression(eval);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "b");
    const cypher_astnode_t *name = cypher_ast_property_operator_get_prop_name(eval);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_PROP_NAME);
    ck_assert_str_eq(cypher_ast_prop_name_get_value(name), "name");
}
END_TEST


TCase* pattern_comprehension_tcase(void)
{
    TCase *tc = tcase_create("pattern comprehension");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_pattern_comprehension);
    return tc;
}
