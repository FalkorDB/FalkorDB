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


START_TEST (parse_reduce)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN reduce(a = 0, b in list | a + b);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 40);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..40  statement                  body=@1\n"
" @1   0..40  > query                    clauses=[@2]\n"
" @2   0..39  > > RETURN                 projections=[@3]\n"
" @3   7..39  > > > projection           expression=@4, alias=@12\n"
" @4   7..39  > > > > reduce             [@5=@6, @7 IN @8 | @9]\n"
" @5  14..15  > > > > > identifier       `a`\n"
" @6  18..19  > > > > > integer          0\n"
" @7  21..22  > > > > > identifier       `b`\n"
" @8  26..30  > > > > > identifier       `list`\n"
" @9  33..38  > > > > > binary operator  @10 + @11\n"
"@10  33..34  > > > > > > identifier     `a`\n"
"@11  37..38  > > > > > > identifier     `b`\n"
"@12   7..39  > > > > identifier         `reduce(a = 0, b in list | a + b)`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_int_eq(cypher_ast_return_nprojections(clause), 1);

    const cypher_astnode_t *proj = cypher_ast_return_get_projection(clause, 0);
    ck_assert_int_eq(cypher_astnode_type(proj), CYPHER_AST_PROJECTION);
    const cypher_astnode_t *exp = cypher_ast_projection_get_expression(proj);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_REDUCE);

    const cypher_astnode_t *id = cypher_ast_reduce_get_accumulator(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "a");

    const cypher_astnode_t *node = cypher_ast_reduce_get_init(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(node), "0");

    id = cypher_ast_reduce_get_identifier(exp);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "b");

    node = cypher_ast_reduce_get_expression(exp);
    ck_assert_int_eq(cypher_astnode_type(node), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(node), "list");

    const cypher_astnode_t *eval = cypher_ast_reduce_get_eval(exp);
    ck_assert_int_eq(cypher_astnode_type(eval), CYPHER_AST_BINARY_OPERATOR);
}
END_TEST


TCase* reduce_tcase(void)
{
    TCase *tc = tcase_create("reduce");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_reduce);
    return tc;
}
