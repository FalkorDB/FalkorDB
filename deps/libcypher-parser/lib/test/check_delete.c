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


START_TEST (parse_delete)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/*MATCH*/ DELETE n, CASE WHEN exists(n.foo) THEN m ELSE n END;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 62);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment            /*MATCH*/\n"
" @1  10..62  statement                body=@2\n"
" @2  10..62  > query                  clauses=[@3]\n"
" @3  10..61  > > DELETE               expressions=[@4, @5]\n"
" @4  17..18  > > > identifier         `n`\n"
" @5  20..61  > > > case               alternatives=[(@6:@11)], default=@12\n"
" @6  30..43  > > > > apply            @7(@8)\n"
" @7  30..36  > > > > > function name  `exists`\n"
" @8  37..42  > > > > > property       @9.@10\n"
" @9  37..38  > > > > > > identifier   `n`\n"
"@10  39..42  > > > > > > prop name    `foo`\n"
"@11  49..50  > > > > identifier       `m`\n"
"@12  56..57  > > > > identifier       `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *del = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(del), CYPHER_AST_DELETE);

    ck_assert(!cypher_ast_delete_has_detach(del));

    ck_assert_int_eq(cypher_ast_delete_nexpressions(del), 2);
    ck_assert_ptr_eq(cypher_ast_delete_get_expression(del, 2), NULL);

    const cypher_astnode_t *expr = cypher_ast_delete_get_expression(del, 0);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "n");

    expr = cypher_ast_delete_get_expression(del, 1);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_CASE);
}
END_TEST


START_TEST (parse_detach_delete)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/*MATCH*/ DETACH DELETE n, CASE WHEN exists(n.foo) THEN m ELSE n END;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 69);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment            /*MATCH*/\n"
" @1  10..69  statement                body=@2\n"
" @2  10..69  > query                  clauses=[@3]\n"
" @3  10..68  > > DELETE               DETACH, expressions=[@4, @5]\n"
" @4  24..25  > > > identifier         `n`\n"
" @5  27..68  > > > case               alternatives=[(@6:@11)], default=@12\n"
" @6  37..50  > > > > apply            @7(@8)\n"
" @7  37..43  > > > > > function name  `exists`\n"
" @8  44..49  > > > > > property       @9.@10\n"
" @9  44..45  > > > > > > identifier   `n`\n"
"@10  46..49  > > > > > > prop name    `foo`\n"
"@11  56..57  > > > > identifier       `m`\n"
"@12  63..64  > > > > identifier       `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *del = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(del), CYPHER_AST_DELETE);

    ck_assert(cypher_ast_delete_has_detach(del));

    ck_assert_int_eq(cypher_ast_delete_nexpressions(del), 2);
    ck_assert_ptr_eq(cypher_ast_delete_get_expression(del, 2), NULL);

    const cypher_astnode_t *expr = cypher_ast_delete_get_expression(del, 0);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(expr), "n");

    expr = cypher_ast_delete_get_expression(del, 1);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_CASE);
}
END_TEST


TCase* delete_tcase(void)
{
    TCase *tc = tcase_create("delete");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_delete);
    tcase_add_test(tc, parse_detach_delete);
    return tc;
}
