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


START_TEST (parse_foreach)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "/*MATCH*/ FOREACH (x IN [1,2,3] | SET n.foo = x REMOVE n.bar)",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 61);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   2..7   block_comment            /*MATCH*/\n"
" @1  10..61  statement                body=@2\n"
" @2  10..61  > query                  clauses=[@3]\n"
" @3  10..61  > > FOREACH              [@4 IN @5 | @9, @15]\n"
" @4  19..20  > > > identifier         `x`\n"
" @5  24..31  > > > collection         [@6, @7, @8]\n"
" @6  25..26  > > > > integer          1\n"
" @7  27..28  > > > > integer          2\n"
" @8  29..30  > > > > integer          3\n"
" @9  34..48  > > > SET                items=[@10]\n"
"@10  38..48  > > > > set property     @11 = @14\n"
"@11  38..44  > > > > > property       @12.@13\n"
"@12  38..39  > > > > > > identifier   `n`\n"
"@13  40..43  > > > > > > prop name    `foo`\n"
"@14  46..47  > > > > > identifier     `x`\n"
"@15  48..60  > > > REMOVE             items=[@16]\n"
"@16  55..60  > > > > remove property  prop=@17\n"
"@17  55..60  > > > > > property       @18.@19\n"
"@18  55..56  > > > > > > identifier   `n`\n"
"@19  57..60  > > > > > > prop name    `bar`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *foreach = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(foreach), CYPHER_AST_FOREACH);

    const cypher_astnode_t *id = cypher_ast_foreach_get_identifier(foreach);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");

    const cypher_astnode_t *expr = cypher_ast_foreach_get_expression(foreach);
    ck_assert_int_eq(cypher_astnode_type(expr), CYPHER_AST_COLLECTION);

    ck_assert_int_eq(cypher_ast_foreach_nclauses(foreach), 2);
    ck_assert_ptr_eq(cypher_ast_foreach_get_clause(foreach, 2), NULL);

    const cypher_astnode_t *clause = cypher_ast_foreach_get_clause(foreach, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_SET);

    clause = cypher_ast_foreach_get_clause(foreach, 1);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_REMOVE);
}
END_TEST


TCase* foreach_tcase(void)
{
    TCase *tc = tcase_create("foreach");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_foreach);
    return tc;
}
