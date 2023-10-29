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


START_TEST (parse_simple_create)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("CREATE (n)-[:KNOWS]->(f);", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 25);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..25  statement               body=@1\n"
" @1   0..25  > query                 clauses=[@2]\n"
" @2   0..24  > > CREATE              pattern=@3\n"
" @3   7..24  > > > pattern           paths=[@4]\n"
" @4   7..24  > > > > pattern path    (@5)-[@7]-(@9)\n"
" @5   7..10  > > > > > node pattern  (@6)\n"
" @6   8..9   > > > > > > identifier  `n`\n"
" @7  10..21  > > > > > rel pattern   -[:@8]->\n"
" @8  12..18  > > > > > > rel type    :`KNOWS`\n"
" @9  21..24  > > > > > node pattern  (@10)\n"
"@10  22..23  > > > > > > identifier  `f`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *create = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(create), CYPHER_AST_CREATE);

    ck_assert(!cypher_ast_create_is_unique(create));

    const cypher_astnode_t *pattern = cypher_ast_create_get_pattern(create);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
}
END_TEST


START_TEST (parse_create_unique)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("CREATE UNIQUE (n)-[:KNOWS]->(f);", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 32);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..32  statement               body=@1\n"
" @1   0..32  > query                 clauses=[@2]\n"
" @2   0..31  > > CREATE              UNIQUE, pattern=@3\n"
" @3  14..31  > > > pattern           paths=[@4]\n"
" @4  14..31  > > > > pattern path    (@5)-[@7]-(@9)\n"
" @5  14..17  > > > > > node pattern  (@6)\n"
" @6  15..16  > > > > > > identifier  `n`\n"
" @7  17..28  > > > > > rel pattern   -[:@8]->\n"
" @8  19..25  > > > > > > rel type    :`KNOWS`\n"
" @9  28..31  > > > > > node pattern  (@10)\n"
"@10  29..30  > > > > > > identifier  `f`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    const cypher_astnode_t *create = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(create), CYPHER_AST_CREATE);

    ck_assert(cypher_ast_create_is_unique(create));

    const cypher_astnode_t *pattern = cypher_ast_create_get_pattern(create);
    ck_assert_int_eq(cypher_astnode_type(pattern), CYPHER_AST_PATTERN);
    ck_assert_int_eq(cypher_ast_pattern_npaths(pattern), 1);
}
END_TEST


TCase* create_tcase(void)
{
    TCase *tc = tcase_create("create");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_simple_create);
    tcase_add_test(tc, parse_create_unique);
    return tc;
}
