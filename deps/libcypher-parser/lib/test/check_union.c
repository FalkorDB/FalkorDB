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


START_TEST (parse_union)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN 1 UNION RETURN 2;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 24);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..24  statement           body=@1\n"
" @1   0..24  > query             clauses=[@2, @6, @7]\n"
" @2   0..9   > > RETURN          projections=[@3]\n"
" @3   7..9   > > > projection    expression=@4, alias=@5\n"
" @4   7..8   > > > > integer     1\n"
" @5   7..9   > > > > identifier  `1`\n"
" @6   9..15  > > UNION\n"
" @7  15..23  > > RETURN          projections=[@8]\n"
" @8  22..23  > > > projection    expression=@9, alias=@10\n"
" @9  22..23  > > > > integer     2\n"
"@10  22..23  > > > > identifier  `2`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_ast_query_nclauses(query), 3);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    clause = cypher_ast_query_get_clause(query, 1);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_UNION);
    ck_assert(!cypher_ast_union_has_all(clause));

    clause = cypher_ast_query_get_clause(query, 2);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
}
END_TEST


START_TEST (parse_union_all)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "MATCH (x:Foo) RETURN x UNION ALL MATCH (x:Bar) RETURN x;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 56);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..56  statement               body=@1\n"
" @1   0..56  > query                 clauses=[@2, @8, @11, @12, @18]\n"
" @2   0..14  > > MATCH               pattern=@3\n"
" @3   6..13  > > > pattern           paths=[@4]\n"
" @4   6..13  > > > > pattern path    (@5)\n"
" @5   6..13  > > > > > node pattern  (@6:@7)\n"
" @6   7..8   > > > > > > identifier  `x`\n"
" @7   8..12  > > > > > > label       :`Foo`\n"
" @8  14..23  > > RETURN              projections=[@9]\n"
" @9  21..23  > > > projection        expression=@10\n"
"@10  21..22  > > > > identifier      `x`\n"
"@11  23..33  > > UNION               ALL\n"
"@12  33..47  > > MATCH               pattern=@13\n"
"@13  39..46  > > > pattern           paths=[@14]\n"
"@14  39..46  > > > > pattern path    (@15)\n"
"@15  39..46  > > > > > node pattern  (@16:@17)\n"
"@16  40..41  > > > > > > identifier  `x`\n"
"@17  41..45  > > > > > > label       :`Bar`\n"
"@18  47..55  > > RETURN              projections=[@19]\n"
"@19  54..55  > > > projection        expression=@20\n"
"@20  54..55  > > > > identifier      `x`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_ast_query_nclauses(query), 5);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_MATCH);
    clause = cypher_ast_query_get_clause(query, 1);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);

    clause = cypher_ast_query_get_clause(query, 2);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_UNION);
    ck_assert(cypher_ast_union_has_all(clause));

    clause = cypher_ast_query_get_clause(query, 3);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_MATCH);
    clause = cypher_ast_query_get_clause(query, 4);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
}
END_TEST


TCase* union_tcase(void)
{
    TCase *tc = tcase_create("union");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_union);
    tcase_add_test(tc, parse_union_all);
    return tc;
}
