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


START_TEST (parse_query_with_no_options)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("MATCH (n) RETURN n;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 19);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..19  statement               body=@1\n"
"@1   0..19  > query                 clauses=[@2, @7]\n"
"@2   0..10  > > MATCH               pattern=@3\n"
"@3   6..9   > > > pattern           paths=[@4]\n"
"@4   6..9   > > > > pattern path    (@5)\n"
"@5   6..9   > > > > > node pattern  (@6)\n"
"@6   7..8   > > > > > > identifier  `n`\n"
"@7  10..18  > > RETURN              projections=[@8]\n"
"@8  17..18  > > > projection        expression=@9\n"
"@9  17..18  > > > > identifier      `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 19);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    ck_assert_int_eq(cypher_astnode_range(query).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(query).end.offset, 19);

    ck_assert_int_eq(cypher_ast_query_noptions(query), 0);
    ck_assert_ptr_eq(cypher_ast_query_get_option(query, 0), NULL);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 2);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_MATCH);
    clause = cypher_ast_query_get_clause(query, 1);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_RETURN);
    ck_assert_ptr_eq(cypher_ast_query_get_clause(query, 2), NULL);
}
END_TEST


START_TEST (parse_query_with_periodic_commit_option)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("USING PERIODIC COMMIT 500 CREATE (n);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 37);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..37  statement                  body=@1\n"
"@1   0..37  > query                    clauses=[@4]\n"
"@2   0..26  > > USING PERIODIC_COMMIT  limit=@3\n"
"@3  22..25  > > > integer              500\n"
"@4  26..36  > > CREATE                 pattern=@5\n"
"@5  33..36  > > > pattern              paths=[@6]\n"
"@6  33..36  > > > > pattern path       (@7)\n"
"@7  33..36  > > > > > node pattern     (@8)\n"
"@8  34..35  > > > > > > identifier     `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 37);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    ck_assert_int_eq(cypher_astnode_range(query).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(query).end.offset, 37);

    ck_assert_int_eq(cypher_ast_query_noptions(query), 1);
    const cypher_astnode_t *option = cypher_ast_query_get_option(query, 0);
    ck_assert_int_eq(cypher_astnode_type(option),
            CYPHER_AST_USING_PERIODIC_COMMIT);

    const cypher_astnode_t *limit =
        cypher_ast_using_periodic_commit_get_limit(option);
    ck_assert_int_eq(cypher_astnode_type(limit), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(limit), "500");

    ck_assert_ptr_eq(cypher_ast_query_get_option(query, 1), NULL);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 1);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CREATE);
    ck_assert_ptr_eq(cypher_ast_query_get_clause(query, 1), NULL);
}
END_TEST


START_TEST (parse_query_with_periodic_commit_option_with_no_limit)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("USING PERIODIC COMMIT CREATE (n);",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 33);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..33  statement                  body=@1\n"
"@1   0..33  > query                    clauses=[@3]\n"
"@2   0..22  > > USING PERIODIC_COMMIT\n"
"@3  22..32  > > CREATE                 pattern=@4\n"
"@4  29..32  > > > pattern              paths=[@5]\n"
"@5  29..32  > > > > pattern path       (@6)\n"
"@6  29..32  > > > > > node pattern     (@7)\n"
"@7  30..31  > > > > > > identifier     `n`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 33);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);
    ck_assert_int_eq(cypher_astnode_range(query).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(query).end.offset, 33);

    ck_assert_int_eq(cypher_ast_query_noptions(query), 1);
    const cypher_astnode_t *option = cypher_ast_query_get_option(query, 0);
    ck_assert_int_eq(cypher_astnode_type(option),
            CYPHER_AST_USING_PERIODIC_COMMIT);
    ck_assert_ptr_eq(cypher_ast_using_periodic_commit_get_limit(option), NULL);
    ck_assert_ptr_eq(cypher_ast_query_get_option(query, 1), NULL);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 1);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_CREATE);
    ck_assert_ptr_eq(cypher_ast_query_get_clause(query, 1), NULL);
}
END_TEST


TCase* query_tcase(void)
{
    TCase *tc = tcase_create("query");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_query_with_no_options);
    tcase_add_test(tc, parse_query_with_periodic_commit_option);
    tcase_add_test(tc, parse_query_with_periodic_commit_option_with_no_limit);
    return tc;
}
