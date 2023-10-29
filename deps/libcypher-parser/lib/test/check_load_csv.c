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


START_TEST (parse_load_csv)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "LOAD CSV FROM 'file:///movies.csv' AS m RETURN m;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 49);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..49  statement           body=@1\n"
"@1   0..49  > query             clauses=[@2, @5]\n"
"@2   0..40  > > LOAD CSV        url=@3, identifier=@4\n"
"@3  14..34  > > > string        \"file:///movies.csv\"\n"
"@4  38..39  > > > identifier    `m`\n"
"@5  40..48  > > RETURN          projections=[@6]\n"
"@6  47..48  > > > projection    expression=@7\n"
"@7  47..48  > > > > identifier  `m`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 2);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_LOAD_CSV);

    ck_assert(!cypher_ast_load_csv_has_with_headers(clause));

    const cypher_astnode_t *url = cypher_ast_load_csv_get_url(clause);
    ck_assert_int_eq(cypher_astnode_type(url), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(url), "file:///movies.csv");

    const cypher_astnode_t *id = cypher_ast_load_csv_get_identifier(clause);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    const cypher_astnode_t *field_terminator =
            cypher_ast_load_csv_get_field_terminator(clause);
    ck_assert_ptr_eq(field_terminator, NULL);
}
END_TEST


START_TEST (parse_load_csv_with_headers)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "LOAD CSV WITH HEADERS FROM {source} AS m RETURN m;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 50);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..50  statement           body=@1\n"
"@1   0..50  > query             clauses=[@2, @5]\n"
"@2   0..41  > > LOAD CSV        WITH HEADERS, url=@3, identifier=@4\n"
"@3  27..35  > > > parameter     $`source`\n"
"@4  39..40  > > > identifier    `m`\n"
"@5  41..49  > > RETURN          projections=[@6]\n"
"@6  48..49  > > > projection    expression=@7\n"
"@7  48..49  > > > > identifier  `m`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 2);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_LOAD_CSV);

    ck_assert(cypher_ast_load_csv_has_with_headers(clause));

    const cypher_astnode_t *url = cypher_ast_load_csv_get_url(clause);
    ck_assert_int_eq(cypher_astnode_type(url), CYPHER_AST_PARAMETER);
    ck_assert_str_eq(cypher_ast_parameter_get_name(url), "source");

    const cypher_astnode_t *id = cypher_ast_load_csv_get_identifier(clause);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    const cypher_astnode_t *field_terminator =
            cypher_ast_load_csv_get_field_terminator(clause);
    ck_assert_ptr_eq(field_terminator, NULL);
}
END_TEST


START_TEST (parse_load_csv_with_field_terminator)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse(
            "LOAD CSV FROM {source} AS m FIELDTERMINATOR '|' RETURN m;",
            &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 57);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..57  statement           body=@1\n"
"@1   0..57  > query             clauses=[@2, @6]\n"
"@2   0..48  > > LOAD CSV        url=@3, identifier=@4, field_terminator=@5\n"
"@3  14..22  > > > parameter     $`source`\n"
"@4  26..27  > > > identifier    `m`\n"
"@5  44..47  > > > string        \"|\"\n"
"@6  48..56  > > RETURN          projections=[@7]\n"
"@7  55..56  > > > projection    expression=@8\n"
"@8  55..56  > > > > identifier  `m`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(query), CYPHER_AST_QUERY);

    ck_assert_int_eq(cypher_ast_query_nclauses(query), 2);

    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_LOAD_CSV);

    ck_assert(!cypher_ast_load_csv_has_with_headers(clause));

    const cypher_astnode_t *url = cypher_ast_load_csv_get_url(clause);
    ck_assert_int_eq(cypher_astnode_type(url), CYPHER_AST_PARAMETER);
    ck_assert_str_eq(cypher_ast_parameter_get_name(url), "source");

    const cypher_astnode_t *id = cypher_ast_load_csv_get_identifier(clause);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "m");

    const cypher_astnode_t *field_terminator =
            cypher_ast_load_csv_get_field_terminator(clause);
    ck_assert_int_eq(cypher_astnode_type(field_terminator), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(field_terminator), "|");
}
END_TEST


TCase* load_csv_tcase(void)
{
    TCase *tc = tcase_create("load_csv");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_load_csv);
    tcase_add_test(tc, parse_load_csv_with_headers);
    tcase_add_test(tc, parse_load_csv_with_field_terminator);
    return tc;
}
