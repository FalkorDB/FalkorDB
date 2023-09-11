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


START_TEST (parse_unwind)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("UNWIND [1,2,3] AS x RETURN x;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 29);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..29  statement           body=@1\n"
" @1   0..29  > query             clauses=[@2, @8]\n"
" @2   0..20  > > UNWIND          expression=@3, alias=@7\n"
" @3   7..14  > > > collection    [@4, @5, @6]\n"
" @4   8..9   > > > > integer     1\n"
" @5  10..11  > > > > integer     2\n"
" @6  12..13  > > > > integer     3\n"
" @7  18..19  > > > identifier    `x`\n"
" @8  20..28  > > RETURN          projections=[@9]\n"
" @9  27..28  > > > projection    expression=@10\n"
"@10  27..28  > > > > identifier  `x`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    const cypher_astnode_t *query = cypher_ast_statement_get_body(ast);
    const cypher_astnode_t *clause = cypher_ast_query_get_clause(query, 0);
    ck_assert_int_eq(cypher_astnode_type(clause), CYPHER_AST_UNWIND);

    const cypher_astnode_t *exp = cypher_ast_unwind_get_expression(clause);
    ck_assert_int_eq(cypher_astnode_type(exp), CYPHER_AST_COLLECTION);

    const cypher_astnode_t *id = cypher_ast_unwind_get_alias(clause);
    ck_assert_int_eq(cypher_astnode_type(id), CYPHER_AST_IDENTIFIER);
    ck_assert_str_eq(cypher_ast_identifier_get_name(id), "x");
}
END_TEST


TCase* unwind_tcase(void)
{
    TCase *tc = tcase_create("unwind");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_unwind);
    return tc;
}
