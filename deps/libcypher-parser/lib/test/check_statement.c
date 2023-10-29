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


START_TEST (parse_statement_with_no_options)
{
    struct cypher_input_position last = cypher_input_position_zero;
    result = cypher_parse("RETURN 1;", &last, NULL, 0);
    ck_assert_ptr_ne(result, NULL);
    ck_assert_int_eq(last.offset, 9);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..9  statement           body=@1\n"
"@1  0..9  > query             clauses=[@2]\n"
"@2  0..8  > > RETURN          projections=[@3]\n"
"@3  7..8  > > > projection    expression=@4, alias=@5\n"
"@4  7..8  > > > > integer     1\n"
"@5  7..8  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);
    ck_assert_int_eq(cypher_astnode_range(ast).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(ast).end.offset, 9);

    const cypher_astnode_t *body = cypher_ast_statement_get_body(ast);
    ck_assert_int_eq(cypher_astnode_type(body), CYPHER_AST_QUERY);
    ck_assert_int_eq(cypher_astnode_range(body).start.offset, 0);
    ck_assert_int_eq(cypher_astnode_range(body).end.offset, 9);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
    ck_assert_ptr_eq(cypher_ast_statement_get_option(ast, 0), NULL);
}
END_TEST


START_TEST (parse_statement_with_cypher_option)
{
    result = cypher_parse("CYPHER RETURN 1;", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..16  statement           options=[@1], body=@2\n"
"@1   0..7   > CYPHER\n"
"@2   7..16  > query             clauses=[@3]\n"
"@3   7..15  > > RETURN          projections=[@4]\n"
"@4  14..15  > > > projection    expression=@5, alias=@6\n"
"@5  14..15  > > > > integer     1\n"
"@6  14..15  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    ck_assert_ptr_eq(cypher_ast_cypher_option_get_version(option), NULL);
    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 0);
    ck_assert_ptr_eq(cypher_ast_cypher_option_get_param(option, 0), NULL);
}
END_TEST


START_TEST (parse_statement_with_cypher_option_containing_version)
{
    result = cypher_parse("CYPHER 3.0 RETURN 1;", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..20  statement           options=[@1], body=@3\n"
"@1   0..10  > CYPHER            version=@2\n"
"@2   7..10  > > string          \"3.0\"\n"
"@3  11..20  > query             clauses=[@4]\n"
"@4  11..19  > > RETURN          projections=[@5]\n"
"@5  18..19  > > > projection    expression=@6, alias=@7\n"
"@6  18..19  > > > > integer     1\n"
"@7  18..19  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    const cypher_astnode_t *version =
            cypher_ast_cypher_option_get_version(option);
    ck_assert_int_eq(cypher_astnode_type(version), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(version), "3.0");

    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 0);
    ck_assert_ptr_eq(cypher_ast_cypher_option_get_param(option, 0), NULL);
}
END_TEST


START_TEST (parse_statement_params_types)
{
    result = cypher_parse("CYPHER pos_int_val=1 neg_int_val=-1 pos_float_val=2.3 neg_float_val=-2.3 true_val=true false_val=false null_val=NULL string_val='str' arr_val=[1,2,3] map_val={ int_val:1, float_val:2.3} RETURN 1", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0    0..194  statement             options=[@1], body=@41\n"
" @1    0..186  > CYPHER              params=[@2, @5, @9, @12, @16, @19, @22, @25, @28, @34]\n"
" @2    7..21   > > cypher parameter  @3 = @4\n"
" @3    7..18   > > > string          \"pos_int_val\"\n"
" @4   19..20   > > > integer         1\n"
" @5   21..36   > > cypher parameter  @6 = @7\n"
" @6   21..32   > > > string          \"neg_int_val\"\n"
" @7   33..36   > > > unary operator  - @8\n"
" @8   34..35   > > > > integer       1\n"
" @9   36..54   > > cypher parameter  @10 = @11\n"
"@10   36..49   > > > string          \"pos_float_val\"\n"
"@11   50..53   > > > float           2.3\n"
"@12   54..73   > > cypher parameter  @13 = @14\n"
"@13   54..67   > > > string          \"neg_float_val\"\n"
"@14   68..73   > > > unary operator  - @15\n"
"@15   69..72   > > > > float         2.3\n"
"@16   73..87   > > cypher parameter  @17 = @18\n"
"@17   73..81   > > > string          \"true_val\"\n"
"@18   82..86   > > > TRUE\n"
"@19   87..103  > > cypher parameter  @20 = @21\n"
"@20   87..96   > > > string          \"false_val\"\n"
"@21   97..102  > > > FALSE\n"
"@22  103..117  > > cypher parameter  @23 = @24\n"
"@23  103..111  > > > string          \"null_val\"\n"
"@24  112..116  > > > NULL\n"
"@25  117..134  > > cypher parameter  @26 = @27\n"
"@26  117..127  > > > string          \"string_val\"\n"
"@27  128..133  > > > string          \"str\"\n"
"@28  134..150  > > cypher parameter  @29 = @30\n"
"@29  134..141  > > > string          \"arr_val\"\n"
"@30  142..149  > > > collection      [@31, @32, @33]\n"
"@31  143..144  > > > > integer       1\n"
"@32  145..146  > > > > integer       2\n"
"@33  147..148  > > > > integer       3\n"
"@34  150..186  > > cypher parameter  @35 = @36\n"
"@35  150..157  > > > string          \"map_val\"\n"
"@36  158..185  > > > map             {@37:@38, @39:@40}\n"
"@37  160..167  > > > > prop name     `int_val`\n"
"@38  168..169  > > > > integer       1\n"
"@39  171..180  > > > > prop name     `float_val`\n"
"@40  181..184  > > > > float         2.3\n"
"@41  186..194  > query               clauses=[@42]\n"
"@42  186..194  > > RETURN            projections=[@43]\n"
"@43  193..194  > > > projection      expression=@44, alias=@45\n"
"@44  193..194  > > > > integer       1\n"
"@45  193..194  > > > > identifier    `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);
    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 10);

    const cypher_astnode_t *param =
            cypher_ast_cypher_option_get_param(option, 0);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);

    const cypher_astnode_t *name =
            cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    const cypher_astnode_t *value = 
            cypher_ast_cypher_option_param_get_value(param);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "pos_int_val");
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(value), "1");
     
    param = cypher_ast_cypher_option_get_param(option, 1);
    ck_assert_int_eq(cypher_astnode_type(param),CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    const cypher_astnode_t * op = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "neg_int_val");
    ck_assert_int_eq(cypher_astnode_type(op), CYPHER_AST_UNARY_OPERATOR);
    ck_assert(cypher_ast_unary_operator_get_operator(op) == CYPHER_OP_UNARY_MINUS);
    value = cypher_ast_unary_operator_get_argument(op);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(value), "1");


    param = cypher_ast_cypher_option_get_param(option, 2);
    ck_assert_int_eq(cypher_astnode_type(param),CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "pos_float_val");
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_FLOAT);
    ck_assert_str_eq(cypher_ast_float_get_valuestr(value), "2.3");

    param = cypher_ast_cypher_option_get_param(option, 3);
    ck_assert_int_eq(cypher_astnode_type(param),CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    op = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "neg_float_val");
    ck_assert_int_eq(cypher_astnode_type(op), CYPHER_AST_UNARY_OPERATOR);
    ck_assert(cypher_ast_unary_operator_get_operator(op) == CYPHER_OP_UNARY_MINUS);
    value = cypher_ast_unary_operator_get_argument(op);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_FLOAT);
    ck_assert_str_eq(cypher_ast_float_get_valuestr(value), "2.3");

    param = cypher_ast_cypher_option_get_param(option, 4);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_TRUE);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "true_val");

    param = cypher_ast_cypher_option_get_param(option, 5);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_FALSE);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "false_val");

    param = cypher_ast_cypher_option_get_param(option, 6);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_NULL);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "null_val");

    param = cypher_ast_cypher_option_get_param(option, 7);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "string_val");
    ck_assert_str_eq(cypher_ast_string_get_value(value), "str");

    param = cypher_ast_cypher_option_get_param(option, 8);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_COLLECTION);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "arr_val");

    param = cypher_ast_cypher_option_get_param(option, 9);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_MAP);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "map_val");
}
END_TEST

START_TEST (parse_params_only)
{
    result = cypher_parse("CYPHER param1=1 param2='str' MATCH (n) WHERE n.x = $param1 and n.y = $param2 RETURN n", NULL, NULL, CYPHER_PARSE_ONLY_PARAMETERS);
    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..85  statement             options=[@1], body=@8\n"
"@1   0..29  > CYPHER              params=[@2, @5]\n"
"@2   7..16  > > cypher parameter  @3 = @4\n"
"@3   7..13  > > > string          \"param1\"\n"
"@4  14..15  > > > integer         1\n"
"@5  16..29  > > cypher parameter  @6 = @7\n"
"@6  16..22  > > > string          \"param2\"\n"
"@7  23..28  > > > string          \"str\"\n"
"@8  29..85  > string              \"MATCH (n) WHERE n.x = $param1 and n.y = $param2 RETURN n\"\n";
    ck_assert_str_eq(memstream_buffer, expected);
    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 2);

    const cypher_astnode_t *param =
            cypher_ast_cypher_option_get_param(option, 0);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);

    const cypher_astnode_t *name =
            cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    const cypher_astnode_t *value = 
            cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_INTEGER);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "param1");
    ck_assert_str_eq(cypher_ast_integer_get_valuestr(value), "1");

    param = cypher_ast_cypher_option_get_param(option, 1);
    ck_assert_int_eq(cypher_astnode_type(param),CYPHER_AST_CYPHER_OPTION_PARAM);
    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(name), "param2");
    ck_assert_str_eq(cypher_ast_string_get_value(value), "str");
}
END_TEST

START_TEST (parse_params_only_without_params)
{
    result = cypher_parse("MATCH (n) WHERE n.x = $param1 and n.y = $param2 RETURN n", NULL, NULL, CYPHER_PARSE_ONLY_PARAMETERS);
    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0  0..56  statement  body=@1\n"
"@1  0..56  > string   \"MATCH (n) WHERE n.x = $param1 and n.y = $param2 RETURN n\"\n";
    ck_assert_str_eq(memstream_buffer, expected);
    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 0);
}
END_TEST


START_TEST (parse_statement_with_cypher_option_containing_params)
{
    result = cypher_parse("CYPHER runtime=\"fast\" RETURN 1;",
            NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..31  statement             options=[@1], body=@5\n"
"@1   0..22  > CYPHER              params=[@2]\n"
"@2   7..22  > > cypher parameter  @3 = @4\n"
"@3   7..14  > > > string          \"runtime\"\n"
"@4  15..21  > > > string          \"fast\"\n"
"@5  22..31  > query               clauses=[@6]\n"
"@6  22..30  > > RETURN            projections=[@7]\n"
"@7  29..30  > > > projection      expression=@8, alias=@9\n"
"@8  29..30  > > > > integer       1\n"
"@9  29..30  > > > > identifier    `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    ck_assert_ptr_eq(cypher_ast_cypher_option_get_version(option), NULL);

    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 1);
    const cypher_astnode_t *param =
            cypher_ast_cypher_option_get_param(option, 0);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);

    const cypher_astnode_t *name =
            cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    const cypher_astnode_t *value =
            cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);

    ck_assert_str_eq(cypher_ast_string_get_value(name), "runtime");
    ck_assert_str_eq(cypher_ast_string_get_value(value), "fast");
}
END_TEST


START_TEST (parse_statement_with_cypher_option_containing_version_and_params)
{
    result = cypher_parse("CYPHER 2.3 runtime=\"fast\" planner=\"slow\" RETURN 1;",
            NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
" @0   0..50  statement             options=[@1], body=@9\n"
" @1   0..41  > CYPHER              version=@2, params=[@3, @6]\n"
" @2   7..10  > > string            \"2.3\"\n"
" @3  11..26  > > cypher parameter  @4 = @5\n"
" @4  11..18  > > > string          \"runtime\"\n"
" @5  19..25  > > > string          \"fast\"\n"
" @6  26..41  > > cypher parameter  @7 = @8\n"
" @7  26..33  > > > string          \"planner\"\n"
" @8  34..40  > > > string          \"slow\"\n"
" @9  41..50  > query               clauses=[@10]\n"
"@10  41..49  > > RETURN            projections=[@11]\n"
"@11  48..49  > > > projection      expression=@12, alias=@13\n"
"@12  48..49  > > > > integer       1\n"
"@13  48..49  > > > > identifier    `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    const cypher_astnode_t *version =
            cypher_ast_cypher_option_get_version(option);
    ck_assert_int_eq(cypher_astnode_type(version), CYPHER_AST_STRING);
    ck_assert_str_eq(cypher_ast_string_get_value(version), "2.3");

    ck_assert_int_eq(cypher_ast_cypher_option_nparams(option), 2);

    const cypher_astnode_t *param =
            cypher_ast_cypher_option_get_param(option, 0);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);

    const cypher_astnode_t *name =
            cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    const cypher_astnode_t *value =
            cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);

    ck_assert_str_eq(cypher_ast_string_get_value(name), "runtime");
    ck_assert_str_eq(cypher_ast_string_get_value(value), "fast");

    param = cypher_ast_cypher_option_get_param(option, 1);
    ck_assert_int_eq(cypher_astnode_type(param),
            CYPHER_AST_CYPHER_OPTION_PARAM);

    name = cypher_ast_cypher_option_param_get_name(param);
    ck_assert_int_eq(cypher_astnode_type(name), CYPHER_AST_STRING);
    value = cypher_ast_cypher_option_param_get_value(param);
    ck_assert_int_eq(cypher_astnode_type(value), CYPHER_AST_STRING);

    ck_assert_str_eq(cypher_ast_string_get_value(name), "planner");
    ck_assert_str_eq(cypher_ast_string_get_value(value), "slow");
}
END_TEST


START_TEST (parse_statement_with_explain_option)
{
    result = cypher_parse("EXPLAIN RETURN 1;", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..17  statement           options=[@1], body=@2\n"
"@1   0..7   > EXPLAIN\n"
"@2   8..17  > query             clauses=[@3]\n"
"@3   8..16  > > RETURN          projections=[@4]\n"
"@4  15..16  > > > projection    expression=@5, alias=@6\n"
"@5  15..16  > > > > integer     1\n"
"@6  15..16  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_EXPLAIN_OPTION);
}
END_TEST


START_TEST (parse_statement_with_profile_option)
{
    result = cypher_parse("PROFILE RETURN 1;", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..17  statement           options=[@1], body=@2\n"
"@1   0..7   > PROFILE\n"
"@2   8..17  > query             clauses=[@3]\n"
"@3   8..16  > > RETURN          projections=[@4]\n"
"@4  15..16  > > > projection    expression=@5, alias=@6\n"
"@5  15..16  > > > > integer     1\n"
"@6  15..16  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 1);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_PROFILE_OPTION);
}
END_TEST


START_TEST (parse_statement_with_multiple_options)
{
    result = cypher_parse("CYPHER 3.0 PROFILE RETURN 1;", NULL, NULL, 0);

    ck_assert(cypher_parse_result_fprint_ast(result, memstream, 0, NULL, 0) == 0);
    fflush(memstream);
    const char *expected = "\n"
"@0   0..28  statement           options=[@1, @3], body=@4\n"
"@1   0..10  > CYPHER            version=@2\n"
"@2   7..10  > > string          \"3.0\"\n"
"@3  11..18  > PROFILE\n"
"@4  19..28  > query             clauses=[@5]\n"
"@5  19..27  > > RETURN          projections=[@6]\n"
"@6  26..27  > > > projection    expression=@7, alias=@8\n"
"@7  26..27  > > > > integer     1\n"
"@8  26..27  > > > > identifier  `1`\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(cypher_parse_result_ndirectives(result), 1);
    const cypher_astnode_t *ast = cypher_parse_result_get_directive(result, 0);
    ck_assert_int_eq(cypher_astnode_type(ast), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(cypher_ast_statement_noptions(ast), 2);
    const cypher_astnode_t *option = cypher_ast_statement_get_option(ast, 0);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_CYPHER_OPTION);

    option = cypher_ast_statement_get_option(ast, 1);
    ck_assert(cypher_astnode_instanceof(option, CYPHER_AST_STATEMENT_OPTION));
    ck_assert_int_eq(cypher_astnode_type(option), CYPHER_AST_PROFILE_OPTION);
}
END_TEST


TCase* statement_tcase(void)
{
    TCase *tc = tcase_create("statement");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, parse_statement_with_no_options);
    tcase_add_test(tc, parse_statement_with_cypher_option);
    tcase_add_test(tc, parse_statement_with_cypher_option_containing_version);
    tcase_add_test(tc, parse_statement_with_cypher_option_containing_params);
    tcase_add_test(tc, parse_statement_with_cypher_option_containing_version_and_params);
    tcase_add_test(tc, parse_statement_with_explain_option);
    tcase_add_test(tc, parse_statement_with_profile_option);
    tcase_add_test(tc, parse_statement_with_multiple_options);
    tcase_add_test(tc, parse_statement_params_types);
    tcase_add_test(tc, parse_params_only);
    tcase_add_test(tc, parse_params_only_without_params);
    return tc;
}
