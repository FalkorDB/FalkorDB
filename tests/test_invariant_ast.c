#include <check.h>
#include <stdlib.h>
#include <string.h>

/* Include the AST header to access AST types and functions */
#include "../../src/ast/ast.h"

START_TEST(test_unbounded_traversal_rejected)
{
    /* Security contract: unbounded variable-length paths must be rejected by
     * validation (V-004 fix); bounded and non-variable-length paths must be
     * accepted. This directly asserts the invariant introduced by this PR. */
    struct {
        const char *query;
        AST_Validation expected;
    } cases[] = {
        { "MATCH (a)-[*]->(b) RETURN b",      AST_INVALID }, /* unbounded: must reject */
        { "MATCH (a)-[*1..10]->(b) RETURN b", AST_VALID },   /* bounded: must accept  */
        { "MATCH (a)-[:REL]->(b) RETURN b",   AST_VALID }    /* no var-len: must accept */
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int i = 0; i < num_cases; i++) {
        cypher_parse_result_t *parse_result = cypher_parse(
            cases[i].query, NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);

        if (parse_result == NULL) {
            ck_abort_msg("cypher_parse failed for query: %s", cases[i].query);
        }

        AST *ast = AST_Build(parse_result);
        ck_assert_ptr_nonnull(ast);

        AST_Validation result = AST_Validate(ast);
        ck_assert_msg(result == cases[i].expected,
            "Query '%s': expected validation result %d but got %d",
            cases[i].query, cases[i].expected, result);

        /* AST_Free releases both ast and its owned parse_result */
        AST_Free(ast);
    }
}
END_TEST

START_TEST(test_ast_ref_count_security_invariant)
{
    /* Invariant: AST reference count must never go negative or wrap around
     * under adversarial query inputs, ensuring no use-after-free or
     * resource exhaustion via unbounded traversal patterns */
    const char *payloads[] = {
        "MATCH (a)-[*]->(b) RETURN b",          /* exact exploit: unbounded traversal */
        "MATCH (a)-[*1..999999]->(b) RETURN b", /* boundary: large depth bound */
        "MATCH (a)-[:REL]->(b) RETURN b"        /* valid: bounded single-hop */
    };
    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);

    for (int i = 0; i < num_payloads; i++) {
        AST *ast = NULL;

        /* Parse the query into an AST */
        cypher_parse_result_t *parse_result = cypher_parse(
            payloads[i], NULL, NULL, CYPHER_PARSE_ONLY_STATEMENTS);

        if (parse_result == NULL) {
            /* Parsing failed — acceptable, no AST to check */
            continue;
        }

        /* Build AST from parse result */
        ast = AST_Build(parse_result);
        if (ast == NULL) {
            /* AST_Build failed; ownership of parse_result was not transferred,
             * so it must be freed explicitly here. */
            cypher_parse_result_free(parse_result);
            continue;
        }

        /* Invariant: initial ref_count must be positive (>= 1) */
        ck_assert_int_ge(*ast->ref_count, 1);

        /* Simulate a retain/release cycle */
        AST_Retain(ast);
        ck_assert_int_ge(*ast->ref_count, 2);

        int after_release = AST_Free(ast);
        /* After one release, ref_count must still be >= 1 (not freed yet) */
        ck_assert_int_ge(after_release, 1);

        /* Final free — AST_Build transferred parse_result ownership to AST,
         * so AST_Free releases parse_result as well. Do NOT call
         * cypher_parse_result_free(parse_result) here: it would double-free. */
        AST_Free(ast);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_unbounded_traversal_rejected);
    tcase_add_test(tc_core, test_ast_ref_count_security_invariant);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
