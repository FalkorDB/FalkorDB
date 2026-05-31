#include <check.h>
#include <stdlib.h>
#include <string.h>

/* Include the AST header to access AST types and functions */
#include "../../src/ast/ast.h"

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

        /* Final free */
        AST_Free(ast);

        cypher_parse_result_free(parse_result);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

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