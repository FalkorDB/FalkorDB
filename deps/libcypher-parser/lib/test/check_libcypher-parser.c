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
#include <check.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

Suite *libcypher_parser_suite(void);


int main(void)
{
    unsigned int seed = random();
    printf("Initialising check using random seed: %d\n", seed);
    srandom(seed);

    int number_failed;
    Suite *s = libcypher_parser_suite();
    SRunner *sr = srunner_create(s);
    srunner_set_xml(sr, "results.xml");
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
