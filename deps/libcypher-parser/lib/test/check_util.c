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
#include "../../lib/src/util.h"
#include <check.h>
#include <errno.h>
#include <unistd.h>


static const char *sample_text =
//   0        10        20        30        40        50
    "The California climate is perfect for motorcycles,\r\n"
    "as well as surfboards, swimming pools and convertibles.\n"
    "Most of the cyclists are harmless weekend types,\n"
    "members of the American Motorcycle Association,\n"
    "and no\n"
    "more dangerous than skiers or skin divers.\n";

static char *ctx;


static void setup(void)
{
    ctx = NULL;
}


static void teardown(void)
{
    free(ctx);
}


START_TEST (obtain_simple_context)
{
    size_t offset = 130;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 80);
    ck_assert_str_eq(ctx, "Most of the cyclists are harmless weekend types,");
    ck_assert_int_eq(sample_text[130], ctx[offset]);
}
END_TEST


START_TEST (obtain_short_context)
{
    size_t offset = 130;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 40);
    ck_assert_str_eq(ctx, "...of the cyclists are harmless weeke...");
    ck_assert_int_eq(sample_text[130], ctx[offset]);
}
END_TEST


START_TEST (obtain_short_context_at_start)
{
    size_t offset = 120;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 40);
    ck_assert_str_eq(ctx, "Most of the cyclists are harmless wee...");
    ck_assert_int_eq(sample_text[120], ctx[offset]);
}
END_TEST


START_TEST (obtain_short_context_at_end)
{
    size_t offset = 140;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 40);
    ck_assert_str_eq(ctx, "... cyclists are harmless weekend types,");
    ck_assert_int_eq(sample_text[140], ctx[offset]);
}
END_TEST


START_TEST (obtain_context_at_end_of_line)
{
    size_t offset = 51;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 80);
    ck_assert_str_eq(ctx, "The California climate is perfect for motorcycles,");
    ck_assert_int_eq(offset, 51);
    free(ctx);

    offset = 50;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 80);
    ck_assert_str_eq(ctx, "The California climate is perfect for motorcycles,");
    ck_assert_int_eq(offset, 50);
}
END_TEST


START_TEST (obtain_context_of_short_line)
{
    size_t offset = 208;
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 80);

    ck_assert(ctx != NULL);
    ck_assert_str_eq(ctx, "and no");
    ck_assert_int_eq(sample_text[208], ctx[offset]);
}
END_TEST


START_TEST (obtain_context_of_end_of_input)
{
    size_t offset = strlen(sample_text);
    ctx = line_context(sample_text, strlen(sample_text)+1, &offset, 80);

    ck_assert(ctx != NULL);
    ck_assert_str_eq(ctx, "more dangerous than skiers or skin divers.");
    ck_assert_int_eq(offset, 43);
}
END_TEST


TCase* util_tcase(void)
{
    TCase *tc = tcase_create("util");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, obtain_simple_context);
    tcase_add_test(tc, obtain_short_context);
    tcase_add_test(tc, obtain_short_context_at_start);
    tcase_add_test(tc, obtain_short_context_at_end);
    tcase_add_test(tc, obtain_context_at_end_of_line);
    tcase_add_test(tc, obtain_context_of_short_line);
    tcase_add_test(tc, obtain_context_of_end_of_input);
    return tc;
}
