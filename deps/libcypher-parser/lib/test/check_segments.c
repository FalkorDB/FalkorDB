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


#define MAX_SEGMENTS 8

static char *memstream_buffer;
static size_t memstream_size;
static FILE *memstream;
static bool retain;
static unsigned int nsegments;
static cypher_parse_segment_t *segments[MAX_SEGMENTS];
static struct cypher_input_range ranges[MAX_SEGMENTS];
static const cypher_astnode_t *directives[MAX_SEGMENTS];


static void setup(void)
{
    memstream = open_memstream(&memstream_buffer, &memstream_size);
    fputc('\n', memstream);
    retain = false;
    nsegments = 0;
    memset(segments, 0, sizeof(segments));
    memset(ranges, 0, sizeof(ranges));
    memset(directives, 0, sizeof(directives));
}


static void teardown(void)
{
    fclose(memstream);
    free(memstream_buffer);

    if (retain)
    {
        for (unsigned int i = 0; i < nsegments; ++i)
        {
            cypher_parse_segment_release(segments[i]);
        }
    }
}


static int segment_callback(void *data, cypher_parse_segment_t *segment)
{
    ck_assert(nsegments < MAX_SEGMENTS);

    if (retain)
    {
        segments[nsegments] = segment;
        cypher_parse_segment_retain(segment);
    }

    ranges[nsegments] = cypher_parse_segment_get_range(segment);
    directives[nsegments] = cypher_parse_segment_get_directive(segment);
    ++nsegments;

    ck_assert(cypher_parse_segment_fprint_ast(segment, memstream, 0, NULL, 0) == 0);
    fprintf(memstream, "--%d--\n", nsegments);

    return 0;
}


START_TEST (empty_input)
{
    struct cypher_input_position last = cypher_input_position_zero;
    int result = cypher_parse_each("", segment_callback, NULL, &last, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(last.offset, 0);

    fflush(memstream);
    const char *expected = "\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(nsegments, 0);
}
END_TEST


START_TEST (single_segment_without_directive)
{
    struct cypher_input_position last = cypher_input_position_zero;
    int result = cypher_parse_each(" ;", segment_callback, NULL, &last, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(last.offset, 2);

    fflush(memstream);
    const char *expected = "\n"
"--1--\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(nsegments, 1);

    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 3);
    ck_assert_int_eq(ranges[0].end.offset, 2);
    ck_assert_ptr_eq(directives[0], NULL);
}
END_TEST


START_TEST (single_segment_with_only_a_comment)
{
    struct cypher_input_position last = cypher_input_position_zero;
    int result = cypher_parse_each("  /* foo */;", segment_callback, NULL,
            &last, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(last.offset, 12);

    fflush(memstream);
    const char *expected = "\n"
"@0  4..9  block_comment  /* foo */\n"
"--1--\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(nsegments, 1);

    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 13);
    ck_assert_int_eq(ranges[0].end.offset, 12);
    ck_assert_ptr_eq(directives[0], NULL);
}
END_TEST


START_TEST (segments_with_directives)
{
    retain = true;

    struct cypher_input_position last = cypher_input_position_zero;
    int result = cypher_parse_each(" return 1; /* foo */; return 2; return 3",
            segment_callback, NULL, &last, NULL, 0);
    ck_assert_int_eq(result, 0);
    ck_assert_int_eq(last.offset, 40);

    fflush(memstream);
    const char *expected = "\n"
"@0  1..10  statement           body=@1\n"
"@1  1..10  > query             clauses=[@2]\n"
"@2  1..9   > > RETURN          projections=[@3]\n"
"@3  8..9   > > > projection    expression=@4, alias=@5\n"
"@4  8..9   > > > > integer     1\n"
"@5  8..9   > > > > identifier  `1`\n"
"--1--\n"
"@6  13..18  block_comment  /* foo */\n"
"--2--\n"
" @7  22..31  statement           body=@8\n"
" @8  22..31  > query             clauses=[@9]\n"
" @9  22..30  > > RETURN          projections=[@10]\n"
"@10  29..30  > > > projection    expression=@11, alias=@12\n"
"@11  29..30  > > > > integer     2\n"
"@12  29..30  > > > > identifier  `2`\n"
"--3--\n"
"@13  32..40  statement           body=@14\n"
"@14  32..40  > query             clauses=[@15]\n"
"@15  32..40  > > RETURN          projections=[@16]\n"
"@16  39..40  > > > projection    expression=@17, alias=@18\n"
"@17  39..40  > > > > integer     3\n"
"@18  39..40  > > > > identifier  `3`\n"
"--4--\n";
    ck_assert_str_eq(memstream_buffer, expected);

    ck_assert_int_eq(nsegments, 4);

    ck_assert_int_eq(ranges[0].start.line, 1);
    ck_assert_int_eq(ranges[0].start.column, 1);
    ck_assert_int_eq(ranges[0].start.offset, 0);
    ck_assert_int_eq(ranges[0].end.line, 1);
    ck_assert_int_eq(ranges[0].end.column, 11);
    ck_assert_int_eq(ranges[0].end.offset, 10);
    ck_assert_ptr_ne(directives[0], NULL);
    ck_assert_int_eq(cypher_astnode_type(directives[0]), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(ranges[1].start.line, 1);
    ck_assert_int_eq(ranges[1].start.column, 11);
    ck_assert_int_eq(ranges[1].start.offset, 10);
    ck_assert_int_eq(ranges[1].end.line, 1);
    ck_assert_int_eq(ranges[1].end.column, 22);
    ck_assert_int_eq(ranges[1].end.offset, 21);
    ck_assert_ptr_eq(directives[1], NULL);

    ck_assert_int_eq(ranges[2].start.line, 1);
    ck_assert_int_eq(ranges[2].start.column, 22);
    ck_assert_int_eq(ranges[2].start.offset, 21);
    ck_assert_int_eq(ranges[2].end.line, 1);
    ck_assert_int_eq(ranges[2].end.column, 32);
    ck_assert_int_eq(ranges[2].end.offset, 31);
    ck_assert_ptr_ne(directives[2], NULL);
    ck_assert_int_eq(cypher_astnode_type(directives[2]), CYPHER_AST_STATEMENT);

    ck_assert_int_eq(ranges[3].start.line, 1);
    ck_assert_int_eq(ranges[3].start.column, 32);
    ck_assert_int_eq(ranges[3].start.offset, 31);
    ck_assert_int_eq(ranges[3].end.line, 1);
    ck_assert_int_eq(ranges[3].end.column, 41);
    ck_assert_int_eq(ranges[3].end.offset, 40);
    ck_assert_ptr_ne(directives[3], NULL);
    ck_assert_int_eq(cypher_astnode_type(directives[3]), CYPHER_AST_STATEMENT);
}
END_TEST


TCase* segments_tcase(void)
{
    TCase *tc = tcase_create("segments");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, empty_input);
    tcase_add_test(tc, single_segment_without_directive);
    tcase_add_test(tc, single_segment_with_only_a_comment);
    tcase_add_test(tc, segments_with_directives);
    return tc;
}
