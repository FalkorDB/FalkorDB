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
#include "cypher-parser.h"
#include "util.h"
#include "vector.h"
#include <assert.h>
#include <errno.h>
#include <setjmp.h>

DECLARE_VECTOR(offsets, unsigned int, 0);

typedef struct _yycontext yycontext;
typedef int (*yyrule)(yycontext *yy);
typedef int (*source_cb_t)(void *data, char *buf, int n);

static int parse(source_cb_t source, void *sourcedata,
        cypher_parser_quick_segment_callback_t callback, void *userdata,
        uint_fast32_t flags);
static void source(yycontext *yy, char *buf, int *result, int max_size);


struct source_from_buffer_data
{
    const char *buffer;
    size_t length;
};


static int source_from_buffer(void *data, char *buf, int n)
{
    struct source_from_buffer_data *input = data;
    int len = min(input->length, n);
    input->length -= len;
    if (len == 0)
    {
        return len;
    }
    memcpy(buf, input->buffer, len);
    return len;
}


static int source_from_stream(void *data, char *buf, int n)
{
    FILE *stream = data;
    int c = getc(stream);
    if (c == EOF)
    {
        return 0;
    }
    *buf = c;
    return 1;
}


int cypher_quick_uparse(const char *s, size_t n,
        cypher_parser_quick_segment_callback_t callback, void *userdata,
        uint_fast32_t flags)
{
    struct source_from_buffer_data sourcedata = { .buffer = s, .length = n };
    return parse(source_from_buffer, &sourcedata, callback, userdata, flags);
}


int cypher_quick_fparse(FILE *stream,
        cypher_parser_quick_segment_callback_t callback, void *userdata,
        uint_fast32_t flags)
{
    return parse(source_from_stream, stream, callback, userdata, flags);
}


#define YY_CTX_LOCAL
#define YY_PARSE(T) static T
#define YY_CTX_MEMBERS \
    source_cb_t source; \
    void *source_data; \
    cypher_parser_quick_segment_callback_t callback; \
    void *callback_data; \
    sigjmp_buf abort_env; \
    struct cypher_input_position position_offset; \
    offsets_t line_start_offsets; \
    bool eof; \
    int result;

#define YY_MALLOC abort_malloc
#define YY_REALLOC abort_realloc

#define YY_INPUT(yy, buf, result, max_size) \
    source(yy, buf, &result, max_size)

static void *abort_malloc(yycontext *yy, size_t size);
static void *abort_realloc(yycontext *yy, void *ptr, size_t size);
static inline void source(yycontext *yy, char *buf, int *result, int max_size);
static inline void segment(bool is_statement, yycontext *yy);
static inline void line_start(yycontext *yy);

#pragma GCC diagnostic ignored "-Wunused-function"
#include "quick_parser_leg.c"


#define abort_parse(yy) \
    do { assert(errno != 0); siglongjmp(yy->abort_env, errno); } while (0)
static int safe_yyparsefrom(yycontext *yy, yyrule rule);
static unsigned int backtrack_lines(yycontext *yy, unsigned int pos);
static struct cypher_input_position input_position(yycontext *yy,
        unsigned int pos);


void source(yycontext *yy, char *buf, int *result, int max_size)
{
    if (buf == NULL)
    {
        *result = 0;
        return;
    }
    assert(yy != NULL && yy->source != NULL);
    *result = yy->source(yy->source_data, buf, max_size);
}


int parse(source_cb_t source, void *sourcedata,
        cypher_parser_quick_segment_callback_t callback, void *userdata,
        uint_fast32_t flags)
{
    yycontext yy;
    memset(&yy, 0, sizeof(yycontext));

    yy.source = source;
    yy.source_data = sourcedata;
    yy.callback = callback;
    yy.callback_data = userdata;
    yy.position_offset = cypher_input_position_zero;
    offsets_init(&(yy.line_start_offsets));
    int err = -1;

    if (offsets_push(&(yy.line_start_offsets), 0))
    {
        goto cleanup;
    }

    yyrule rule = (flags & CYPHER_PARSE_ONLY_STATEMENTS)?
            yy_statement : yy_directive;

    for (;;)
    {
        int result = safe_yyparsefrom(&yy, rule);
        if (result <= 0)
        {
            goto cleanup;
        }

        if (yy.result > 0)
        {
            break;
        }
        else if (yy.result < 0)
        {
            err = yy.result;
            goto cleanup;
        }

        if (yy.eof || flags & CYPHER_PARSE_SINGLE)
        {
            break;
        }

        offsets_clear(&(yy.line_start_offsets));
        if (offsets_push(&(yy.line_start_offsets), 0))
        {
            goto cleanup;
        }
    }

    err = 0;

cleanup:
    offsets_cleanup(&(yy.line_start_offsets));
    yyrelease(&yy);
    return err;
}


int safe_yyparsefrom(yycontext *yy, yyrule rule)
{
    int err;
    if ((err = sigsetjmp(yy->abort_env, 0)) != 0)
    {
        errno = err;
        return -1;
    }

    int result = yyparsefrom(yy, rule);
    memset(yy->abort_env, 0, sizeof(sigjmp_buf));
    return result;
}


void *abort_malloc(yycontext *yy, size_t size)
{
    void *m = malloc(size);
    if (m == NULL)
    {
        abort_parse(yy);
    }
    return m;
}


void *abort_realloc(yycontext *yy, void *ptr, size_t size)
{
    void *m = realloc(ptr, size);
    if (m == NULL)
    {
        abort_parse(yy);
    }
    return m;
}


struct cypher_quick_parse_segment
{
    bool is_statement;
    const char *ptr;
    size_t length;
    struct cypher_input_range range;
    struct cypher_input_position next;
    bool eof;
};


void segment(bool is_statement, yycontext *yy)
{
    if (yy->__end == 0 && yy->eof)
    {
        yy->result = 0;
        return;
    }

    assert(yy->__pos >= 0);
    struct cypher_input_position consumed_offset =
            input_position(yy, (unsigned int)yy->__pos);
    struct cypher_input_position end =
            input_position(yy, (unsigned int)yy->__end);
    struct cypher_input_position start =
            input_position(yy, (unsigned int)yy->__begin);
    struct cypher_quick_parse_segment segment =
        { .is_statement = is_statement,
          .ptr = yy->__buf + yy->__begin,
          .length = yy->__end - yy->__begin,
          .range = { .start = start, .end = end },
          .next = consumed_offset,
          .eof = yy->eof };

    yy->result = yy->callback(yy->callback_data, &segment);
    yy->position_offset = consumed_offset;
}


bool cypher_quick_parse_segment_is_statement(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->is_statement;
}


bool cypher_quick_parse_segment_is_command(
        const cypher_quick_parse_segment_t *segment)
{
    return !(segment->is_statement);
}


const char *cypher_quick_parse_segment_get_text(
        const cypher_quick_parse_segment_t *segment, size_t *n)
{
    REQUIRE(n != NULL, NULL);
    *n = (segment->length);
    return segment->ptr;
}


const char *cypher_quick_parse_segment_get_ptr(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->ptr;
}


size_t cypher_quick_parse_segment_get_length(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->length;
}


struct cypher_input_range cypher_quick_parse_segment_get_range(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->range;
}


struct cypher_input_position cypher_quick_parse_segment_get_next(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->next;
}


bool cypher_quick_parse_segment_is_eof(
        const cypher_quick_parse_segment_t *segment)
{
    return segment->eof;
}


void line_start(yycontext *yy)
{
    assert(yy->__pos >= 0);
    unsigned int pos = (unsigned int)yy->__pos;
    unsigned int line_start_pos = backtrack_lines(yy, pos);
    if (line_start_pos == pos)
    {
        return;
    }
    if (offsets_push(&(yy->line_start_offsets), pos))
    {
        abort_parse(yy);
    }
}


unsigned int backtrack_lines(yycontext *yy, unsigned int pos)
{
    unsigned int top;
    while ((top = offsets_last(&(yy->line_start_offsets))) > pos)
    {
        offsets_pop(&(yy->line_start_offsets));
    }
    return top;
}


struct cypher_input_position input_position(yycontext *yy, unsigned int pos)
{
    assert(offsets_size(&(yy->line_start_offsets)) > 0);

    unsigned int depth = offsets_size(&(yy->line_start_offsets));
    unsigned int line_start_pos;
    do
    {
        line_start_pos = offsets_get(&(yy->line_start_offsets), depth-1);
    } while (line_start_pos > pos && depth > 0 && depth--);

    assert(depth > 0);
    assert(line_start_pos <= pos);

    struct cypher_input_position position =
        { .line = (depth-1) + yy->position_offset.line,
          .column = pos - line_start_pos +
              ((depth == 1)? yy->position_offset.column : 1),
          .offset = pos + yy->position_offset.offset };
    return position;
}
