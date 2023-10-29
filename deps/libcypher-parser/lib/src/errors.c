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
#include "errors.h"
#include "util.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>


#define CYPHER_ERROR_LABELS_BLOCK_SIZE 8
#define CYPHER_PARSER_ERRORS_BLOCK_SIZE 8

static char *chardesc(char *buf, size_t size, char c);
static char *error_report(cp_error_tracking_t *et, char **buffer, size_t *cap,
        const char *prefix_format, ...) __cypherlang_format(4, 5);


struct cypher_input_position cypher_parse_error_position(
        const cypher_parse_error_t *error)
{
    REQUIRE(error != NULL, cypher_input_position_zero);
    return error->position;
}


const char *cypher_parse_error_message(const cypher_parse_error_t *error)
{
    REQUIRE(error != NULL, 0);
    return error->msg;
}


const char *cypher_parse_error_context(const cypher_parse_error_t *error)
{
    REQUIRE(error != NULL, 0);
    return error->context;
}


void cp_errors_vcleanup(cypher_parse_error_t *errors, unsigned int n)
{
    for (unsigned int i = n; i-- > 0; errors++)
    {
        free(errors->msg);
        errors->msg = NULL;
        free(errors->context);
        errors->context = NULL;
    }
}


void cp_et_init(cp_error_tracking_t *et,
        const struct cypher_parser_colorization *colorization)
{
    assert(colorization != NULL);
    memset(et, 0, sizeof(cp_error_tracking_t));
    et->colorization = colorization;
}


int cp_et_note_potential_error(cp_error_tracking_t *et,
        struct cypher_input_position position, char c, const char *label)
{
    if (position.offset < et->last_position.offset || 
            (et->nerrors > 0 && position.offset <= et->last_error_offset))
    {
        return 0;
    }

    if (et->nlabels == 0 || position.offset > et->last_position.offset)
    {
        et->last_position = position;
        et->last_char = c;
        et->nlabels = 0;
    }

#ifndef NDEBUG
    if (et->nlabels != 0)
    {
        assert(et->last_position.line == position.line);
        assert(et->last_position.column == position.column);
        assert(et->last_char == c);
    }
#endif

    for (unsigned int i = et->nlabels; i-- > 0; )
    {
        if (strcmp(label, et->labels[i]) == 0)
        {
            return 0;
        }
    }

    assert(et->nlabels <= et->labels_capacity);
    if (et->nlabels >= et->labels_capacity)
    {
        unsigned int newcap = (et->labels_capacity == 0)?
            CYPHER_ERROR_LABELS_BLOCK_SIZE : et->labels_capacity * 2;
        void *labels = realloc(et->labels, newcap * sizeof(const char *));
        if (labels == NULL)
        {
            return -1;
        }
        et->labels_capacity = newcap;
        et->labels = labels;
    }
    assert(et->nlabels < et->labels_capacity);

    et->labels[et->nlabels] = label;
    ++(et->nlabels);
    return 0;
}


int cp_et_reify_potentials(cp_error_tracking_t *et)
{
    if (et->nlabels == 0)
    {
        return 0;
    }

    assert(et->nerrors <= et->errors_capacity);
    if (et->nerrors >= et->errors_capacity)
    {
        unsigned int newcap = (et->errors_capacity == 0)?
            CYPHER_PARSER_ERRORS_BLOCK_SIZE : et->errors_capacity * 2;
        void *errors = realloc(et->errors,
                newcap * sizeof(cypher_parse_error_t));
        if (errors == NULL)
        {
            return -1;
        }
        et->errors_capacity = newcap;
        et->errors = errors;
    }
    assert(et->nerrors < et->errors_capacity);

    char buf[4];
    char *msg = error_report(et, NULL, NULL,
            "%sInvalid input%s %s%s%s: ",
            et->colorization->error[0], et->colorization->error[1],
            et->colorization->error_token[0],
            chardesc(buf, sizeof(buf), et->last_char),
            et->colorization->error_token[1]);
    if (msg == NULL)
    {
        return -1;
    }

    memset(&(et->errors[et->nerrors]), 0, sizeof(cypher_parse_error_t));
    et->errors[et->nerrors].position = et->last_position;
    et->errors[et->nerrors].msg = msg;
    ++(et->nerrors);
    et->last_error_offset = et->last_position.offset;

    et->last_position.line = 0;
    et->last_position.column = 0;
    et->last_position.offset = 0;
    et->last_char = 0;
    et->nlabels = 0;
    return 0;
}


char *chardesc(char *buf, size_t size, char c)
{
    assert(size >= 4);
    switch (c)
    {
    case '\a': return "'\\a'";
    case '\b': return "'\\b'";
    case '\f': return "'\\f'";
    case '\n': return "'\\n'";
    case '\r': return "'\\r'";
    case '\t': return "'\\t'";
    case '\v': return "'\\v'";
    case '\'': return "'\\''";
    case '\0': return "at end of input";
    default:
        buf[0] = '\'';
        buf[1] = c;
        buf[2] = '\'';
        buf[3] = '\0';
        return buf;
    }
}


char *error_report(cp_error_tracking_t *et, char **buffer, size_t *cap,
        const char *prefix_format, ...)
{
    ENSURE_NOT_NULL(char *, buffer, NULL);
    ENSURE_NOT_NULL(size_t, cap, 0);

    va_list ap;

    va_start(ap, prefix_format);
    int prefix_len = vsnprintf(*buffer, *cap, prefix_format, ap);
    va_end(ap);
    if (prefix_len < 0)
    {
        return NULL;
    }

    size_t color_start_len = strlen(et->colorization->error_message[0]);
    size_t color_end_len = strlen(et->colorization->error_message[1]);
    size_t len = prefix_len + color_start_len + 8 + color_end_len + 1;
    if (et->nlabels > 0)
    {
        for (unsigned int i = et->nlabels; i-- > 0; )
        {
            len += strlen(et->labels[i]) + 2;
        }
        len -= 1;
        if (et->nlabels > 1)
        {
            len += 2;
        }
    }

    size_t orig_cap = *cap;
    if (orig_cap < len)
    {
        void *newbuf = realloc(*buffer, len);
        if (newbuf == NULL)
        {
            return NULL;
        }
        *buffer = newbuf;
        *cap = len;
    }

    if (orig_cap < ((size_t)prefix_len + 1))
    {
        va_start(ap, prefix_format);
        int r = vsnprintf(*buffer, *cap, prefix_format, ap);
        va_end(ap);
        if (r < 0)
        {
            return NULL;
        }
    }

    char *dest = *buffer + prefix_len;

    memcpy(dest, et->colorization->error_message[0], color_start_len);
    dest += color_start_len;
    memcpy(dest, "expected", 8);
    dest += 8;

    for (unsigned int i = 0; i < et->nlabels;)
    {
        *(dest++) = ' ';
        size_t n = strlen(et->labels[i]);
        memcpy(dest, et->labels[i], n);
        dest += n;
        ++i;
        if (i+1 < et->nlabels)
        {
            *(dest++) = ',';
        }
        else if (i < et->nlabels)
        {
            memcpy(dest, " or", 3);
            dest += 3;
        }
    }

    memcpy(dest, et->colorization->error_message[1], color_end_len);
    dest += color_end_len;

    assert((*buffer + *cap) > dest);
    *dest = '\0';
    return *buffer;
}


cypher_parse_error_t *cp_et_extract_errors(cp_error_tracking_t *et,
        unsigned int *nerrors)
{
    cypher_parse_error_t *errors = et->errors;
    *nerrors = et->nerrors;
    et->errors = NULL;
    et->errors_capacity = 0;
    et->nerrors = 0;
    return errors;
}


void cp_et_cleanup(cp_error_tracking_t *et)
{
    free(et->labels);
    et->labels_capacity = et->nlabels = 0;
    et->labels = NULL;

    cp_errors_vcleanup(et->errors, et->nerrors);
    free(et->errors);
    et->errors_capacity = et->nerrors = 0;
    et->errors = NULL;
}
