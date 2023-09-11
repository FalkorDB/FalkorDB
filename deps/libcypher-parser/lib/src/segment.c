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
#include "segment.h"
#include "ast.h"
#include "util.h"
#include <assert.h>


cypher_parse_segment_t *cypher_parse_segment(unsigned int ordinal,
        struct cypher_input_range range, cypher_parse_error_t *errors,
        unsigned int nerrors, cypher_astnode_t **roots, unsigned int nroots,
        const cypher_astnode_t *directive, bool eof)
{
    struct cypher_parse_segment *segment = calloc(1,
            sizeof(cypher_parse_segment_t));
    if (segment == NULL)
    {
        return NULL;
    }

    segment->refcount = 1;
    segment->range = range;
    if (nerrors > 0)
    {
        segment->errors = mdup(errors, nerrors * sizeof(cypher_parse_error_t));
        if (segment->errors == NULL)
        {
            goto failure;
        }
        segment->nerrors = nerrors;
    }
    if (nroots > 0)
    {
        segment->roots = mdup(roots, nroots * sizeof(cypher_astnode_t *));
        if (segment->roots == NULL)
        {
            goto failure;
        }
        segment->nroots = nroots;
    }
    segment->directive = directive;
    segment->eof = eof;

    unsigned int initial_ordinal = ordinal;
    for (unsigned int i = 0; i < nroots; ++i)
    {
        ordinal = cypher_ast_set_ordinals(roots[i], ordinal);
    }
    segment->nnodes = ordinal - initial_ordinal;

    return segment;

    int errsv;
failure:
    errsv = errno;
    if (segment != NULL)
    {
        free(segment->errors);
        free(segment->roots);
    }
    free(segment);
    errno = errsv;
    return NULL;
}


void cypher_parse_segment_retain(cypher_parse_segment_t *segment)
{
    assert(segment != NULL);
    assert(segment->refcount > 0);
    ++(segment->refcount);
}


void cypher_parse_segment_release(cypher_parse_segment_t* segment)
{
    if (segment == NULL)
    {
        return;
    }
    assert(segment->refcount > 0);
    if (--(segment->refcount) > 0)
    {
        return;
    }

    cp_errors_vcleanup(segment->errors, segment->nerrors);
    free(segment->errors);
    cypher_ast_vfree(segment->roots, segment->nroots);
    free(segment->roots);

    memset(segment, 0, sizeof(cypher_parse_segment_t));
    free(segment);
}


struct cypher_input_range cypher_parse_segment_get_range(
        const cypher_parse_segment_t *segment)
{
    return segment->range;
}


unsigned int cypher_parse_segment_nerrors(const cypher_parse_segment_t *segment)
{
    return segment->nerrors;
}


const cypher_parse_error_t *cypher_parse_segment_get_error(
        const cypher_parse_segment_t *segment, unsigned int index)
{
    if (index >= segment->nerrors)
    {
        return NULL;
    }
    return &(segment->errors[index]);
}


unsigned int cypher_parse_segment_nroots(const cypher_parse_segment_t *segment)
{
    return segment->nroots;
}


const cypher_astnode_t *cypher_parse_segment_get_node(
        const cypher_parse_segment_t *segment, unsigned int index)
{
    if (index >= segment->nroots)
    {
        return NULL;
    }
    return segment->roots[index];
}


unsigned int cypher_parse_segment_nnodes(const cypher_parse_segment_t *segment)
{
    return segment->nnodes;
}


const cypher_astnode_t *cypher_parse_segment_get_directive(
        const cypher_parse_segment_t *segment)
{
    return segment->directive;
}


bool cypher_parse_segment_is_eof(const cypher_parse_segment_t *segment)
{
    return segment->eof;
}


int cypher_parse_segment_fprint_ast(const cypher_parse_segment_t *segment,
        FILE *stream, unsigned int width,
        const struct cypher_parser_colorization *colorization,
        uint_fast32_t flags)
{
    return cypher_ast_fprintv(segment->roots, segment->nroots,
            stream, width, colorization, flags);
}
