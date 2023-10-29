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
#include "result.h"
#include "ast.h"
#include "segment.h"
#include "util.h"
#include <assert.h>


unsigned int cypher_parse_result_nroots(const cypher_parse_result_t *result)
{
    return result->nroots;
}


const cypher_astnode_t *cypher_parse_result_get_root(
        const cypher_parse_result_t *result, unsigned int index)
{
    if (index >= result->nroots)
    {
        return NULL;
    }
    return result->roots[index];
}


unsigned int cypher_parse_result_nnodes(const cypher_parse_result_t *result)
{
    return result->nnodes;
}


unsigned int cypher_parse_result_ndirectives(
        const cypher_parse_result_t *result)
{
    return result->ndirectives;
}


const cypher_astnode_t *cypher_parse_result_get_directive(
                const cypher_parse_result_t *result, unsigned int index)
{
    if (index >= result->ndirectives)
    {
        return NULL;
    }
    return result->directives[index];
}


unsigned int cypher_parse_result_nerrors(const cypher_parse_result_t *result)
{
    return result->nerrors;
}


const cypher_parse_error_t *cypher_parse_result_get_error(
        const cypher_parse_result_t *result, unsigned int index)
{
    if (index >= result->nerrors)
    {
        return NULL;
    }
    return &(result->errors[index]);
}


bool cypher_parse_result_eof(const cypher_parse_result_t *result)
{
    return result->eof;
}


int cp_result_merge_segment(cypher_parse_result_t *result,
        cypher_parse_segment_t *segment)
{
    if (!result->eof && segment->eof &&
            (segment->directive != NULL || segment->nerrors > 0))
    {
        result->eof = true;
    }

    if (segment->nerrors > 0)
    {
        unsigned int n = result->nerrors + segment->nerrors;
        cypher_parse_error_t *errors = realloc(result->errors,
                n * sizeof(cypher_parse_error_t));
        if (errors == NULL)
        {
            return -1;
        }
        memcpy(errors + result->nerrors, segment->errors,
                segment->nerrors * sizeof(cypher_parse_error_t));
        segment->nerrors = 0;
        result->errors = errors;
        result->nerrors = n;
    }

    if (segment->nroots > 0)
    {
        unsigned int n = result->nroots + segment->nroots;
        cypher_astnode_t **roots = realloc(result->roots,
                n * sizeof(cypher_astnode_t *));
        if (roots == NULL)
        {
            return -1;
        }
        memcpy(roots + result->nroots, segment->roots,
                segment->nroots * sizeof(cypher_astnode_t *));
        segment->nroots = 0;
        result->roots = roots;
        result->nroots = n;
    }

    result->nnodes += segment->nnodes;

    if (segment->directive != NULL)
    {
        assert(result->directives_cap >= result->ndirectives);
        if (result->directives_cap <= result->ndirectives)
        {
            unsigned int n = (result->directives_cap == 0)?
                    8 : result->directives_cap * 2;
            const cypher_astnode_t **directives = realloc(result->directives,
                    n * sizeof(const cypher_astnode_t *));
            if (directives == NULL)
            {
                return -1;
            }
            result->directives = directives;
            result->directives_cap = n;
        }
        result->directives[(result->ndirectives)++] = segment->directive;
        segment->directive = NULL;
    }

    return 0;
}


int cypher_parse_result_fprint_ast(const cypher_parse_result_t *result,
        FILE *stream, unsigned int width,
        const struct cypher_parser_colorization *colorization,
        uint_fast32_t flags)
{
    return cypher_ast_fprintv(result->roots, result->nroots,
            stream, width, colorization, flags);
}


size_t cypher_parse_error_context_offset(const cypher_parse_error_t *error)
{
    REQUIRE(error != NULL, 0);
    return error->context_offset;
}


void cypher_parse_result_free(cypher_parse_result_t *result)
{
    if (result == NULL)
    {
        return;
    }

    cp_errors_vcleanup(result->errors, result->nerrors);
    free(result->errors);
    cypher_ast_vfree(result->roots, result->nroots);
    free(result->roots);
    free(result->directives);
    free(result);
}
