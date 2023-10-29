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
#ifndef CYPHER_PARSER_SEGMENT_H
#define CYPHER_PARSER_SEGMENT_H

#include "cypher-parser.h"
#include "errors.h"


struct cypher_parse_segment
{
    unsigned int refcount;

    struct cypher_input_range range;

    cypher_parse_error_t *errors;
    unsigned int nerrors;

    cypher_astnode_t **roots;
    unsigned int nroots;
    unsigned int nnodes;

    const cypher_astnode_t *directive;
    bool eof;
};


cypher_parse_segment_t *cypher_parse_segment(unsigned int ordinal,
        struct cypher_input_range range, cypher_parse_error_t *errors,
        unsigned int nerrors, cypher_astnode_t **roots, unsigned int nroots,
        const cypher_astnode_t *directive, bool eof);


#endif/*CYPHER_PARSER_SEGMENT_H*/
