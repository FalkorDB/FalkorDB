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
#ifndef CYPHER_PARSER_RESULT_H
#define CYPHER_PARSER_RESULT_H

#include "cypher-parser.h"
#include "errors.h"


struct cypher_parse_result
{
    cypher_parse_error_t *errors;
    unsigned int nerrors;

    cypher_astnode_t **roots;
    unsigned int nroots;
    unsigned int nnodes;

    const cypher_astnode_t **directives;
    unsigned int ndirectives;
    unsigned int directives_cap;

    bool eof;
};


int cp_result_merge_segment(cypher_parse_result_t *result,
        cypher_parse_segment_t *segment);


#endif/*CYPHER_PARSER_RESULT_H*/
