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
#ifndef CYPHER_PARSER_AST_H
#define CYPHER_PARSER_AST_H

#include "cypher-parser.h"


unsigned int cypher_ast_set_ordinals(cypher_astnode_t *ast, unsigned int n);

int cypher_ast_fprintv(cypher_astnode_t * const *asts, unsigned int n,
        FILE *stream, unsigned int width,
        const struct cypher_parser_colorization *colorization,
        uint_fast32_t flags);

void cypher_ast_vfree(cypher_astnode_t * const *ast, unsigned int n);

cypher_astnode_t **cypher_ast_vclone(cypher_astnode_t * const *ast,
        unsigned int n);


#endif/*CYPHER_PARSER_AST_H*/
