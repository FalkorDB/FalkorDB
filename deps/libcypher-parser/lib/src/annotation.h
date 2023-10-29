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
#ifndef CYPHER_PARSER_ANNOTATION_H
#define CYPHER_PARSER_ANNOTATION_H

#include "cypher-parser.h"


struct cypher_astnode_annotation
{
    cypher_ast_annotation_context_t *context;
    const cypher_astnode_t *astnode;
    void *data;

    struct cypher_astnode_annotation *node_prev;
    struct cypher_astnode_annotation *node_next;

    struct cypher_astnode_annotation *ctx_prev;
    struct cypher_astnode_annotation *ctx_next;
};


struct cypher_ast_annotation_context
{
    cypher_ast_annotation_context_release_handler_t release_cb;
    void *release_cb_userdata;
    struct cypher_astnode_annotation *annotations;
};


void cp_release_annotation(struct cypher_astnode_annotation *annotation);


#endif/*CYPHER_PARSER_ANNOTATION_H*/
