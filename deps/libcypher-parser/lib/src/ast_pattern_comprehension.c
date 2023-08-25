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
#include "astnode.h"
#include "util.h"
#include <assert.h>


struct pattern_comprehension
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *pattern;
    const cypher_astnode_t *predicate;
    const cypher_astnode_t *eval;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_pattern_comprehension_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "pattern comprehension",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_pattern_comprehension(
        const cypher_astnode_t *identifier, const cypher_astnode_t *pattern,
        const cypher_astnode_t *predicate, const cypher_astnode_t *eval,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD_OPTIONAL(children, nchildren, identifier,
            CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, pattern, CYPHER_AST_PATTERN_PATH, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD(children, nchildren, eval, CYPHER_AST_EXPRESSION, NULL);

    struct pattern_comprehension *node =
            calloc(1, sizeof(struct pattern_comprehension));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_PATTERN_COMPREHENSION,
                children, nchildren, range))
    {
        free(node);
        return NULL;
    }

    node->identifier = identifier;
    node->pattern = pattern;
    node->predicate = predicate;
    node->eval = eval;

    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN_COMPREHENSION, NULL);
    struct pattern_comprehension *node = container_of(self,
            struct pattern_comprehension, _astnode);

    cypher_astnode_t *identifier = (node->identifier == NULL) ? NULL :
            children[child_index(self, node->identifier)];
    cypher_astnode_t *pattern = children[child_index(self, node->pattern)];
    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
            children[child_index(self, node->predicate)];
    cypher_astnode_t *eval = children[child_index(self, node->eval)];

    return cypher_ast_pattern_comprehension(identifier, pattern, predicate,
            eval, children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_pattern_comprehension_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_COMPREHENSION, NULL);
    struct pattern_comprehension *node = container_of(astnode,
            struct pattern_comprehension, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_pattern_comprehension_get_pattern(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_COMPREHENSION, NULL);
    struct pattern_comprehension *node = container_of(astnode,
            struct pattern_comprehension, _astnode);
    return node->pattern;
}


const cypher_astnode_t *cypher_ast_pattern_comprehension_get_predicate(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_COMPREHENSION, NULL);
    struct pattern_comprehension *node = container_of(astnode,
            struct pattern_comprehension, _astnode);
    return node->predicate;
}


const cypher_astnode_t *cypher_ast_pattern_comprehension_get_eval(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN_COMPREHENSION, NULL);
    struct pattern_comprehension *node = container_of(astnode,
            struct pattern_comprehension, _astnode);
    return node->eval;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN_COMPREHENSION, -1);
    struct pattern_comprehension *node = container_of(self,
            struct pattern_comprehension, _astnode);

    size_t n = 0;
    if (n < size)
    {
        str[n] = '[';
    }
    n++;

    ssize_t r;
    if (node->identifier != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "@%u = ",
                node->identifier->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    r = snprintf(str+n, (n < size)? size-n : 0, "@%u ", node->pattern->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->predicate != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "WHERE @%u ",
                node->predicate->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    r = snprintf(str+n, (n < size)? size-n : 0, "| @%u]", node->eval->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    return n;
}
