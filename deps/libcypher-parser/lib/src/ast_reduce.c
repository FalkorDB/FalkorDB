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
#include "operators.h"
#include "util.h"
#include <assert.h>


struct reduce
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *accumulator;
    const cypher_astnode_t *init;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *eval;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_reduce_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "reduce",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_reduce(const cypher_astnode_t *accumulator,
        const cypher_astnode_t *init, const cypher_astnode_t *identifier,
        const cypher_astnode_t *expression, const cypher_astnode_t *eval,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, accumulator, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, init, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD(children, nchildren, identifier, CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, eval, CYPHER_AST_EXPRESSION, NULL);

    struct reduce *node = calloc(1, sizeof(struct reduce));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_REDUCE,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->accumulator = accumulator;
    node->init = init;
    node->identifier = identifier;
    node->expression = expression;
    node->eval = eval;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(self, struct reduce, _astnode);

    cypher_astnode_t *accumulator = children[child_index(self, node->accumulator)];
    cypher_astnode_t *init = children[child_index(self, node->init)];
    cypher_astnode_t *identifier = children[child_index(self, node->identifier)];
    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *eval = (node->eval == NULL) ? NULL :
            children[child_index(self, node->eval)];

    return cypher_ast_reduce(accumulator, init, identifier, expression, eval,
            children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_reduce_get_accumulator(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(astnode, struct reduce, _astnode);
    return node->accumulator;
}


const cypher_astnode_t *cypher_ast_reduce_get_init(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(astnode, struct reduce, _astnode);
    return node->init;
}


const cypher_astnode_t *cypher_ast_reduce_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(astnode, struct reduce, _astnode);
    return node->identifier;
}


const cypher_astnode_t *cypher_ast_reduce_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(astnode, struct reduce, _astnode);
    return node->expression;
}


const cypher_astnode_t *cypher_ast_reduce_get_eval(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REDUCE, NULL);
    struct reduce *node = container_of(astnode, struct reduce, _astnode);
    return node->eval;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_REDUCE, -1);
    struct reduce *node = container_of(self, struct reduce, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "[@%u=@%u, @%u IN @%u",
            node->accumulator->ordinal,
            node->init->ordinal,
            node->identifier->ordinal,
            node->expression->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->eval != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, " | @%u",
                node->eval->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (n+1 < size)
    {
        str[n] = ']';
        str[n+1] = '\0';
    }

    return n+1;
}
