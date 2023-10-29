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


struct comparison
{
    cypher_astnode_t _astnode;
    unsigned int length;
    const cypher_operator_t **ops;
    const cypher_astnode_t *args[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);
static void comparison_release(cypher_astnode_t *self);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_comparison_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "comparison",
      .detailstr = detailstr,
      .release = comparison_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_comparison(unsigned int length,
        const cypher_operator_t * const *ops, cypher_astnode_t * const *args,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE(length > 0, NULL);
    REQUIRE(ops != NULL, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, args, length+1,
            CYPHER_AST_EXPRESSION, NULL);

    struct comparison *node = calloc(1, sizeof(struct comparison) +
            (length + 1) * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_COMPARISON,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->length = length;
    node->ops = mdup(ops, length * sizeof(cypher_astnode_t *));
    if (node->ops == NULL)
    {
        goto cleanup;
    }
    memcpy(node->args, args, (length+1) * sizeof(cypher_astnode_t *));
    return &(node->_astnode);

    int errsv;
cleanup:
    errsv = errno;
    free(node->ops);
    free(node);
    errno = errsv;
    return NULL;
}


void comparison_release(cypher_astnode_t *self)
{
    struct comparison *node = container_of(self, struct comparison, _astnode);
    free(node->ops);
    cypher_astnode_release(self);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_COMPARISON, NULL);
    struct comparison *node = container_of(self, struct comparison, _astnode);

    cypher_astnode_t **args = calloc(node->length + 1,
            sizeof(cypher_astnode_t *));
    if (args == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->length + 1; ++i)
    {
        args[i] = children[child_index(self, node->args[i])];
    }

    cypher_astnode_t *clone = cypher_ast_comparison(node->length, node->ops,
            args, children, self->nchildren, self->range);
    int errsv = errno;
    free(args);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_comparison_get_length(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMPARISON, 0);
    struct comparison *node = container_of(astnode, struct comparison, _astnode);
    return node->length;
}


const cypher_operator_t *cypher_ast_comparison_get_operator(
        const cypher_astnode_t *astnode, unsigned int pos)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMPARISON, NULL);
    struct comparison *node = container_of(astnode, struct comparison, _astnode);
    if (pos >= node->length)
    {
        return NULL;
    }
    return node->ops[pos];
}


const cypher_astnode_t *cypher_ast_comparison_get_argument(
        const cypher_astnode_t *astnode, unsigned int pos)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMPARISON, NULL);
    struct comparison *node = container_of(astnode, struct comparison, _astnode);
    if (pos > node->length)
    {
        return NULL;
    }
    return node->args[pos];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_COMPARISON, -1);
    struct comparison *node = container_of(self, struct comparison, _astnode);

    size_t n = 0;
    for (unsigned int i = 0; i < node->length; ++i)
    {
        ssize_t r = snprintf(str + n, (n < size)? size-n : 0, "@%u %s ",
                node->args[i]->ordinal, node->ops[i]->str);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    ssize_t r = snprintf(str + n, (n < size)? size-n : 0, "@%u",
            node->args[node->length]->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    return n;
}
