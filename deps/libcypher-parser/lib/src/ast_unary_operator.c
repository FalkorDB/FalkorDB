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


struct unary_operator
{
    cypher_astnode_t _astnode;
    const cypher_operator_t *op;
    const cypher_astnode_t *arg;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_unary_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "unary operator",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_unary_operator(const cypher_operator_t *op,
        const cypher_astnode_t *arg, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE(op != NULL, NULL);
    REQUIRE_CHILD(children, nchildren, arg, CYPHER_AST_EXPRESSION, NULL);

    struct unary_operator *node = calloc(1, sizeof(struct unary_operator));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_UNARY_OPERATOR,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->op = op;
    node->arg = arg;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNARY_OPERATOR, NULL);
    struct unary_operator *node =
            container_of(self, struct unary_operator, _astnode);

    cypher_astnode_t *arg = children[child_index(self, node->arg)];

    return cypher_ast_unary_operator(node->op, arg, children, self->nchildren,
            self->range);
}


const cypher_operator_t *cypher_ast_unary_operator_get_operator(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_UNARY_OPERATOR, NULL);
    struct unary_operator *node =
        container_of(astnode, struct unary_operator, _astnode);
    return node->op;
}


const cypher_astnode_t *cypher_ast_unary_operator_get_argument(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_UNARY_OPERATOR, NULL);
    struct unary_operator *node =
        container_of(astnode, struct unary_operator, _astnode);
    return node->arg;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNARY_OPERATOR, -1);
    struct unary_operator *node =
        container_of(self, struct unary_operator, _astnode);
    return snprintf(str, size, "%s @%u", node->op->str, node->arg->ordinal);
}
