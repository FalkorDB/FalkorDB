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


struct unwind
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *alias;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_unwind_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "UNWIND",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_unwind(const cypher_astnode_t *expression,
        const cypher_astnode_t *alias, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD(children, nchildren, alias, CYPHER_AST_IDENTIFIER, NULL);

    struct unwind *node = calloc(1, sizeof(struct unwind));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_UNWIND,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    node->alias = alias;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNWIND, NULL);
    struct unwind *node = container_of(self, struct unwind, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *alias = children[child_index(self, node->alias)];

    return cypher_ast_unwind(expression, alias, children, self->nchildren,
            self->range);
}


const cypher_astnode_t *cypher_ast_unwind_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_UNWIND, NULL);
    struct unwind *node =
        container_of(astnode, struct unwind, _astnode);
    return node->expression;
}


const cypher_astnode_t *cypher_ast_unwind_get_alias(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_UNWIND, NULL);
    struct unwind *node =
        container_of(astnode, struct unwind, _astnode);
    return node->alias;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNWIND, -1);
    struct unwind *node =
        container_of(self, struct unwind, _astnode);
    return snprintf(str, size, "expression=@%u, alias=@%u",
            node->expression->ordinal, node->alias->ordinal);
}
