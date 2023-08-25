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


struct slice_operator
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *expression;
    const cypher_astnode_t *start;
    const cypher_astnode_t *end;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_slice_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "slice",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_slice_operator(const cypher_astnode_t *expression,
        const cypher_astnode_t *start, const cypher_astnode_t *end,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, expression, CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, start,
            CYPHER_AST_EXPRESSION, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, end,
            CYPHER_AST_EXPRESSION, NULL);

    struct slice_operator *node = calloc(1, sizeof(struct slice_operator));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_SLICE_OPERATOR,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->expression = expression;
    node->start = start;
    node->end = end;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_SLICE_OPERATOR, NULL);
    struct slice_operator *node =
            container_of(self, struct slice_operator, _astnode);

    cypher_astnode_t *expression = children[child_index(self, node->expression)];
    cypher_astnode_t *start = (node->start == NULL) ? NULL :
            children[child_index(self, node->start)];
    cypher_astnode_t *end = (node->end == NULL) ? NULL :
            children[child_index(self, node->end)];

    return cypher_ast_slice_operator(expression, start, end, children,
            self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_slice_operator_get_expression(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SLICE_OPERATOR, NULL);
    struct slice_operator *node =
            container_of(astnode, struct slice_operator, _astnode);
    return node->expression;
}


const cypher_astnode_t *cypher_ast_slice_operator_get_start(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SLICE_OPERATOR, NULL);
    struct slice_operator *node =
            container_of(astnode, struct slice_operator, _astnode);
    return node->start;
}


const cypher_astnode_t *cypher_ast_slice_operator_get_end(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_SLICE_OPERATOR, NULL);
    struct slice_operator *node =
            container_of(astnode, struct slice_operator, _astnode);
    return node->end;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_SLICE_OPERATOR, -1);
    struct slice_operator *node =
            container_of(self, struct slice_operator, _astnode);

    size_t n = 0;
    ssize_t w = snprintf(str, size, "@%u[", node->expression->ordinal);
    if (w < 0)
    {
        return -1;
    }
    n += w;

    if (node->start != NULL)
    {
        w = snprintf(str + n, (n < size)? size-n : 0, "@%u",
                node->start->ordinal);
        if (w < 0)
        {
            return -1;
        }
        n += w;
    }

    if (n+1 < size)
    {
        str[n] = '.';
        str[n+1] = '\0';
    }
    n++;

    if (n+1 < size)
    {
        str[n] = '.';
        str[n+1] = '\0';
    }
    n++;

    if (node->end != NULL)
    {
        w = snprintf(str + n, (n < size)? size-n : 0, "@%u",
                node->end->ordinal);
        if (w < 0)
        {
            return -1;
        }
        n += w;
    }

    if (n+1 < size)
    {
        str[n] = ']';
        str[n+1] = '\0';
    }
    n++;

    return n;
}
