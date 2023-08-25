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


struct apply_operator
{
    cypher_astnode_t _astnode;
    bool distinct;
    const cypher_astnode_t *func_name;
    unsigned int nargs;
    const cypher_astnode_t *args[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_apply_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "apply",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_apply_operator(const cypher_astnode_t *func_name,
        bool distinct, cypher_astnode_t * const *args, unsigned int nargs,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, func_name,
            CYPHER_AST_FUNCTION_NAME, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, args, nargs,
            CYPHER_AST_EXPRESSION, NULL);

    struct apply_operator *node = calloc(1, sizeof(struct apply_operator) +
            nargs * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_APPLY_OPERATOR,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->distinct = distinct;
    node->func_name = func_name;

    memcpy(node->args, args, nargs * sizeof(cypher_astnode_t *));
    node->nargs = nargs;
    return &(node->_astnode);

    int errsv;
cleanup:
    errsv = errno;
    free(node);
    errno = errsv;
    return NULL;
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_APPLY_OPERATOR, NULL);
    struct apply_operator *node =
        container_of(self, struct apply_operator, _astnode);

    cypher_astnode_t *func_name = children[child_index(self, node->func_name)];
    cypher_astnode_t **args = calloc(node->nargs, sizeof(cypher_astnode_t *));
    if (args == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nargs; ++i)
    {
        args[i] = children[child_index(self, node->args[i])];
    }

    cypher_astnode_t *clone = cypher_ast_apply_operator(func_name,
            node->distinct, args, node->nargs, children, self->nchildren,
            self->range);
    int errsv = errno;
    free(args);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_apply_operator_get_func_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_OPERATOR, NULL);
    struct apply_operator *node =
            container_of(astnode, struct apply_operator, _astnode);
    return node->func_name;
}


bool cypher_ast_apply_operator_get_distinct(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_OPERATOR, false);
    struct apply_operator *node =
            container_of(astnode, struct apply_operator, _astnode);
    return node->distinct;
}


unsigned int cypher_ast_apply_operator_narguments(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_OPERATOR, 0);
    struct apply_operator *node =
            container_of(astnode, struct apply_operator, _astnode);
    return node->nargs;
}


const cypher_astnode_t *cypher_ast_apply_operator_get_argument(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_OPERATOR, NULL);
    struct apply_operator *node =
            container_of(astnode, struct apply_operator, _astnode);

    if (index >= node->nargs)
    {
        return NULL;
    }
    return node->args[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_APPLY_OPERATOR, -1);
    struct apply_operator *node =
        container_of(self, struct apply_operator, _astnode);

    ssize_t r = snprintf(str, size, "@%u(%s", node->func_name->ordinal,
            node->distinct? "DISTINCT " : "");
    if (r < 0)
    {
        return -1;
    }
    size_t n = r;

    for (unsigned int i = 0; i < node->nargs; )
    {
        ssize_t r = snprintf(str+n, (n < size)? size-n : 0,
                "@%u", node->args[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
        if (++i < node->nargs)
        {
            if (n < size)
            {
                str[n] = ',';
            }
            n++;
            if (n < size)
            {
                str[n] = ' ';
            }
            n++;
        }
    }

    if (n+1 < size)
    {
        str[n] = ')';
    }
    n++;
    return n;
}
