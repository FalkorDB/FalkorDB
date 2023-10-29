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


struct apply_all_operator
{
    cypher_astnode_t _astnode;
    bool distinct;
    const cypher_astnode_t *func_name;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_expression_astnode_vt };

const struct cypher_astnode_vt cypher_apply_all_operator_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "apply all",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_apply_all_operator(
        const cypher_astnode_t *func_name, bool distinct,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, func_name,
            CYPHER_AST_FUNCTION_NAME, NULL);

    struct apply_all_operator *node =
            calloc(1, sizeof(struct apply_all_operator));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_APPLY_ALL_OPERATOR,
                children, nchildren, range))
    {
        goto cleanup;
    }
    node->distinct = distinct;
    node->func_name = func_name;
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
    REQUIRE_TYPE(self, CYPHER_AST_APPLY_ALL_OPERATOR, NULL);
    struct apply_all_operator *node =
        container_of(self, struct apply_all_operator, _astnode);

    cypher_astnode_t *func_name = children[child_index(self, node->func_name)];

    return cypher_ast_apply_all_operator(func_name, node->distinct,
            children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_apply_all_operator_get_func_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_ALL_OPERATOR, NULL);
    struct apply_all_operator *node =
            container_of(astnode, struct apply_all_operator, _astnode);
    return node->func_name;
}


bool cypher_ast_apply_all_operator_get_distinct(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_APPLY_ALL_OPERATOR, NULL);
    struct apply_all_operator *node =
            container_of(astnode, struct apply_all_operator, _astnode);
    return node->distinct;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_APPLY_ALL_OPERATOR, -1);
    struct apply_all_operator *node =
        container_of(self, struct apply_all_operator, _astnode);

    return snprintf(str, size, "@%u(%s*)", node->func_name->ordinal,
            node->distinct? "DISTINCT " : "");
}
