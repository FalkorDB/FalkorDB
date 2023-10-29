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


struct union_clause
{
    cypher_astnode_t _astnode;
    bool all;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_union_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "UNION",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_union(bool all, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    struct union_clause *node = calloc(1, sizeof(struct union_clause));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_UNION,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->all = all;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNION, false);
    struct union_clause *node =
            container_of(self, struct union_clause, _astnode);

    return cypher_ast_union(node->all, children, self->nchildren, self->range);
}


bool cypher_ast_union_has_all(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_UNION, false);
    struct union_clause *node =
            container_of(astnode, struct union_clause, _astnode);
    return node->all;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_UNION, -1);
    struct union_clause *node =
            container_of(self, struct union_clause, _astnode);
    if (node->all)
    {
        if (size > 0)
        {
            strncpy(str, "ALL", size);
            str[size-1] = '\0';
        }
        return 3;
    }
    return 0;
}
