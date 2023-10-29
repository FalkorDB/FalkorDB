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


struct using_periodic_commit
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *limit;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_option_astnode_vt };

const struct cypher_astnode_vt cypher_using_periodic_commit_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "USING PERIODIC_COMMIT",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_using_periodic_commit(
        const cypher_astnode_t *limit, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD_OPTIONAL(children, nchildren, limit, CYPHER_AST_INTEGER, NULL);

    struct using_periodic_commit *node =
            calloc(1, sizeof(struct using_periodic_commit));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_USING_PERIODIC_COMMIT,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->limit = limit;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_USING_PERIODIC_COMMIT, NULL);
    struct using_periodic_commit *node =
            container_of(self, struct using_periodic_commit, _astnode);

    cypher_astnode_t *limit = (node->limit == NULL) ? NULL :
            children[child_index(self, node->limit)];

    return cypher_ast_using_periodic_commit(limit, children, self->nchildren,
            self->range);
}


const cypher_astnode_t *cypher_ast_using_periodic_commit_get_limit(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_USING_PERIODIC_COMMIT, NULL);
    struct using_periodic_commit *node =
            container_of(astnode, struct using_periodic_commit, _astnode);
    return node->limit;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_USING_PERIODIC_COMMIT, -1);
    struct using_periodic_commit *node =
            container_of(self, struct using_periodic_commit, _astnode);
    if (node->limit == NULL)
    {
        return 0;
    }
    return snprintf(str, size, "limit=@%u", node->limit->ordinal);
}
