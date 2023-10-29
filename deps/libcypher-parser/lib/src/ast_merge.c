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


struct merge
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *path;
    unsigned int nactions;
    const cypher_astnode_t *actions[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_merge_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "MERGE",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_merge(const cypher_astnode_t *path,
        cypher_astnode_t * const *actions, unsigned int nactions,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, path, CYPHER_AST_PATTERN_PATH, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, actions, nactions,
            CYPHER_AST_MERGE_ACTION, NULL);

    struct merge *node = calloc(1, sizeof(struct merge) +
            nactions * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_MERGE,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->path = path;
    memcpy(node->actions, actions, nactions * sizeof(cypher_astnode_t *));
    node->nactions = nactions;
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
    REQUIRE_TYPE(self, CYPHER_AST_MERGE, NULL);
    struct merge *node = container_of(self, struct merge, _astnode);

    cypher_astnode_t *path = children[child_index(self, node->path)];
    cypher_astnode_t **actions = calloc(node->nactions,
            sizeof(cypher_astnode_t *));
    if (actions == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nactions; ++i)
    {
        actions[i] = children[child_index(self, node->actions[i])];
    }

    cypher_astnode_t *clone = cypher_ast_merge(path, actions, node->nactions,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(actions);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_merge_get_pattern_path(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MERGE, NULL);
    struct merge *node = container_of(astnode, struct merge, _astnode);
    return node->path;
}


unsigned int cypher_ast_merge_nactions(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MERGE, 0);
    struct merge *node = container_of(astnode, struct merge, _astnode);
    return node->nactions;
}


const cypher_astnode_t *cypher_ast_merge_get_action(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MERGE, NULL);
    struct merge *node = container_of(astnode, struct merge, _astnode);
    if (index >= node->nactions)
    {
        return NULL;
    }
    return node->actions[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MERGE, -1);
    struct merge *node = container_of(self, struct merge, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "path=@%d", node->path->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->nactions > 0)
    {
        if (n < size)
        {
          strncpy(str + n, ", actions=", size-n);
        }
        if (size > 0)
        {
            str[size-1] = '\0';
        }
        n += 8;

        r = snprint_sequence(str+n, (n < size)? size-n : 0,
                node->actions, node->nactions);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
