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


struct pattern
{
    cypher_astnode_t _astnode;
    size_t npaths;
    const cypher_astnode_t *paths[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_pattern_astnode_vt =
    { .name = "pattern",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_pattern(cypher_astnode_t * const *paths,
        unsigned int npaths, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE(npaths > 0, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, paths, npaths,
            CYPHER_AST_PATTERN_PATH, NULL);

    struct pattern *node = calloc(1, sizeof(struct pattern) +
            npaths * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_PATTERN,
                children, nchildren, range))
    {
        goto cleanup;
    }
    memcpy(node->paths, paths, npaths * sizeof(cypher_astnode_t *));
    node->npaths = npaths;
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
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN, NULL);
    struct pattern *node = container_of(self, struct pattern, _astnode);

    cypher_astnode_t **paths = calloc(node->npaths, sizeof(cypher_astnode_t *));
    if (paths == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->npaths; ++i)
    {
        paths[i] = children[child_index(self, node->paths[i])];
    }

    cypher_astnode_t *clone = cypher_ast_pattern(paths, node->npaths,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(paths);
    errno = errsv;
    return clone;
}


unsigned int cypher_ast_pattern_npaths(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN, 0);
    struct pattern *node = container_of(astnode, struct pattern, _astnode);
    return node->npaths;
}


const cypher_astnode_t *cypher_ast_pattern_get_path(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_PATTERN, NULL);
    struct pattern *node = container_of(astnode, struct pattern, _astnode);
    if (index >= node->npaths)
    {
        return NULL;
    }
    return node->paths[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_PATTERN, -1);
    struct pattern *node = container_of(self, struct pattern, _astnode);

    strncpy(str, "paths=", size);
    if (size > 0)
    {
        str[size-1] = '\0';
    }
    size_t n = 6;

    ssize_t r = snprint_sequence(str+n, (n < size)? size-n : 0,
            node->paths, node->npaths);
    if (r < 0)
    {
        return -1;
    }
    return n+r;
}
