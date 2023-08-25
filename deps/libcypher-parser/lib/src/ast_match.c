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


struct match
{
    cypher_astnode_t _astnode;
    bool optional;
    const cypher_astnode_t *pattern;
    const cypher_astnode_t *predicate;
    unsigned int nhints;
    const cypher_astnode_t *hints[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_query_clause_astnode_vt };

const struct cypher_astnode_vt cypher_match_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "MATCH",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_match(bool optional,
        const cypher_astnode_t *pattern, cypher_astnode_t * const *hints,
        unsigned int nhints, const cypher_astnode_t *predicate,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, pattern, CYPHER_AST_PATTERN, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, hints, nhints,
            CYPHER_AST_MATCH_HINT, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, predicate,
            CYPHER_AST_EXPRESSION, NULL);

    struct match *node = calloc(1, sizeof(struct match) +
            nhints * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_MATCH,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->optional = optional;
    node->pattern = pattern;
    memcpy(node->hints, hints, nhints * sizeof(cypher_astnode_t *));
    node->nhints = nhints;
    node->predicate = predicate;
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
    REQUIRE_TYPE(self, CYPHER_AST_MATCH, NULL);
    struct match *node = container_of(self, struct match, _astnode);

    cypher_astnode_t *pattern = children[child_index(self, node->pattern)];
    cypher_astnode_t **hints = calloc(node->nhints,
            sizeof(cypher_astnode_t *));
    if (hints == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nhints; ++i)
    {
        hints[i] = children[child_index(self, node->hints[i])];
    }
    cypher_astnode_t *predicate = (node->predicate == NULL) ? NULL :
            children[child_index(self, node->predicate)];

    cypher_astnode_t *clone = cypher_ast_match(node->optional,
            pattern, hints, node->nhints, predicate, children, self->nchildren,
            self->range);
    int errsv = errno;
    free(hints);
    errno = errsv;
    return clone;
}


bool cypher_ast_match_is_optional(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MATCH, false);
    struct match *node = container_of(astnode, struct match, _astnode);
    return node->optional;
}


const cypher_astnode_t *cypher_ast_match_get_pattern(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MATCH, NULL);
    struct match *node = container_of(astnode, struct match, _astnode);
    return node->pattern;
}


unsigned int cypher_ast_match_nhints(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MATCH, 0);
    struct match *node = container_of(astnode, struct match, _astnode);
    return node->nhints;
}


const cypher_astnode_t *cypher_ast_match_get_hint(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MATCH, NULL);
    struct match *node = container_of(astnode, struct match, _astnode);
    if (index >= node->nhints)
    {
        return NULL;
    }
    return node->hints[index];
}


const cypher_astnode_t *cypher_ast_match_get_predicate(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_MATCH, NULL);
    struct match *node = container_of(astnode, struct match, _astnode);
    return node->predicate;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_MATCH, -1);
    struct match *node = container_of(self, struct match, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "%spattern=@%d",
            node->optional? "OPTIONAL, " : "", node->pattern->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->nhints > 0)
    {
        if (n < size)
        {
          strncpy(str + n, ", hints=", size - n);
        }
        if (size > 0)
        {
            str[size-1] = '\0';
        }
        n += 8;

        r = snprint_sequence(str+n, (n < size)? size-n : 0,
                node->hints, node->nhints);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->predicate != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, ", where=@%u",
                node->predicate->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
