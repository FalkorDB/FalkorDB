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


struct range
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *start;
    const cypher_astnode_t *end;
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_range_astnode_vt =
    { .name = "range",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_range(const cypher_astnode_t *start,
        const cypher_astnode_t *end, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD_OPTIONAL(children, nchildren, start, CYPHER_AST_INTEGER, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, end, CYPHER_AST_INTEGER, NULL);

    struct range *node = calloc(1, sizeof(struct range));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_RANGE,
            children, nchildren, range))
    {
        free(node);
        return NULL;
    }
    node->start = start;
    node->end = end;
    return &(node->_astnode);
}


cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children)
{
    REQUIRE_TYPE(self, CYPHER_AST_RANGE, NULL);
    struct range *node = container_of(self, struct range, _astnode);

    cypher_astnode_t *start = (node->start == NULL) ? NULL :
            children[child_index(self, node->start)];
    cypher_astnode_t *end = (node->end == NULL) ? NULL :
            children[child_index(self, node->end)];

    return cypher_ast_range(start, end, children, self->nchildren, self->range);
}


const cypher_astnode_t *cypher_ast_range_get_start(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RANGE, NULL);
    struct range *node = container_of(astnode, struct range, _astnode);
    return node->start;
}


const cypher_astnode_t *cypher_ast_range_get_end(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_RANGE, NULL);
    struct range *node = container_of(astnode, struct range, _astnode);
    return node->end;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_RANGE, -1);
    struct range *node = container_of(self, struct range, _astnode);

    size_t n = 0;
    if (n < size)
    {
        str[n] = '*';
    }
    n++;

    if (node->start == NULL && node->end == NULL)
    {
        return n;
    }

    ssize_t r;
    if (node->start != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "@%u",
                node->start->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (n < size)
    {
        str[n] = '.';
    }
    n++;
    if (n < size)
    {
        str[n] = '.';
    }
    n++;

    if (node->end != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "@%u",
                node->end->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }
    return n;
}
