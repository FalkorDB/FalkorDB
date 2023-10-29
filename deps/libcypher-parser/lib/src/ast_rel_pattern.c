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


struct rel_pattern
{
    cypher_astnode_t _astnode;
    enum cypher_rel_direction direction;
    const cypher_astnode_t *identifier;
    const cypher_astnode_t *varlength;
    const cypher_astnode_t *properties;
    size_t nreltypes;
    const cypher_astnode_t *reltypes[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_rel_pattern_astnode_vt =
    { .name = "rel pattern",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_rel_pattern(enum cypher_rel_direction direction,
        const cypher_astnode_t *identifier, cypher_astnode_t * const *reltypes,
        unsigned int nreltypes, const cypher_astnode_t *properties,
        const cypher_astnode_t *varlength, cypher_astnode_t **children,
        unsigned int nchildren, struct cypher_input_range range)
{
    REQUIRE_CHILD_OPTIONAL(children, nchildren, identifier,
            CYPHER_AST_IDENTIFIER, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, reltypes, nreltypes,
            CYPHER_AST_RELTYPE, NULL);
    REQUIRE(properties == NULL ||
            cypher_astnode_instanceof(properties, CYPHER_AST_MAP) ||
            cypher_astnode_instanceof(properties, CYPHER_AST_PARAMETER), NULL);
    REQUIRE_CONTAINS_OPTIONAL(children, nchildren, properties, NULL);
    REQUIRE_CHILD_OPTIONAL(children, nchildren, varlength,
            CYPHER_AST_RANGE, NULL);

    struct rel_pattern *node = calloc(1, sizeof(struct rel_pattern) +
            nreltypes * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_REL_PATTERN,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->direction = direction;
    node->identifier = identifier;
    memcpy(node->reltypes, reltypes, nreltypes * sizeof(cypher_astnode_t *));
    node->nreltypes = nreltypes;
    node->varlength = varlength;
    node->properties = properties;
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
    REQUIRE_TYPE(self, CYPHER_AST_REL_PATTERN, NULL);
    struct rel_pattern *node = container_of(self, struct rel_pattern, _astnode);

    cypher_astnode_t *identifier = (node->identifier == NULL) ? NULL :
            children[child_index(self, node->identifier)];
    cypher_astnode_t **reltypes =
            calloc(node->nreltypes, sizeof(cypher_astnode_t *));
    if (reltypes == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nreltypes; ++i)
    {
        reltypes[i] = children[child_index(self, node->reltypes[i])];
    }
    cypher_astnode_t *properties = (node->properties == NULL) ? NULL :
            children[child_index(self, node->properties)];
    cypher_astnode_t *varlength = (node->varlength == NULL) ? NULL :
            children[child_index(self, node->varlength)];

    cypher_astnode_t *clone = cypher_ast_rel_pattern(node->direction,
            identifier, reltypes, node->nreltypes, properties, varlength,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(reltypes);
    errno = errsv;
    return clone;
}


enum cypher_rel_direction cypher_ast_rel_pattern_get_direction(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, CYPHER_REL_OUTBOUND);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    return node->direction;
}


const cypher_astnode_t *cypher_ast_rel_pattern_get_identifier(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, NULL);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    return node->identifier;
}


unsigned int cypher_ast_rel_pattern_nreltypes(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, 0);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    return node->nreltypes;
}


const cypher_astnode_t *cypher_ast_rel_pattern_get_reltype(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, NULL);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    if (index >= node->nreltypes)
    {
        return NULL;
    }
    return node->reltypes[index];
}


const cypher_astnode_t *cypher_ast_rel_pattern_get_varlength(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, NULL);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    return node->varlength;
}


const cypher_astnode_t *cypher_ast_rel_pattern_get_properties(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_REL_PATTERN, NULL);
    struct rel_pattern *node =
            container_of(astnode, struct rel_pattern, _astnode);
    return node->properties;
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_REL_PATTERN, -1);
    struct rel_pattern *node = container_of(self, struct rel_pattern, _astnode);

    size_t n = 0;
    ssize_t r = snprintf(str, size, "%s-[",
            (node->direction == CYPHER_REL_INBOUND)? "<" : "");
    if (r < 0)
    {
        return -1;
    }
    n += r;

    if (node->identifier != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "@%u",
                node->identifier->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    for (unsigned int i = 0; i < node->nreltypes; ++i)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "%s@%u",
                (i == 0)? ":" : "|:", node->reltypes[i]->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->varlength != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "*@%u",
                node->varlength->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->properties != NULL)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, " {@%u}",
                node->properties->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    r = snprintf(str+n, (n < size)? size-n : 0, "]-%s",
            (node->direction == CYPHER_REL_OUTBOUND)? ">" : "");
    if (r < 0)
    {
        return -1;
    }
    n += r;

    return n;
}
