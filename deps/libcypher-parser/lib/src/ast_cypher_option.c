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


struct cypher_option
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *version;
    unsigned int nparams;
    const cypher_astnode_t *params[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


static const struct cypher_astnode_vt *parents[] =
    { &cypher_statement_option_astnode_vt };

const struct cypher_astnode_vt cypher_cypher_option_astnode_vt =
    { .parents = parents,
      .nparents = 1,
      .name = "CYPHER",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_cypher_option(const cypher_astnode_t *version,
        cypher_astnode_t * const *params, unsigned int nparams,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD_OPTIONAL(children, nchildren, version,
            CYPHER_AST_STRING, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, params, nparams,
            CYPHER_AST_CYPHER_OPTION_PARAM, NULL);

    struct cypher_option *node = calloc(1, sizeof(struct cypher_option) +
            nparams * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_CYPHER_OPTION,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->version = version;

    memcpy(node->params, params, nparams * sizeof(cypher_astnode_t *));
    node->nparams = nparams;
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
    REQUIRE_TYPE(self, CYPHER_AST_CYPHER_OPTION, NULL);
    struct cypher_option *node = container_of(self,
            struct cypher_option, _astnode);

    cypher_astnode_t *version = (node->version == NULL) ? NULL :
            children[child_index(self, node->version)];
    cypher_astnode_t **params = calloc(node->nparams,
            sizeof(cypher_astnode_t *));
    if (params == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nparams; ++i)
    {
        params[i] = children[child_index(self, node->params[i])];
    }

    cypher_astnode_t *clone = cypher_ast_cypher_option(version,
            params, node->nparams, children, self->nchildren, self->range);
    int errsv = errno;
    free(params);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_cypher_option_get_version(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CYPHER_OPTION, NULL);
    struct cypher_option *node = container_of(astnode,
            struct cypher_option, _astnode);
    return node->version;
}


unsigned int cypher_ast_cypher_option_nparams(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CYPHER_OPTION, 0);
    struct cypher_option *node = container_of(astnode,
            struct cypher_option, _astnode);
    return node->nparams;
}


const cypher_astnode_t *cypher_ast_cypher_option_get_param(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_CYPHER_OPTION, NULL);
    struct cypher_option *node = container_of(astnode,
            struct cypher_option, _astnode);
    if (index >= node->nparams)
    {
        return NULL;
    }
    return node->params[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_CYPHER_OPTION, -1);
    struct cypher_option *node =
        container_of(self, struct cypher_option, _astnode);

    size_t n = 0;
    ssize_t r;
    if (node->version != NULL)
    {
        r = snprintf(str, size, "version=@%u", node->version->ordinal);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    if (node->nparams > 0)
    {
        r = snprintf(str+n, (n < size)? size-n : 0, "%sparams=",
                (node->version != NULL)? ", ":"");
        if (r < 0)
        {
            return -1;
        }
        n += r;

        r = snprint_sequence(str+n, (n < size)? size-n : 0, node->params,
                node->nparams);
        if (r < 0)
        {
            return -1;
        }
        n += r;
    }

    return n;
}
