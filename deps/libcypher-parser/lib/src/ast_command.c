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


struct command
{
    cypher_astnode_t _astnode;
    const cypher_astnode_t *name;
    unsigned int nargs;
    const cypher_astnode_t *args[];
};


static cypher_astnode_t *clone(const cypher_astnode_t *self,
        cypher_astnode_t **children);
static ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size);


const struct cypher_astnode_vt cypher_command_astnode_vt =
    { .name = "command",
      .detailstr = detailstr,
      .release = cypher_astnode_release,
      .clone = clone };


cypher_astnode_t *cypher_ast_command(const cypher_astnode_t *name,
        cypher_astnode_t * const *args, unsigned int nargs,
        cypher_astnode_t **children, unsigned int nchildren,
        struct cypher_input_range range)
{
    REQUIRE_CHILD(children, nchildren, name, CYPHER_AST_STRING, NULL);
    REQUIRE_CHILD_ALL(children, nchildren, args, nargs,
            CYPHER_AST_STRING, NULL);

    struct command *node = calloc(1, sizeof(struct command) +
            nargs * sizeof(cypher_astnode_t *));
    if (node == NULL)
    {
        return NULL;
    }
    if (cypher_astnode_init(&(node->_astnode), CYPHER_AST_COMMAND,
            children, nchildren, range))
    {
        goto cleanup;
    }
    node->name = name;
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
    REQUIRE_TYPE(self, CYPHER_AST_COMMAND, NULL);
    struct command *node = container_of(self, struct command, _astnode);

    cypher_astnode_t *name = children[child_index(self, node->name)];
    cypher_astnode_t **args = calloc(node->nargs, sizeof(cypher_astnode_t *));
    if (args == NULL)
    {
        return NULL;
    }
    for (unsigned int i = 0; i < node->nargs; ++i)
    {
        args[i] = children[child_index(self, node->args[i])];
    }

    cypher_astnode_t *clone = cypher_ast_command(name, args, node->nargs,
            children, self->nchildren, self->range);
    int errsv = errno;
    free(args);
    errno = errsv;
    return clone;
}


const cypher_astnode_t *cypher_ast_command_get_name(
        const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMMAND, NULL);
    struct command *node = container_of(astnode, struct command, _astnode);
    return node->name;
}


unsigned int cypher_ast_command_narguments(const cypher_astnode_t *astnode)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMMAND, 0);
    struct command *node = container_of(astnode, struct command, _astnode);
    return node->nargs;
}


const cypher_astnode_t *cypher_ast_command_get_argument(
        const cypher_astnode_t *astnode, unsigned int index)
{
    REQUIRE_TYPE(astnode, CYPHER_AST_COMMAND, NULL);
    struct command *node = container_of(astnode, struct command, _astnode);
    if (index >= node->nargs)
    {
        return NULL;
    }
    return node->args[index];
}


ssize_t detailstr(const cypher_astnode_t *self, char *str, size_t size)
{
    REQUIRE_TYPE(self, CYPHER_AST_COMMAND, -1);
    struct command *node = container_of(self, struct command, _astnode);
    size_t n = 0;
    ssize_t r = snprintf(str, size, "name=@%u, args=", node->name->ordinal);
    if (r < 0)
    {
        return -1;
    }
    n += r;
    r = snprint_sequence(str+n, (n < size)? size-n : 0,
            node->args, node->nargs);
    if (r < 0)
    {
        return -1;
    }
    return n + r;
}
