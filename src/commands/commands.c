/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "commands.h"

// convert from string representation to an enum
GRAPH_Commands CommandFromString(const char *cmd_name) {
	if (!strcasecmp(cmd_name, "graph.INFO"))     return CMD_INFO;
	if (!strcasecmp(cmd_name, "graph.LIST"))     return CMD_LIST;
	if (!strcasecmp(cmd_name, "graph.QUERY"))    return CMD_QUERY;
	if (!strcasecmp(cmd_name, "graph.DEBUG"))    return CMD_DEBUG;
	if (!strcasecmp(cmd_name, "graph.EFFECT"))   return CMD_EFFECT;
	if (!strcasecmp(cmd_name, "graph.DELETE"))   return CMD_DELETE;
	if (!strcasecmp(cmd_name, "graph.CONFIG"))   return CMD_CONFIG;
	if (!strcasecmp(cmd_name, "graph.PROFILE"))  return CMD_PROFILE;
	if (!strcasecmp(cmd_name, "graph.EXPLAIN"))  return CMD_EXPLAIN;
	if (!strcasecmp(cmd_name, "graph.SLOWLOG"))  return CMD_SLOWLOG;
	if (!strcasecmp(cmd_name, "graph.RO_QUERY")) return CMD_RO_QUERY;
	if (!strcasecmp(cmd_name, "graph.BULK"))     return CMD_BULK_INSERT;

	// we shouldn't reach this point
	ASSERT(false);
	return CMD_UNKNOWN;
}

