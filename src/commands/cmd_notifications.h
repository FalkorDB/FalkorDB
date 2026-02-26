/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// utility for deferring keyspace notifications to the main Redis thread
// Redis 8 requires pubsub (used by NotifyKeyspaceEvent) to run on the main
// thread, so we schedule notifications via EventLoopAddOneShot

#pragma once

// schedule a "graph.modified" keyspace notification on the main Redis thread
void Notify_Keyspace_GraphModified
(
	const char *graph_name
);

// schedule a "graph.copy_to" keyspace notification on the main Redis thread
void Notify_Keyspace_GraphCopyTo
(
	const char *graph_name
);
