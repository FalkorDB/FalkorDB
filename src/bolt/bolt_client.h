/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "socket.h"
#include "../redismodule.h"
#include "../util/circular_buffer.h"

typedef enum bolt_client_state {
	BS_NEGOTIATION,
	BS_AUTHENTICATION,
	BS_READY,
	BS_STREAMING,
	BS_TX_READY,
	BS_TX_STREAMING,
	BS_FAILED,
	BS_INTERRUPTED,
	BS_DEFUNCT,
} bolt_client_state;

typedef struct bolt_client_t {
	socket_t socket;
	bolt_client_state state;
	RedisModuleEventLoopFunc on_write;
	uint32_t nwrite;
	uint32_t nread;
	uint32_t nmessage;
	uint32_t last_read_index;
    uint32_t has_message;
	char messasge_buffer[65536];
	char write_buffer[1024];
	char read_buffer[65536];
} bolt_client_t;

typedef struct bolt_version_t {
	uint32_t major;
	uint32_t minor;
} bolt_version_t;

bolt_client_t *bolt_client_new
(
	socket_t socket,
	RedisModuleEventLoopFunc on_write
);

void bolt_change_client_state
(
	bolt_client_t *client   
);

void bolt_client_finish_write
(
	bolt_client_t *client
);

void bolt_client_send
(
	bolt_client_t *client
);

bool bolt_check_handshake
(
	socket_t socket
);

bolt_version_t bolt_read_supported_version
(
	socket_t socket
);
