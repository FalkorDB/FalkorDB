/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "buffer.h"
#include "socket.h"
#include "../redismodule.h"

typedef enum bolt_structure_type bolt_structure_type;

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
	RedisModuleCtx *ctx;
	RedisModuleEventLoopFunc on_write;
	bool reset;
	bool shutdown;
    bool processing;
	buffer_index_t write;
	buffer_t read_buf;
	buffer_t write_buf;
	buffer_t msg_buf;
} bolt_client_t;

typedef struct bolt_version_t {
	uint32_t major;
	uint32_t minor;
} bolt_version_t;

bolt_client_t *bolt_client_new
(
	socket_t socket,
	RedisModuleCtx *ctx,
	RedisModuleEventLoopFunc on_write
);

void bolt_client_reply_for
(
	bolt_client_t *client,
	bolt_structure_type request_type,
	bolt_structure_type response_type,
	uint32_t size
);

void bolt_client_end_message
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

void bolt_client_free
(
	bolt_client_t *client
);
