/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "buffer.h"
#include "socket.h"
#include "../redismodule.h"

typedef enum bolt_structure_type {
	BST_HELLO = 0x01,                 // hello message from client
	BST_GOODBYE = 0x02,               // goodbye message from client
	BST_RESET = 0x0F,                 // reset message from client
	BST_RUN = 0x10,                   // run query message from client
	BST_BEGIN = 0x11,                 // begin transaction message from client
	BST_COMMIT = 0x12,                // commit transaction message from client
	BST_ROLLBACK = 0x13,              // rollback transaction message from client
	BST_DISCARD = 0x2F,               // discard all message from client
	BST_PULL = 0x3F,                  // pull records message from client
	BST_NODE = 0x4E,                  // node value
	BST_PATH = 0x50,                  // path value
	BST_RELATIONSHIP = 0x52,          // relationship value
	BST_POINT2D = 0x58,               // point2d value
	BST_ROUTE = 0x66,                 // route message from client
	BST_LOGON = 0x6A,                 // logon message from client
	BST_LOGOFF = 0x6B,                // logoff message from client
	BST_SUCCESS = 0x70,               // success message
	BST_RECORD = 0x71,                // record message
	BST_UNBOUND_RELATIONSHIP = 0x72,  // unbound relationship value
	BST_IGNORED = 0x7E,               // ignored message
	BST_FAILURE = 0x7F,               // failure message
} bolt_structure_type;

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

typedef struct bolt_message_t {
	buffer_index_t ws_header; // the websocket header
	buffer_index_t bolt_header; // the bolt header
	buffer_index_t start;  // the start of the message
	buffer_index_t end;    // the end of the message
} bolt_message_t;

typedef struct bolt_client_t {
	socket_t socket;                    // the socket file descriptor
	bolt_client_state state;            // the state of the client
	bool ws;                            // is the connection a websocket
	bool reset;                         // should the connection be reset
	bool shutdown;                      // should the connection be shutdown
    bool processing;                    // is the client processing a message
	buffer_t msg_buf;                   // the message buffer
	buffer_t read_buf;                  // the read buffer
	buffer_t write_buf;                 // the write buffer
	bolt_message_t *write_messages;     // the messages to write
	buffer_index_t ws_frame;            // last websocket frame index
	RedisModuleCtx *ctx;                // the redis module context
	RedisModuleEventLoopFunc on_write;  // the write callback
} bolt_client_t;

typedef struct bolt_version_t {
	uint32_t major;  // the major version
	uint32_t minor;  // the minor version
} bolt_version_t;

// create a new bolt client
bolt_client_t *bolt_client_new
(
	socket_t socket,                   // the socket file descriptor
	RedisModuleCtx *ctx,               // the redis module context
	RedisModuleEventLoopFunc on_write  // the write callback
);

// reply the response type
// and change the client state according to the request and response type
void bolt_client_reply_for
(
	bolt_client_t *client,              // the client
	bolt_structure_type request_type,   // the request type
	bolt_structure_type response_type,  // the response type
	uint32_t size                       // the size of the response structure
);

// finish the current message and prepare for the next
void bolt_client_end_message
(
	bolt_client_t *client  // the client
);

// write all messages to the socket on the main thread
void bolt_client_finish_write
(
	bolt_client_t *client  // the client
);

// write all messages to the socket
void bolt_client_send
(
	bolt_client_t *client  // the client
);

// validate bolt handshake
bool bolt_check_handshake
(
	bolt_client_t *client  // the client
);

// return the latest supported bolt version
bolt_version_t bolt_read_supported_version
(
	bolt_client_t *client  // the client
);

// free the bolt client
void bolt_client_free
(
	bolt_client_t *client  // the client
);
