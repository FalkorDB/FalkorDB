/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "socket.h"
#include "../redismodule.h"

typedef enum bolt_value_type {
    BVT_NULL,
    BVT_BOOL,
    BVT_INT8,
    BVT_INT16,
    BVT_INT32,
    BVT_INT64,
    BVT_FLOAT,
    BVT_STRING,
    BVT_BYTES,
    BVT_LIST,
    BVT_MAP,
    BVT_STRUCTURE
} bolt_value_type;

typedef enum bolt_structure_type {
    BST_HELLO = 0x01,
    BST_LOGON = 0x6A,
    BST_LOGOFF = 0x6B,
    BST_GOODBYE = 0x02,
    BST_RESET = 0x0F,
    BST_RUN = 0x10,
    BST_DISCARD = 0x2F,
    BST_PULL = 0x3F,
    BST_BEGIN = 0x11,
    BST_COMMIT = 0x12,
    BST_ROLLBACK = 0x13,
    BST_ROUTE = 0x66,
    BST_SUCCESS = 0x70,
    BST_IGNORED = 0x7E,
    BST_FAILURE = 0x7F,
    BST_RECORD = 0x71,
    BST_NODE = 0x4E,
    BST_RELATIONSHIP = 0x52,
    BST_UNBOUND_RELATIONSHIP = 0x72,
    BST_PATH = 0x50,
    BST_POINT2D = 0x58
} bolt_structure_type;

typedef enum bolt_state {
    BS_NEGOTIATION,
    BS_AUTHENTICATION,
    BS_READY,
    BS_STREAMING,
    BS_TX_READY,
    BS_TX_STREAMING,
    BS_FAILED,
    BS_INTERRUPTED,
    BS_DEFUNCT,
} bolt_state;

typedef struct bolt_client_t {
    socket_t socket;
    bolt_state state;
    RedisModuleEventLoopFunc on_write;
    uint32_t write_index;
    uint32_t read_index;
    char write_buffer[1024];
    char read_buffer[65536];
} bolt_client_t;

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

void bolt_reply_null
(
    bolt_client_t *client
);

void bolt_reply_bool
(
    bolt_client_t *client,
    bool data
);

void bolt_reply_int8
(
    bolt_client_t *client,
    int8_t data
);

void bolt_reply_int16
(
    bolt_client_t *client,
    int16_t data
);

void bolt_reply_int32
(
    bolt_client_t *client,
    int32_t data
);

void bolt_reply_int64
(
    bolt_client_t *client,
    int64_t data
);

void bolt_reply_float
(
    bolt_client_t *client,
    double data
);

void bolt_reply_string
(
    bolt_client_t *client,
    const char *data
);

void bolt_reply_list
(
    bolt_client_t *client,
    uint32_t size
);

void bolt_reply_map
(
    bolt_client_t *client,
    uint32_t size
);

void bolt_reply_structure
(
    bolt_client_t *client,
    bolt_structure_type type,
    uint32_t size
);

typedef struct bolt_version_t {
    uint32_t major;
    uint32_t minor;
} bolt_version_t;

bolt_value_type bolt_read_type
(
    char *data
);

bool bolt_read_bool
(
    char *data
);

int8_t bolt_read_int8
(
    char *data
);

int16_t bolt_read_int16
(
    char *data
);

int32_t bolt_read_int32
(
    char *data
);

int64_t bolt_read_int64
(
    char *data
);

double bolt_read_float
(
    char *data
);

uint32_t bolt_read_string_size
(
    char *data
);

char *bolt_read_string
(
    char *data
);

uint32_t bolt_read_list_size
(
    char *data
);

char *bolt_read_list_item
(
    char *data,
    uint32_t index
);

uint32_t bolt_read_map_size
(
    char *data
);

char *bolt_read_map_key
(
    char *data,
    uint32_t index
);

char *bolt_read_map_value
(
    char *data,
    uint32_t index
);

bolt_structure_type bolt_read_structure_type
(
    char *data
);

uint32_t bolt_read_structure_size
(
    char *data
);

char *bolt_read_structure_value
(
    char *data,
    uint32_t index
);

bool bolt_check_handshake
(
    socket_t socket
);

bolt_version_t bolt_read_supported_version
(
    socket_t socket
);
