/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "bolt_api.h"
#include "../commands/commands.h"

#include <string.h>

void BoltHelloCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "server");
	bolt_reply_string(client, "Neo4j/5.11.0");
	bolt_client_finish_write(client);
}

void BoltLogonCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

void BoltResetCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

RedisModuleString *get_db
(
	RedisModuleCtx *ctx,
	bolt_client_t *client
) {
	char *db = bolt_read_structure_value(client->read_buffer + client->current_message_index, 2);
	char *db_str;
	size_t db_len;
	if(bolt_read_map_size(db) == 0) {
		db_str = "falkordb";
		db_len = 8;
	} else {
		db = bolt_read_map_value(db, 0);
		db_str = bolt_read_string(db);
		db_len = bolt_read_string_size(db);
	}
	return RedisModule_CreateString(ctx, db_str, db_len);
}

int print_parameter_value
(
	char *buff,
	char *value
) {
	switch (bolt_read_type(value))
	{
		case BVT_NULL:
			return sprintf(buff, "NULL");
		case BVT_BOOL:
			return sprintf(buff, "%s", bolt_read_bool(value) ? "true" : "false");
		case BVT_INT8:
			return sprintf(buff, "%d", bolt_read_int8(value));
		case BVT_INT16:
			return sprintf(buff, "%d", bolt_read_int16(value));
		case BVT_INT32:
			return sprintf(buff, "%d", bolt_read_int32(value));
		case BVT_INT64:
			return sprintf(buff, "%lld", bolt_read_int64(value));
		case BVT_FLOAT:
			return sprintf(buff, "%f", bolt_read_float(value));
		case BVT_STRING:
			return sprintf(buff, "'%.*s'", bolt_read_string_size(value), bolt_read_string(value));
		case BVT_LIST: {
			uint32_t size = bolt_read_list_size(value);
			int n = 0;
			n += sprintf(buff, "[");
			if(size > 0) {
				char *item = bolt_read_list_item(value, 0);
				n += print_parameter_value(buff + n, item);
				for (int i = 1; i < size - 0; i++) {
					n += sprintf(buff + n, ", ");
					item = bolt_read_list_item(value, i);
					n += print_parameter_value(buff + n, item);
				}
			}
			n += sprintf(buff + n, "]");
			return n;
		}
		case BVT_MAP: {
			uint32_t size = bolt_read_map_size(value);
			int n = 0;
			n += sprintf(buff, "{");
			if(size > 0) {
				char *key = bolt_read_map_key(value, 0);
				char *val = bolt_read_map_value(value, 0);
				n += sprintf(buff + n, "%.*s: ", bolt_read_string_size(key), bolt_read_string(key));
				n += print_parameter_value(buff + n, val);
				for (int i = 1; i < size - 0; i++) {
					n += sprintf(buff + n, ", ");
					key = bolt_read_map_key(value, i);
					val = bolt_read_map_value(value, i);
					n += sprintf(buff + n, "%.*s: ", bolt_read_string_size(key), bolt_read_string(key));
					n += print_parameter_value(buff + n, val);
				}
			}
			n += sprintf(buff + n, "}");
			return n;
		}
		case BVT_STRUCTURE:
			if(bolt_read_structure_type(value) == BST_POINT2D) {
				char *x = bolt_read_structure_value(value, 1);
				char *y = bolt_read_structure_value(value, 2);
				sprintf(buff, "POINT({longitude: %f, latitude: %f})", bolt_read_float(x), bolt_read_float(y));
				break;
			}
			ASSERT(false);
			break;
		default:
			ASSERT(false);
			break;
	}
}

RedisModuleString *get_query
(
	RedisModuleCtx *ctx,
	bolt_client_t *client
) {
	char *query = bolt_read_structure_value(client->read_buffer + client->current_message_index, 0);
	char *parameters = bolt_read_structure_value(client->read_buffer + client->current_message_index, 1);
	uint32_t params_count = bolt_read_map_size(parameters);
	int n = 0;
	if(params_count > 0) {
		char prametrize_query[1024];
		n += sprintf(prametrize_query, "CYPHER ");
		for (int i = 0; i < params_count; i++) {
			char *key = bolt_read_map_key(parameters, i);
			char *value = bolt_read_map_value(parameters, i);
			n += sprintf(prametrize_query + n, "%.*s=", bolt_read_string_size(key), bolt_read_string(key));
			n += print_parameter_value(prametrize_query + n, value);
			n += sprintf(prametrize_query + n, " ");
		}
		sprintf(prametrize_query + n, "%.*s", bolt_read_string_size(query), bolt_read_string(query));
		return RedisModule_CreateString(ctx, prametrize_query, strlen(prametrize_query));
	}

	return RedisModule_CreateString(ctx, bolt_read_string(query), bolt_read_string_size(query));
}

void BoltRunCommand
(
	bolt_client_t *client
) {
	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
	RedisModuleString *args[5];

	args[0] = RedisModule_CreateString(ctx, "graph.QUERY", 11);
	args[1] = get_db(ctx, client);
	args[2] = get_query(ctx, client);
	args[3] = RedisModule_CreateString(ctx, "--bolt", 6);
	args[4] = RedisModule_CreateStringFromLongLong(ctx, (long long)client);

	CommandDispatch(ctx, args, 5);

	RedisModule_FreeString(ctx, args[0]);
	RedisModule_FreeString(ctx, args[1]);
	RedisModule_FreeString(ctx, args[2]);
	RedisModule_FreeString(ctx, args[3]);
	RedisModule_FreeString(ctx, args[4]);
	RedisModule_FreeThreadSafeContext(ctx);
}

void BoltPullCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

void BoltBeginCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

void BoltCommitCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

void BoltRollbackCommand
(
	bolt_client_t *client
) {
	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

void BoltRequestHandler
(
	bolt_client_t *client
) {
	if(client->current_message_index != -1 || client->nmessages == 0) {
		return;
	}

	client->current_message_index = client->messages[0];

	switch (bolt_read_structure_type(client->read_buffer + client->current_message_index))
	{
		case BST_HELLO:
			BoltHelloCommand(client);
			break;
		case BST_LOGON:
			BoltLogonCommand(client);
			break;
		case BST_LOGOFF:
			break;
		case BST_GOODBYE:
			break;
		case BST_RESET:
			BoltResetCommand(client);
			break;
		case BST_RUN:
			BoltRunCommand(client);
			break;
		case BST_DISCARD:
			break;
		case BST_PULL:
			BoltPullCommand(client);
			break;
		case BST_BEGIN:
			BoltBeginCommand(client);
			break;
		case BST_COMMIT:
			BoltCommitCommand(client);
			break;
		case BST_ROLLBACK:
			BoltRollbackCommand(client);
			break;
		case BST_ROUTE:
			break;
		default:
			break;
	}
}

void BoltReadHandler
(
	int fd,
	void *user_data,
	int mask
) {
	bolt_client_t *client = (bolt_client_t*)user_data;
	int nread = socket_read(client->socket, client->read_buffer + client->nread, 65536 - client->nread);
	if(nread == 0) {
		rm_free(client);
		socket_close(fd);
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		return;
	} else {
		client->nread += nread;
	}

	while(client->nread - client->last_message_index > 2) {
		uint16_t size = ntohs(*(uint16_t*)(client->read_buffer + client->last_message_index));
		if(client->last_message_index + size <= client->nread) {
			client->messages[client->nmessages++] = client->last_message_index + 2;
			client->last_message_index += size + 4;
		}
	}

	BoltRequestHandler(client);
}

void BoltResponseHandler
(
	int fd,
	void *user_data,
	int mask
) {
	RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_WRITABLE);
	bolt_client_t *client = (bolt_client_t*)user_data;
	bolt_client_send(client);
	uint16_t size = ntohs(*(uint16_t*)(client->read_buffer + client->current_message_index));
	client->nmessages--;
	memmove(client->messages, client->messages + 1, client->nmessages * sizeof(uint32_t));
	client->current_message_index = -1;
	BoltRequestHandler(client);
}

void BoltAcceptHandler
(
	int fd,
	void *user_data,
	int mask
) {
	socket_t client = socket_accept(fd);
	if(client == -1) return;

	if(!bolt_check_handshake(client)) {
		socket_close(client);
		return;
	}

	bolt_version_t version = bolt_read_supported_version(client);

	char data[4];
	data[0] = 0x00;
	data[1] = 0x00;
	data[2] = version.minor;
	data[3] = version.major;
	socket_write(client, data, 4);

	bolt_client_t *bolt_client = bolt_client_new(client, BoltResponseHandler);
	RedisModule_EventLoopAdd(client, REDISMODULE_EVENTLOOP_READABLE, BoltReadHandler, bolt_client);
}

int BoltApi_Register
(
    RedisModuleCtx *ctx
) {
    socket_t bolt = socket_bind(7687);
	if(bolt == -1) {
		RedisModule_Log(ctx, "warning", "Failed to bind to port 7687");
		return REDISMODULE_ERR;
	}

	if(RedisModule_EventLoopAdd(bolt, REDISMODULE_EVENTLOOP_READABLE, BoltAcceptHandler, NULL) == REDISMODULE_ERR) {
		RedisModule_Log(ctx, "warning", "Failed to register socket accept handler");
		return REDISMODULE_ERR;
	}
	RedisModule_Log(NULL, "notice", "bolt server initialized");
    return REDISMODULE_OK;
}