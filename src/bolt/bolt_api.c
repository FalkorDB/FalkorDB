/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "endian.h"
#include "bolt_api.h"
#include "../commands/commands.h"

// handle the HELLO message
static void BoltHelloCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The HELLO message request the connection to be authorized for use with the remote database
	// input:
	// extra::Dictionary(
	//   user_agent::String,
	//   patch_bolt::List<String>,
	//   routing::Dictionary(address::String),
	//   notifications_minimum_severity::String,
	//   notifications_disabled_categories::List<String>,
	//   bolt_agent::Dictionary(
	//     product::String,
	//     platform::String,
	//     language::String,
	//     language_details::String
	//   )
	// )
	// output:
	// SUCCESS::Dictionary(
	//   server::String,
	//   connection_id::String,
	// }

	ASSERT(client != NULL);
	ASSERT(client->state == BS_NEGOTIATION);

	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 2);
	bolt_reply_string(client, "server");
	bolt_reply_string(client, "Neo4j/5.11.0");
	bolt_reply_string(client, "connection_id");
	bolt_reply_string(client, "bolt-connection-1");
	bolt_client_finish_write(client);
}

// handle the LOGON message
static void BoltLogonCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// A LOGON message carries an authentication request
	// input:
	// auth::Dictionary(
	//   scheme::String,
	//   ...
	// )
	// when schema:
	// 	 basic: principal::String and credentials::String required
	//   bearer: credentials::String required
	// output:
	// SUCCESS

	ASSERT(client != NULL);
	ASSERT(client->state == BS_AUTHENTICATION);

	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

// read the graph name from the message buffer
static RedisModuleString *get_graph_name
(
	RedisModuleCtx *ctx,   // the redis context
	bolt_client_t *client  // the client that sent the message
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	char *graph_name = bolt_read_structure_value(client->messasge_buffer, 2);
	char *graph_name_str;
	size_t graph_name_len;
	if(bolt_read_map_size(graph_name) == 0) {
		// default graph name
		graph_name_str = "falkordb";
		graph_name_len = 8;
	} else {
		graph_name = bolt_read_map_value(graph_name, 0);
		graph_name_str = bolt_read_string(graph_name);
		graph_name_len = bolt_read_string_size(graph_name);
	}
	return RedisModule_CreateString(ctx, graph_name_str, graph_name_len);
}

// write the bolt value to the buffer as string
int write_value
(
	char *buff,  // the buffer to write to
	char *value  // the value to write
) {
	ASSERT(buff != NULL);
	ASSERT(value != NULL);

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
				n += write_value(buff + n, item);
				for (int i = 1; i < size - 0; i++) {
					n += sprintf(buff + n, ", ");
					item = bolt_read_list_item(value, i);
					n += write_value(buff + n, item);
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
				n += write_value(buff + n, val);
				for (int i = 1; i < size - 0; i++) {
					n += sprintf(buff + n, ", ");
					key = bolt_read_map_key(value, i);
					val = bolt_read_map_value(value, i);
					n += sprintf(buff + n, "%.*s: ", bolt_read_string_size(key), bolt_read_string(key));
					n += write_value(buff + n, val);
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

// read the query from the message buffer
RedisModuleString *get_query
(
	RedisModuleCtx *ctx,   // the redis context
	bolt_client_t *client  // the client that sent the message
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	char *query = bolt_read_structure_value(client->messasge_buffer, 0);
	uint32_t query_len = bolt_read_string_size(query);
	query = bolt_read_string(query);
	char *parameters = bolt_read_structure_value(client->messasge_buffer, 1);
	uint32_t params_count = bolt_read_map_size(parameters);
	if(params_count > 0) {
		char prametrize_query[1024];
		int n = sprintf(prametrize_query, "CYPHER ");
		for (int i = 0; i < params_count; i++) {
			char *key = bolt_read_map_key(parameters, i);
			char *value = bolt_read_map_value(parameters, i);
			uint32_t key_len = bolt_read_string_size(key);
			key = bolt_read_string(key);
			n += sprintf(prametrize_query + n, "%.*s=", key_len, key);
			n += write_value(prametrize_query + n, value);
			n += sprintf(prametrize_query + n, " ");
		}
		n += sprintf(prametrize_query + n, "%.*s", query_len, query);
		return RedisModule_CreateString(ctx, prametrize_query, n);
	}

	return RedisModule_CreateString(ctx, query, query_len);
}

RedisModuleString *COMMAND;
RedisModuleString *BOLT;

// handle the RUN message
void BoltRunCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The RUN message requests that a Cypher query is executed with a set of parameters and additional extra data
	// input:
	// query::String,
	// parameters::Dictionary,
	// extra::Dictionary(
	//   bookmarks::List<String>,
	//   tx_timeout::Integer,
	//   tx_metadata::Dictionary,
	//   mode::String,
	//   db:String,
	//   imp_user::String,
	//   notifications_minimum_severity::String,
	//   notifications_disabled_categories::List<String>
	// )

	ASSERT(client != NULL);

	client->pull = false;
	RedisModuleCtx *ctx = client->ctx;
	RedisModuleString *args[5];

	args[0] = COMMAND;
	args[1] = get_graph_name(ctx, client);
	args[2] = get_query(ctx, client);
	args[3] = BOLT;
	args[4] = RedisModule_CreateString(ctx, (const char *)&client, sizeof(bolt_client_t*));

	CommandDispatch(ctx, args, 5);

	RedisModule_FreeString(ctx, args[1]);
	RedisModule_FreeString(ctx, args[2]);
	RedisModule_FreeString(ctx, args[4]);
}

// handle the PULL message
void BoltPullCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The PULL message requests data from the remainder of the result stream
	// input:
	// extra::Dictionary{
	//   n::Integer,
	//   qid::Integer,
	// }

	ASSERT(client != NULL);

	pthread_mutex_lock(&client->pull_condv_mutex);
	client->pull = true;
	pthread_cond_signal(&client->pull_condv);
	pthread_mutex_unlock(&client->pull_condv_mutex);
}

// handle the BEGIN message
void BoltBeginCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The BEGIN message request the creation of a new Explicit Transaction
	// input:
	// extra::Dictionary(
	//   bookmarks::List<String>,
	//   tx_timeout::Integer,
	//   tx_metadata::Dictionary,
	//   mode::String,
	//   db::String,
	//   imp_user::String,
	//   notifications_minimum_severity::String,
	//   notifications_disabled_categories::List<String>
	// )

	ASSERT(client != NULL);

	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

// handle the COMMIT message
void BoltCommitCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The COMMIT message request that the Explicit Transaction is done

	ASSERT(client != NULL);

	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

// handle the ROLLBACK message
void BoltRollbackCommand
(
	bolt_client_t *client
) {
	// The ROLLBACK message requests that the Explicit Transaction rolls back

	ASSERT(client != NULL);

	bolt_reply_structure(client, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_finish_write(client);
}

// process next message from the client
void BoltRequestHandler
(
	bolt_client_t *client // the client that sent the message
) {
	ASSERT(client != NULL);

	// if there is a message already in process or
	// not enough data to read the message
	if(client->has_message || client->nread - client->last_read_index <= 2) {
		return;
	}

	// read chunked message
	uint32_t last_read_index = client->last_read_index;
	uint32_t nmessage = 0;
	uint16_t size = ntohs(*(uint16_t*)(client->read_buffer + last_read_index));
	ASSERT(size > 0);
	while(size > 0) {
		if(last_read_index + 2 + size > client->nread) return;
		memcpy(client->messasge_buffer + nmessage, client->read_buffer + last_read_index + 2, size);
		nmessage += size;
		last_read_index += size + 2;
		size = ntohs(*(uint16_t*)(client->read_buffer + last_read_index));
	}
	client->last_read_index = last_read_index + 2;
	if(client->last_read_index == client->nread) {
		client->last_read_index = 0;
		client->nread = 0;
	}

	client->has_message = true;

	switch (bolt_read_structure_type(client->messasge_buffer))
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

// read data from socket to client buffer
void BoltReadHandler
(
	int fd,           // the socket file descriptor
	void *user_data,  // the client that sent the message
	int mask          // the event mask
) {
	ASSERT(fd != -1);
	ASSERT(user_data != NULL);

	bolt_client_t *client = (bolt_client_t*)user_data;
	int nread = socket_read(client->socket, client->read_buffer + client->nread, UINT16_MAX - client->nread);
	if(nread == -1) {
		// error
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		if(client->has_message) {
			client->shutdown = true;
			return;
		}
		rm_free(client);
		socket_close(fd);
		return;
	}
	if(nread == 0) {
		if(client->nread == UINT16_MAX) {
			// not enough space in the buffer
			// try to move the message to the beginning of the buffer
			if(client->has_message) {
				return;
			}
			memmove(client->read_buffer, client->read_buffer + client->last_read_index, client->nread - client->last_read_index);
			client->nread -= client->last_read_index;
			client->last_read_index = 0;
			return;
		}
		// client disconnected
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		if(client->has_message) {
			client->shutdown = true;
			return;
		}
		rm_free(client);
		socket_close(fd);
		return;
	} else {
		client->nread += nread;
	}

	// process interrupt message
	uint32_t last_read_index = client->last_read_index;
	while(last_read_index < client->nread) {
		uint16_t size = ntohs(*(uint16_t*)(client->read_buffer + last_read_index));
		if(last_read_index + 2 + size > client->nread) break;
		bolt_structure_type request_type = bolt_read_structure_type(client->read_buffer + last_read_index + 2);
		if(request_type == BST_RESET) {
			client->reset = true;
			memmove(client->read_buffer + last_read_index, client->read_buffer + last_read_index + size + 4, client->nread - last_read_index - size - 4);
			client->nread -= size + 4;
			bolt_client_finish_write(client);
		}
		last_read_index += size + 2;
		size = ntohs(*(uint16_t*)(client->read_buffer + last_read_index));
		while(size > 0) {
			if(last_read_index + 2 + size > client->nread) break;
			last_read_index += size + 2;
			size = ntohs(*(uint16_t*)(client->read_buffer + last_read_index));
		}
		if(size > 0) break;
		last_read_index += 2;
	}

	BoltRequestHandler(client);
}

// write data from client buffer to socket
void BoltResponseHandler
(
	int fd,           // the socket file descriptor
	void *user_data,  // the client that sent the message
	int mask          // the event mask
) {
	ASSERT(fd != -1);
	ASSERT(user_data != NULL);

	bolt_client_t *client = (bolt_client_t*)user_data;

	if(client->shutdown) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_WRITABLE);
		rm_free(client);
		socket_close(fd);
		return;
	}

	bolt_client_send(client);
	client->has_message = false;

	RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_WRITABLE);

	BoltRequestHandler(client);
}

// handle new client connection
void BoltAcceptHandler
(
	int fd,           // the socket file descriptor
	void *user_data,  // the client that sent the message
	int mask          // the event mask
) {
	ASSERT(fd != -1);
	ASSERT(user_data != NULL);

	RedisModuleCtx *global_ctx = (RedisModuleCtx*)user_data;

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

	bolt_client_t *bolt_client = bolt_client_new(client, global_ctx, BoltResponseHandler);
	RedisModule_EventLoopAdd(client, REDISMODULE_EVENTLOOP_READABLE, BoltReadHandler, bolt_client);
}

// listen to bolt port 7687
// add the socket to the event loop
int BoltApi_Register
(
    RedisModuleCtx *ctx  // redis context
) {
    socket_t bolt = socket_bind(7687);
	if(bolt == -1) {
		RedisModule_Log(ctx, "warning", "Failed to bind to port 7687");
		return REDISMODULE_ERR;
	}

	RedisModuleCtx *global_ctx = RedisModule_GetDetachedThreadSafeContext(ctx);

	if(RedisModule_EventLoopAdd(bolt, REDISMODULE_EVENTLOOP_READABLE, BoltAcceptHandler, global_ctx) == REDISMODULE_ERR) {
		RedisModule_Log(ctx, "warning", "Failed to register socket accept handler");
		return REDISMODULE_ERR;
	}
	RedisModule_Log(NULL, "notice", "bolt server initialized");

	COMMAND = RedisModule_CreateString(global_ctx, "graph.QUERY", 11);
	BOLT = RedisModule_CreateString(global_ctx, "--bolt", 6);

    return REDISMODULE_OK;
}