/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ws.h"
#include "bolt.h"
#include "endian.h"
#include "bolt_api.h"
#include "../util/uuid.h"
#include "../commands/commands.h"
#include "../configuration/config.h"
#include <stdarg.h>
#include <strings.h>

rax *clients;
RedisModuleString *BOLT;
RedisModuleString *COMMAND;

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

	bolt_client_reply_for(client, BST_HELLO, BST_SUCCESS, 1);
	bolt_reply_map(client, 2);
	bolt_reply_string(client, "server", 6);
	bolt_reply_string(client, "Neo4j/5.13.0", 12);
	bolt_reply_string(client, "connection_id", 13);
	char *uuid = UUID_New();
	bolt_reply_string(client, uuid, strlen(uuid));
	rm_free(uuid);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

static bool is_authenticated
(
	bolt_client_t *client  // the client that sent the message
) {
	ASSERT(client != NULL);

	uint32_t auth_size = bolt_read_map_size(&client->msg_buf.read);

	if(auth_size < 3) {
		// no credentials provided
		// probe with AUTH to determine whether a password is required:
		//   - "WRONGPASS ..." → requirepass/ACL is active → deny
		//   - "ERR ... no password is set" → no password needed → allow
		RedisModuleCallReply *reply =
			RedisModule_Call(client->ctx, "AUTH", "c", "");
		if(RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_ERROR) {
			RedisModule_FreeCallReply(reply);
			return true;
		}
		size_t err_len;
		const char *err = RedisModule_CallReplyStringPtr(reply, &err_len);
		// WRONGPASS means a password IS required but was not provided
		bool password_required =
			(err_len >= 9 && memcmp(err, "WRONGPASS", 9) == 0);
		RedisModule_FreeCallReply(reply);
		return !password_required;
	}

	uint32_t len;
	char s[64];

	bolt_read_string_size(&client->msg_buf.read, &len);
	if(len >= sizeof(s)) return false;
	bolt_read_string(&client->msg_buf.read, s);
	// check if the first key is scheme
	if(len != 6 || memcmp(s, "scheme", 6) != 0) {
		return false;
	}

	// check if the scheme is basic
	bolt_read_string_size(&client->msg_buf.read, &len);
	if(len >= sizeof(s)) return false;
	bolt_read_string(&client->msg_buf.read, s);
	if(len != 5 || memcmp(s, "basic", 5) != 0) {
		return false;
	}

	// check if the second key is principal
	bolt_read_string_size(&client->msg_buf.read, &len);
	if(len >= sizeof(s)) return false;
	bolt_read_string(&client->msg_buf.read, s);
	if(len != 9 || memcmp(s, "principal", 9) != 0) {
		return false;
	}

	// read the principal (username) — accept any non-empty username
	uint32_t principal_len;
	bolt_read_string_size(&client->msg_buf.read, &principal_len);
	// reject empty or oversized names; the check guarantees principal_len <= sizeof(s)-1
	// so the null terminator below always fits within principal[sizeof(s)]
	if(principal_len == 0 || principal_len >= sizeof(s)) {
		return false;
	}
	// save principal to a dedicated buffer before s is reused for the next key
	char principal[sizeof(s)];
	bolt_read_string(&client->msg_buf.read, principal);
	principal[principal_len] = '\0';

	// check if the third key is credentials
	bolt_read_string_size(&client->msg_buf.read, &len);
	if(len >= sizeof(s)) return false;
	bolt_read_string(&client->msg_buf.read, s);
	if(len != 11 || memcmp(s, "credentials", 11) != 0) {
		return false;
	}

	// check if the credentials are valid
	bolt_read_string_size(&client->msg_buf.read, &len);
	if(len > 1024) return false;
	char credentials[len + 1];
	bolt_read_string(&client->msg_buf.read, credentials);
	credentials[len] = '\0';

	// try ACL-style two-argument AUTH (AUTH username password)
	// this works for Redis ACL users and for Redis 6+ default user
	RedisModuleCallReply *reply = RedisModule_Call(client->ctx, "AUTH", "bb",
		principal, (size_t)principal_len, credentials, (size_t)len);
	if(reply == NULL) return false;
	bool res = RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_ERROR;

	if(!res) {
		// 2-arg AUTH failed — decide whether to fall back to single-arg AUTH
		//
		// when credentials are non-empty and the error is WRONGPASS, the server
		// supports ACL-style AUTH but rejected the credentials; falling back to
		// single-arg AUTH (AUTH <password>) could bypass ACL restrictions by
		// authenticating as the default user, so we refuse in that case
		if(len > 0) {
			size_t err_len;
			const char *err = RedisModule_CallReplyStringPtr(reply, &err_len);
			if(err_len >= 9 && memcmp(err, "WRONGPASS", 9) == 0) {
				RedisModule_FreeCallReply(reply);
				return false;
			}
		}
		RedisModule_FreeCallReply(reply);

		// fall back to single-argument AUTH (for no-password or legacy setups)
		reply = RedisModule_Call(client->ctx, "AUTH", "b",
			credentials, (size_t)len);
		if(reply == NULL) return false;
		res = RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_ERROR;
		if(!res && len == 0) {
			// empty credentials — check if no password is required
			size_t err_len;
			const char *err = RedisModule_CallReplyStringPtr(reply, &err_len);
			res = !(err_len >= 9 && memcmp(err, "WRONGPASS", 9) == 0);
		}
	}
	RedisModule_FreeCallReply(reply);
	return res;
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

	if(is_authenticated(client)) {
		bolt_client_reply_for(client, BST_LOGON, BST_SUCCESS, 1);
		bolt_reply_map(client, 0);
		bolt_client_end_message(client);
		bolt_client_finish_write(client);
	} else {
		bolt_client_reply_for(client, BST_LOGON, BST_FAILURE, 1);
		bolt_reply_map(client, 1);
		bolt_reply_string(client, "code", 4);
		bolt_reply_string(client, "FalkorDB.ClientError.Security.Unauthorized", 42);
		bolt_client_end_message(client);
		bolt_client_finish_write(client);
	}
}

// read the graph name from the message buffer
static RedisModuleString *get_graph_name
(
	RedisModuleCtx *ctx,   // the redis context
	bolt_client_t *client  // the client that sent the message
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	if(bolt_read_map_size(&client->msg_buf.read) == 0) {
		// default graph name
		return RedisModule_CreateString(ctx, "falkordb", 8);
	}
	
	uint32_t graph_name_len;
	bolt_read_string_size(&client->msg_buf.read, &graph_name_len);
	char *graph_name_str = rm_malloc(graph_name_len);
	bolt_read_string(&client->msg_buf.read, graph_name_str);
	RedisModuleString *res = RedisModule_CreateString(ctx, graph_name_str, graph_name_len);
	rm_free(graph_name_str);
	return res;
}

// growable write buffer for parameterized query construction
typedef struct {
	char *buf;      // heap-allocated buffer
	uint32_t n;     // current write offset
	uint32_t cap;   // buffer capacity
	bool err;       // sticky error flag
} wbuf_t;

// ensure at least 'need' more bytes are available
static bool wbuf_ensure
(
	wbuf_t *wb,      // write buffer
	uint32_t need    // additional bytes needed
) {
	uint64_t target = (uint64_t)wb->n + need;
	if(target <= wb->cap) return true;
	if(target > UINT32_MAX) return false;
	uint64_t new_cap = wb->cap;
	while(new_cap < target) {
		new_cap *= 2;
		if(new_cap > UINT32_MAX) return false;
	}
	char *tmp = rm_realloc(wb->buf, (uint32_t)new_cap);
	if(tmp == NULL) return false;
	wb->buf = tmp;
	wb->cap = (uint32_t)new_cap;
	return true;
}

// append formatted string to the buffer, growing as needed
// returns false and sets wb->err on failure
static bool wbuf_printf
(
	wbuf_t *wb,      // write buffer
	const char *fmt,  // format string
	...
) {
	if(wb->err) return false;
	va_list args;
	uint32_t remaining = wb->cap - wb->n;
	va_start(args, fmt);
	int written = vsnprintf(wb->buf + wb->n, remaining, fmt, args);
	va_end(args);
	if(written < 0) { wb->err = true; return false; }
	if((uint32_t)written >= remaining) {
		if(!wbuf_ensure(wb, written + 1)) { wb->err = true; return false; }
		va_start(args, fmt);
		vsnprintf(wb->buf + wb->n, wb->cap - wb->n, fmt, args);
		va_end(args);
	}
	wb->n += written;
	return true;
}

// maximum nesting depth for recursive value serialization
#define WRITE_VALUE_MAX_DEPTH 128

// write the bolt value to the growable buffer as string
// returns false if a write error occurred
static bool _write_value
(
	wbuf_t *wb,            // growable write buffer
	buffer_index_t *value, // the value to write
	int depth              // current recursion depth
) {
	ASSERT(wb != NULL);
	ASSERT(value != NULL);

	if(wb->err) return false;

	if(depth > WRITE_VALUE_MAX_DEPTH) {
		wb->err = true;
		return false;
	}

	switch (bolt_read_type(*value))
	{
		case BVT_NULL:
			bolt_read_null(value);
			wbuf_printf(wb, "NULL");
			return !wb->err;
		case BVT_BOOL:
			wbuf_printf(wb, "%s", bolt_read_bool(value) ? "true" : "false");
			return !wb->err;
		case BVT_INT8:
			wbuf_printf(wb, "%d", bolt_read_int8(value));
			return !wb->err;
		case BVT_INT16:
			wbuf_printf(wb, "%d", bolt_read_int16(value));
			return !wb->err;
		case BVT_INT32:
			wbuf_printf(wb, "%d", bolt_read_int32(value));
			return !wb->err;
		case BVT_INT64:
			wbuf_printf(wb, "%" PRId64, bolt_read_int64(value));
			return !wb->err;
		case BVT_FLOAT:
			wbuf_printf(wb, "%f", bolt_read_float(value));
			return !wb->err;
		case BVT_STRING: {
			uint32_t len;
			bolt_read_string_size(value, &len);
			char *str = rm_malloc(len);
			if(str == NULL) { wb->err = true; return false; }
			bolt_read_string(value, str);
			wbuf_ensure(wb, len + 3);
			wbuf_printf(wb, "'%.*s'", len, str);
			rm_free(str);
			return !wb->err;
		}
		case BVT_LIST: {
			uint32_t size = bolt_read_list_size(value);
			wbuf_printf(wb, "[");
			if(size > 0) {
				_write_value(wb, value, depth + 1);
				for (uint32_t i = 1; i < size; i++) {
					wbuf_printf(wb, ", ");
					_write_value(wb, value, depth + 1);
				}
			}
			wbuf_printf(wb, "]");
			return !wb->err;
		}
		case BVT_MAP: {
			uint32_t size = bolt_read_map_size(value);
			wbuf_printf(wb, "{");
			if(size > 0) {
				uint32_t key_len;
				bolt_read_string_size(value, &key_len);
				char *key = rm_malloc(key_len);
				if(key == NULL) { wb->err = true; return false; }
				bolt_read_string(value, key);
				wbuf_printf(wb, "%.*s: ", key_len, key);
				rm_free(key);
				_write_value(wb, value, depth + 1);
				for (uint32_t i = 1; i < size; i++) {
					bolt_read_string_size(value, &key_len);
					key = rm_malloc(key_len);
					if(key == NULL) { wb->err = true; return false; }
					bolt_read_string(value, key);
					wbuf_printf(wb, ", %.*s: ", key_len, key);
					rm_free(key);
					_write_value(wb, value, depth + 1);
				}
			}
			wbuf_printf(wb, "}");
			return !wb->err;
		}
		case BVT_STRUCTURE:
			if(bolt_read_structure_type(value) == BST_POINT2D) {
				double x = bolt_read_float(value);
				double y = bolt_read_float(value);
				wbuf_printf(wb, "POINT({longitude: %f, latitude: %f})", x, y);
				return !wb->err;
			}
			ASSERT(false);
			return false;
		default:
			ASSERT(false);
			return false;
	}
}

// public entry point — starts recursion at depth 0
static bool write_value
(
	wbuf_t *wb,            // growable write buffer
	buffer_index_t *value  // the value to write
) {
	return _write_value(wb, value, 0);
}

// read the query from the message buffer
RedisModuleString *get_query
(
	RedisModuleCtx *ctx,   // the redis context
	bolt_client_t *client  // the client that sent the message
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	uint32_t query_len;
	bolt_read_string_size(&client->msg_buf.read, &query_len);
	char *query = rm_malloc(query_len);
	if(query == NULL) return NULL;
	bolt_read_string(&client->msg_buf.read, query);
	uint32_t params_count = bolt_read_map_size(&client->msg_buf.read);
	if(params_count > 0) {
		wbuf_t wb;
		wb.cap = query_len + 4096;
		wb.n   = 0;
		wb.buf = rm_malloc(wb.cap);
		wb.err = (wb.buf == NULL);

		wbuf_printf(&wb, "CYPHER ");
		for (uint32_t i = 0; i < params_count; i++) {
			uint32_t key_len;
			bolt_read_string_size(&client->msg_buf.read, &key_len);
			char *key = rm_malloc(key_len);
			if(key == NULL) { wb.err = true; break; }
			bolt_read_string(&client->msg_buf.read, key);
			if(!wbuf_ensure(&wb, key_len + 2)) { rm_free(key); wb.err = true; break; }
			wbuf_printf(&wb, "%.*s=", key_len, key);
			rm_free(key);
			write_value(&wb, &client->msg_buf.read);
			wbuf_printf(&wb, " ");
			if(wb.err) break;
		}
		if(!wb.err) {
			if(!wbuf_ensure(&wb, query_len + 1)) wb.err = true;
		}
		if(!wb.err) {
			wbuf_printf(&wb, "%.*s", query_len, query);
		}
		rm_free(query);
		if(wb.err) {
			rm_free(wb.buf);
			return NULL;
		}
		RedisModuleString *res = RedisModule_CreateString(ctx, wb.buf, wb.n);
		rm_free(wb.buf);
		return res;
	}

	RedisModuleString *res = RedisModule_CreateString(ctx, query, query_len);
	rm_free(query);
	return res;
}

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

	RedisModuleCtx *ctx = client->ctx;
	RedisModuleString *args[5];
	RedisModuleString *query = get_query(ctx, client);
	if(query == NULL) {
		bolt_client_reply_for(client, BST_RUN, BST_FAILURE, 1);
		bolt_reply_map(client, 2);
		bolt_reply_string(client, "code", 4);
		bolt_reply_string(client, "FalkorDB.ClientError", 20);
		bolt_reply_string(client, "message", 7);
		bolt_reply_string(client, "Query construction failed", 25);
		bolt_client_end_message(client);
		bolt_client_finish_write(client);
		return;
	}
	RedisModuleString *graph_name = get_graph_name(ctx, client);

	const char *q = RedisModule_StringPtrLen(query, NULL);
	if(strcmp(q, "SHOW DATABASES") == 0) {
		// "fields":["name","type","aliases","access","address","role","writer","requestedStatus","currentStatus","statusMessage","default","home","constituents"]
		bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
		bolt_reply_map(client, 3);
		bolt_reply_string(client, "t_first", 7);
		bolt_reply_int(client, 0);
		bolt_reply_string(client, "fields", 6);
		bolt_reply_list(client, 13);
		bolt_reply_string(client, "name", 4);
		bolt_reply_string(client, "type", 4);
		bolt_reply_string(client, "aliases", 7);
		bolt_reply_string(client, "access", 6);
		bolt_reply_string(client, "address", 7);
		bolt_reply_string(client, "role", 4);
		bolt_reply_string(client, "writer", 6);
		bolt_reply_string(client, "requestedStatus", 15);
		bolt_reply_string(client, "currentStatus", 13);
		bolt_reply_string(client, "statusMessage", 13);
		bolt_reply_string(client, "default", 7);
		bolt_reply_string(client, "home", 4);
		bolt_reply_string(client, "constituents", 12);
		bolt_reply_string(client, "qid", 3);
		bolt_reply_int(client, 0);
		bolt_client_end_message(client);
		bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
		bolt_reply_list(client, 13);
		// RECORD {"signature":113,"fields":[["neo4j","standard",[],"read-write","localhost:7687","primary",true,"online","online","",true,true,[]]]}
		bolt_reply_string(client, "falkordb", 8);
		bolt_reply_string(client, "standard", 8);
		bolt_reply_list(client, 0);
		bolt_reply_string(client, "read-write", 10);
		bolt_reply_string(client, "localhost:7687", 14);
		bolt_reply_string(client, "primary", 7);
		bolt_reply_bool(client, true);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "", 0);
		bolt_reply_bool(client, true);
		bolt_reply_bool(client, true);
		bolt_reply_list(client, 0);
		bolt_client_end_message(client);
		bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
		bolt_reply_map(client, 0);
		bolt_client_end_message(client);
		bolt_client_finish_write(client);
	} else {
		args[0] = COMMAND;
		args[1] = graph_name;
		args[2] = query;
		args[3] = BOLT;
		args[4] = (RedisModuleString *)client;

		CommandDispatch(ctx, args, 5);
	}

	RedisModule_FreeString(ctx, query);
	RedisModule_FreeString(ctx, graph_name);
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

	bolt_client_reply_for(client, BST_BEGIN, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the COMMIT message
void BoltCommitCommand
(
	bolt_client_t *client  // the client that sent the message
) {
	// The COMMIT message request that the Explicit Transaction is done

	ASSERT(client != NULL);

	bolt_client_reply_for(client, BST_COMMIT, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the ROLLBACK message
void BoltRollbackCommand
(
	bolt_client_t *client
) {
	// The ROLLBACK message requests that the Explicit Transaction rolls back

	ASSERT(client != NULL);

	bolt_client_reply_for(client, BST_ROLLBACK, BST_SUCCESS, 1);
	bolt_reply_map(client, 0);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the ROUTE message
void BoltRouteCommand
(
	bolt_client_t *client
) {
	// TThe ROUTE instructs the server to return the current routing table
	// input:
	// routing::Dictionary,
	// bookmarks::List<String>,
	// extra::Dictionary(
	//   db::String,
	//   imp_user::String,
	// )
	// output:
	// SUCCESS::Dictionary(
	//   rt::Dictionary(
	//     ttl::Integer,
	//     db::String,
	//     servers::List<Dictionary(
	//       addresses::List<String>,
	//       role::String
	//     )>
	//   )
	// )

	ASSERT(client != NULL);

	bolt_client_reply_for(client, BST_ROUTE, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "rt", 2);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "ttl", 3);
	bolt_reply_int(client, 1000);
	bolt_reply_string(client, "db", 2);
	bolt_reply_string(client, "falkordb", 8);
	bolt_reply_string(client, "servers", 7);
	bolt_reply_list(client, 3);
	bolt_reply_map(client, 2);
	bolt_reply_string(client, "addresses", 9);
	bolt_reply_list(client, 1);
	bolt_reply_string(client, "localhost:7687", 14);
	bolt_reply_string(client, "role", 4);
	bolt_reply_string(client, "ROUTE", 5);
	bolt_reply_map(client, 2);
	bolt_reply_string(client, "addresses", 9);
	bolt_reply_list(client, 1);
	bolt_reply_string(client, "localhost:7687", 14);
	bolt_reply_string(client, "role", 4);
	bolt_reply_string(client, "READ", 4);
	bolt_reply_map(client, 2);
	bolt_reply_string(client, "addresses", 9);
	bolt_reply_list(client, 1);
	bolt_reply_string(client, "localhost:7687", 14);
	bolt_reply_string(client, "role", 4);
	bolt_reply_string(client, "WRITE", 5);
	bolt_client_end_message(client);
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
	if(client->processing || buffer_index_length(&client->read_buf.read) <= 2) {
		return;
	}

	// read chunked message
	buffer_index_set(&client->msg_buf.read, &client->msg_buf, 0);
	buffer_index_set(&client->msg_buf.write, &client->msg_buf, 0);
	buffer_index_t current_read = client->read_buf.read;
	if(client->ws && buffer_index_diff(&client->ws_frame, &current_read) == 0) {
		ws_read_frame(&current_read);
		client->ws_frame = current_read;
	}
	uint16_t size = ntohs(buffer_read_uint16(&current_read));
	ASSERT(size > 0);
	while(size > 0) {
		if(buffer_index_length(&current_read) < size) return;
		buffer_read(&current_read, &client->msg_buf.write, size);
		size = ntohs(buffer_read_uint16(&current_read));
	}
	client->read_buf.read = current_read;
	if(buffer_index_length(&client->read_buf.read) == 0) {
		buffer_index_set(&client->read_buf.read, &client->read_buf, 0);
		client->read_buf.write = client->read_buf.read;
		client->ws_frame = client->read_buf.read;
	}

	client->processing = true;

	switch (bolt_read_structure_type(&client->msg_buf.read))
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
			client->processing = false;
			raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
			bolt_client_free(client);
			break;
		case BST_RUN:
			BoltRunCommand(client);
			break;
		case BST_DISCARD:
			break;
		case BST_PULL:
			BoltPullCommand(client);
			client->processing = false;
			BoltRequestHandler(client);
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
			BoltRouteCommand(client);
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
	if(!buffer_socket_read(&client->read_buf, client->socket)) {
		// client disconnected
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		if(client->processing) {
			client->shutdown = true;
			return;
		}
		raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	// process interrupt message
	buffer_index_t current_read = client->read_buf.read;
	while(buffer_index_length(&current_read) > 0) {
		uint16_t size = ntohs(buffer_read_uint16(&current_read));
		if(buffer_index_length(&current_read) < size) break;
		bolt_structure_type request_type = bolt_read_structure_type(&current_read);
		if(request_type == BST_RESET) {
			ASSERT(size == 2);
			client->reset = true;
			uint16_t res = buffer_read_uint16(&current_read);
			ASSERT(res == 0);
			char *src = client->read_buf.chunks[current_read.chunk] + current_read.offset;
			char *dst = src - size - 4;
			size = buffer_index_length(&current_read);
			memmove(dst, src, size);
			current_read.offset -= 6;
			client->read_buf.write.offset -= 6;
			bolt_client_finish_write(client);
		} else {
			size = ntohs(buffer_read_uint16(&current_read));
			while(size > 0) {
				if(buffer_index_length(&current_read) < size) break;
				buffer_index_advance(&current_read, size);
				size = ntohs(buffer_read_uint16(&current_read));
			}
			if(size > 0) break;
		}
	}

	BoltRequestHandler(client);
}

// handle the handshake process
void BoltHandshakeHandler
(
	int fd,           // the socket file descriptor
	void *user_data,  // the client that sent the message
	int mask          // the event mask
) {
	bolt_client_t *client = (bolt_client_t *)user_data;
	if(!buffer_socket_read(&client->read_buf, client->socket)) {
		// client disconnected
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	if(client->ws && ws_read_frame(&client->read_buf.read) != 20) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	if(!bolt_check_handshake(client)) {
		buffer_index_t write;
		buffer_index_set(&write, &client->write_buf, 0);
		buffer_index_t start = write;
		buffer_index_set(&client->read_buf.read, &client->read_buf, 0);
		if(!ws_handshake(&client->read_buf.read, &write)) {
			RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
			raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
			bolt_client_free(client);
			return;
		}
		buffer_socket_write(&start, &write, client->socket);
		client->ws = true;
		buffer_index_set(&client->write_buf.write, &client->write_buf, 0);
		buffer_index_set(&client->read_buf.read, &client->read_buf, 0);
		buffer_index_set(&client->read_buf.write, &client->read_buf, 0);
		return;
	}

	bolt_version_t version = bolt_read_supported_version(client);
	if(version.major == (uint)-1 || version.major == 255) {
		version.major = 5;
		version.minor = 7;
	}
	if(version.major != 5 || version.minor < 1) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	buffer_index_t write;
	buffer_index_set(&write, &client->write_buf, 0);
	buffer_index_t start = write;
	if(client->ws) {
		buffer_write_uint16(&write, htons(0x8204));
	}
	buffer_write_uint16(&write, 0x0000);
	buffer_write_uint8(&write, MIN(version.minor, 7));
	buffer_write_uint8(&write, version.major);
	buffer_socket_write(&start, &write, client->socket);
	buffer_index_set(&client->read_buf.read, &client->read_buf, 0);
	buffer_index_set(&client->read_buf.write, &client->read_buf, 0);

	RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
	RedisModule_EventLoopAdd(fd, REDISMODULE_EVENTLOOP_READABLE, BoltReadHandler, client);
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
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket, sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	bolt_client_send(client);
	client->processing = false;

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

	socket_t socket = socket_accept(fd);
	if(socket == -1) return;

	if(!socket_set_non_blocking(socket)) {
		close(socket);
		return;
	}

	bolt_client_t *client = bolt_client_new(socket, global_ctx, BoltResponseHandler);
	raxInsert(clients, (unsigned char *)&socket, sizeof(socket), client, NULL);
	RedisModule_EventLoopAdd(socket, REDISMODULE_EVENTLOOP_READABLE, BoltHandshakeHandler, client);
}


// checks if bolt is enabled
static bool _bolt_enabled
(
	int16_t *port  // [output] the bolt port
) {
	// get bolt port from configuration
	int16_t p;
	Config_Option_get(Config_BOLT_PORT, &p);

	// bolt is disabled if port is -1
	bool enable = (p != -1);

	// report port if requested
	if(port != NULL) *port = p;

	return enable;
}


// listen on configured bolt port
// in case bolt port is not configured, bolt is disabled
// add the socket to the event loop
int BoltApi_Register
(
    RedisModuleCtx *ctx  // redis context
) {
	int16_t port;

	// quick return if bolt is disabled
	if(!_bolt_enabled(&port)) {
		return REDISMODULE_OK;
	}

	// bolt disabled
	if(port == -1) {
		return REDISMODULE_OK;
	}

    socket_t bolt = socket_bind(port);
	if(bolt == -1) {
		RedisModule_Log(ctx, "warning", "Failed to bind to port %d", port);
		return REDISMODULE_ERR;
	}

	RedisModuleCtx *global_ctx = RedisModule_GetDetachedThreadSafeContext(ctx);

	if(RedisModule_EventLoopAdd(bolt, REDISMODULE_EVENTLOOP_READABLE, BoltAcceptHandler, global_ctx) == REDISMODULE_ERR) {
		RedisModule_Log(ctx, "warning", "Failed to register socket accept handler");
		return REDISMODULE_ERR;
	}
	RedisModule_Log(NULL, "notice", "Bolt protocol initialized. Port: %d", port);

	COMMAND = RedisModule_CreateString(global_ctx, "graph.QUERY", 11);
	BOLT = RedisModule_CreateString(global_ctx, "--bolt", 6);
	clients = raxNew();

    return REDISMODULE_OK;
}

// free connected clients
void BoltApi_Unregister
(
    void
) {
	// quick return if bolt is disabled
	if(!_bolt_enabled(NULL)) return;

	ASSERT(clients != NULL);

	raxIterator iter;
	raxStart(&iter, clients);
	raxSeek(&iter, "^", NULL, 0);
	while(raxNext(&iter)) {
		bolt_client_t *client = iter.data;
		bolt_client_free(client);
	}
	raxStop(&iter);
	raxFree(clients);
}

