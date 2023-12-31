/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "ws.h"
#include "bolt.h"
#include "endian.h"
#include "globals.h"
#include "bolt_api.h"
#include "../util/uuid.h"
#include "../commands/commands.h"
#include "../configuration/config.h"

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
	bolt_reply_string(client, "Neo4j/5.14.0", 12);
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

	uint32_t auth_size;
	if(!bolt_read_map_size(&client->msg_buf.read, &auth_size)) {
		return false;
	}

	if(auth_size < 3) {
		// if no password provided check we can call PING
		RedisModuleCallReply *reply = RedisModule_Call(client->ctx, "PING", "");
		bool res = RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_ERROR;
		RedisModule_FreeCallReply(reply);
		return res;
	}

	uint32_t len;
	char s[64];
	if(!bolt_read_string_size(&client->msg_buf.read, &len)) {
		return false;
	}
	if(!bolt_read_string(&client->msg_buf.read, s)) {
		return false;
	}
	// check if the first key is scheme
	if(strncmp(s, "scheme", len) != 0) {
		return false;
	}
	
	// check if the scheme is basic
	if(!bolt_read_string_size(&client->msg_buf.read, &len)) {
		return false;
	}
	if(!bolt_read_string(&client->msg_buf.read, s)) {
		return false;
	}
	if(strncmp(s, "basic", len) != 0) {
		return false;
	}

	// check if the second key is principal
	if(!bolt_read_string_size(&client->msg_buf.read, &len)) {
		return false;
	}
	if(!bolt_read_string(&client->msg_buf.read, s)) {
		return false;
	}
	if(strncmp(s, "principal", len) != 0) {
		return false;
	}
	
	// check if the principal is falkordb
	uint32_t principal_len;
	if(!bolt_read_string_size(&client->msg_buf.read, &principal_len)) {
		return false;
	}
	if(principal_len > 64) {
		return false;
	}
	if(!bolt_read_string(&client->msg_buf.read, s)) {
		return false;
	}
	if(strncmp(s, "falkordb", principal_len) != 0) {
		return false;
	}
	
	// check if the third key is credentials
	if(!bolt_read_string_size(&client->msg_buf.read, &len)) {
		return false;
	}
	if(!bolt_read_string(&client->msg_buf.read, s)) {
		return false;
	}
	if(strncmp(s, "credentials", len) != 0) {
		return false;
	}
	
	// check if the credentials are valid
	if(!bolt_read_string_size(&client->msg_buf.read, &len)) {
		return false;
	}
	char credentials[len];
	if(!bolt_read_string(&client->msg_buf.read, credentials)) {
		return false;
	}
	RedisModuleCallReply *reply = RedisModule_Call(client->ctx, "AUTH", "b", credentials, len);
	bool res = RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_ERROR;
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
static bool get_graph_name
(
	RedisModuleCtx *ctx,            // the redis context
	bolt_client_t *client,          // the client that sent the message
	RedisModuleString **graph_name  // the graph name
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	uint32_t size;
	if(!bolt_read_map_size(&client->msg_buf.read, &size)) {
		return false;
	}
	if(size == 0) {
		// default graph name
		*graph_name = RedisModule_CreateString(ctx, "falkordb", 8);
		return true;
	}
	
	uint32_t graph_name_len;
	if(!bolt_read_string_size(&client->msg_buf.read, &graph_name_len)) {
		return false;
	}
	char *graph_name_str = rm_malloc(graph_name_len);
	if(!bolt_read_string(&client->msg_buf.read, graph_name_str)) {
		rm_free(graph_name_str);
		return false;
	}
	RedisModuleString *res = RedisModule_CreateString(ctx, graph_name_str, graph_name_len);
	rm_free(graph_name_str);
	*graph_name = res;
	return true;
}

// write the bolt value to the buffer as string
bool write_value
(
	char *buff,             // the buffer to write to
	buffer_index_t *value,  // the value to write
	int *n 				    // the number of bytes written
) {
	ASSERT(buff != NULL);
	ASSERT(value != NULL);

	bolt_value_type type;
	bolt_read_type(*value, &type);
	switch(type)
	{
		case BVT_NULL:
			if(!bolt_read_null(value)) {
				return false;
			}
			*n += sprintf(buff + *n, "NULL");
			return true;
		case BVT_BOOL: {
			bool b;
			if(!bolt_read_bool(value, &b)) {
				return false;
			}
			*n += sprintf(buff + *n, "%s", b ? "true" : "false");
			return true;
		}
		case BVT_INT8: {
			int8_t i8;
			if(!bolt_read_int8(value, &i8)) {
				return false;
			}
			*n += sprintf(buff + *n, "%d", i8);
			return true;
		}
		case BVT_INT16: {
			int16_t i16;
			if(!bolt_read_int16(value, &i16)) {
				return false;
			}
			*n += sprintf(buff + *n, "%d", i16);
			return true;
		}
		case BVT_INT32: {
			int32_t i32;
			if(!bolt_read_int32(value, &i32)) {
				return false;
			}
			*n += sprintf(buff + *n, "%d", i32);
			return true;
		}
		case BVT_INT64: {
			int64_t i64;
			if(!bolt_read_int64(value, &i64)) {
				return false;
			}
			*n += sprintf(buff + *n, "%lld", i64);
			return true;
		}
		case BVT_FLOAT: {
			double f;
			if(!bolt_read_float(value, &f)) {
				return false;
			}
			*n += sprintf(buff + *n, "%f", f);
			return true;
		}
		case BVT_STRING: {
			uint32_t len;
			if(!bolt_read_string_size(value, &len)) {
				return false;
			}
			char str[len];
			if(!bolt_read_string(value, str)) {
				return false;
			}
			*n += sprintf(buff + *n, "'%.*s'", len, str);
			return true;
		}
		case BVT_LIST: {
			uint32_t size;
			if(!bolt_read_list_size(value, &size)) {
				return false;
			}
			*n += sprintf(buff + *n, "[");
			if(size > 0) {
				if(!write_value(buff, value, n)) {
					return false;
				}
				for (int i = 1; i < size; i++) {
					*n += sprintf(buff + *n, ", ");
					if(!write_value(buff, value, n)) {
						return false;
					}
				}
			}
			*n += sprintf(buff + *n, "]");
			return true;
		}
		case BVT_MAP: {
			uint32_t size;
			if(!bolt_read_map_size(value, &size)) {
				return false;
			}
			*n += sprintf(buff + *n, "{");
			if(size > 0) {
				uint32_t key_len;
				if(!bolt_read_string_size(value, &key_len)) {
					return false;
				}
				char key[key_len];
				if(!bolt_read_string(value, key)) {
					return false;
				}
				*n += sprintf(buff + *n, "%.*s: ", key_len, key);
				if(!write_value(buff, value, n)) {
					return false;
				}
				for (int i = 1; i < size; i++) {
					if(!bolt_read_string_size(value, &key_len)) {
						return false;
					}
					char key[key_len];
					if(!bolt_read_string(value, key)) {
						return false;
					}
					n += sprintf(buff + *n, ", ");
					n += sprintf(buff + *n, "%.*s: ", key_len, key);
					if(!write_value(buff, value, n)) {
						return false;
					}
				}
			}
			*n += sprintf(buff + *n, "}");
			return true;
		}
		case BVT_STRUCTURE: {
			bolt_structure_type type;
			if(!bolt_read_structure_type(value, &type)) {
				return false;
			}
			if(type == BST_POINT2D) {
				double x, y;
				if(!bolt_read_float(value, &x) || !bolt_read_float(value, &y)) {
					return false;
				}
				*n += sprintf(buff + *n, "POINT({longitude: %f, latitude: %f})", x, y);
				return true;
			}
			ASSERT(false);
			return false;
		}
		default:
			ASSERT(false);
			return false;
	}
}

// read the query from the message buffer
bool get_query
(
	RedisModuleCtx *ctx,       // the redis context
	bolt_client_t *client,     // the client that sent the message
	RedisModuleString **query  // the query string
) {
	ASSERT(ctx != NULL);
	ASSERT(client != NULL);

	uint32_t query_len;
	if(!bolt_read_string_size(&client->msg_buf.read, &query_len)) {
		return false;
	}
	char *query_str = rm_malloc(query_len);
	if(!bolt_read_string(&client->msg_buf.read, query_str)) {
		rm_free(query_str);
		return false;
	}
	uint32_t params_count;
	if(!bolt_read_map_size(&client->msg_buf.read, &params_count)) {
		rm_free(query_str);
		return false;
	}
	if(params_count > 0) {
		char prametrize_query[4096];
		int n = sprintf(prametrize_query, "CYPHER ");
		for (int i = 0; i < params_count; i++) {
			uint32_t key_len;
			if(!bolt_read_string_size(&client->msg_buf.read, &key_len)) {
				rm_free(query_str);
				return false;
			}
			char key[key_len];
			if(!bolt_read_string(&client->msg_buf.read, key)) {
				rm_free(query_str);
				return false;
			}
			n += sprintf(prametrize_query + n, "%.*s=", key_len, key);
			if(!write_value(prametrize_query, &client->msg_buf.read, &n)) {
				rm_free(query_str);
				return false;
			}
			n += sprintf(prametrize_query + n, " ");
		}
		n += sprintf(prametrize_query + n, "%.*s", query_len, query_str);
		rm_free(query_str);
		*query = RedisModule_CreateString(ctx, prametrize_query, n);
		return true;
	}

	*query = RedisModule_CreateString(ctx, query_str, query_len);
	rm_free(query_str);
	return true;
}

// handle the SHOW DATABASES query
static void BoltShowDatabases
(
	bolt_client_t *client  // the client that sent the message
) {
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

	KeySpaceGraphIterator it;
	GraphContext *gc = NULL;
	Globals_ScanGraphs(&it);
	int i = 0;
	char address[32];
	int16_t p;
	Config_Option_get(Config_BOLT_PORT, &p);
	int len = sprintf(address, "localhost:%d", p);
	while((gc = GraphIterator_Next(&it)) != NULL) {
		i++;
		// RECORD {"signature":113,"fields":[["falkordb","standard",[],"read-write","localhost:7687","primary",true,"online","online","",true,true,[]]]}
		bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
		bolt_reply_list(client, 13);
		bolt_reply_string(client, gc->graph_name, strlen(gc->graph_name));
		bolt_reply_string(client, "standard", 8);
		bolt_reply_list(client, 0);
		bolt_reply_string(client, "read-write", 10);
		bolt_reply_string(client, address, len);
		bolt_reply_string(client, "primary", 7);
		bolt_reply_bool(client, true);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "", 0);
		bolt_reply_bool(client, true);
		bolt_reply_bool(client, true);
		bolt_reply_list(client, 0);
		bolt_client_end_message(client);

		GraphContext_DecreaseRefCount(gc);
	}

	if(i == 0) {
		// RECORD {"signature":113,"fields":[["falkordb","standard",[],"read-write","localhost:7687","primary",true,"online","online","",true,true,[]]]}
		bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
		bolt_reply_list(client, 13);
		bolt_reply_string(client, "falkordb", 8);
		bolt_reply_string(client, "standard", 8);
		bolt_reply_list(client, 0);
		bolt_reply_string(client, "read-write", 10);
		bolt_reply_string(client, address, len);
		bolt_reply_string(client, "primary", 7);
		bolt_reply_bool(client, true);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "online", 6);
		bolt_reply_string(client, "", 0);
		bolt_reply_bool(client, true);
		bolt_reply_bool(client, true);
		bolt_reply_list(client, 0);
		bolt_client_end_message(client);
	}

	// RECORD {"signature":113,"fields":[["aviavni","standard",[],"read-write","localhost:7687","primary",true,"online","online","",true,true,[]]]}
	bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
	bolt_reply_list(client, 13);
	bolt_reply_string(client, "system", 7);
	bolt_reply_string(client, "system", 8);
	bolt_reply_list(client, 0);
	bolt_reply_string(client, "read-write", 10);
	bolt_reply_string(client, address, len);
	bolt_reply_string(client, "primary", 7);
	bolt_reply_bool(client, true);
	bolt_reply_string(client, "online", 6);
	bolt_reply_string(client, "online", 6);
	bolt_reply_string(client, "", 0);
	bolt_reply_bool(client, false);
	bolt_reply_bool(client, false);
	bolt_reply_list(client, 0);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the CALL dbms.clientConfig() query
void BoltDbmsClientConfig
(
	bolt_client_t *client  // the client that sent the message
) {
	// "fields":["name","versions","edition"],"qid":{"low":0,"high":0}}]}
	bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "t_first", 7);
	bolt_reply_int(client, 0);
	bolt_reply_string(client, "fields", 6);
	bolt_reply_list(client, 2);
	bolt_reply_string(client, "name", 4);
	bolt_reply_string(client, "value", 8);
	bolt_reply_string(client, "qid", 3);
	bolt_reply_int(client, 0);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the CALL dbms.security.showCurrentUser() query
void BoltDbmsSecurityShowCurremtUser
(
	bolt_client_t *client  // the client that sent the message
) {
	// "fields":["name","versions","edition"],"qid":{"low":0,"high":0}}]}
	bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "t_first", 7);
	bolt_reply_int(client, 0);
	bolt_reply_string(client, "fields", 6);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "username", 8);
	bolt_reply_string(client, "roles", 5);
	bolt_reply_string(client, "flags", 5);
	bolt_reply_string(client, "qid", 3);
	bolt_reply_int(client, 0);
	bolt_client_end_message(client);

	bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "falkordb", 7);
	bolt_reply_null(client);
	bolt_reply_list(client, 0);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the CALL dbms.components() YIELD name, versions, edition query
void BoltDbmsComponents
(
	bolt_client_t *client  // the client that sent the message
) {
	// "fields":["name","versions","edition"],"qid":{"low":0,"high":0}}]}
	bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "t_first", 7);
	bolt_reply_int(client, 0);
	bolt_reply_string(client, "fields", 6);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "name", 4);
	bolt_reply_string(client, "versions", 8);
	bolt_reply_string(client, "edition", 7);
	bolt_reply_string(client, "qid", 3);
	bolt_reply_int(client, 0);
	bolt_client_end_message(client);

	// RECORD {"signature":113,"fields":[["Neo4j Kernel",["5.14.0"],"community"]]}
	bolt_client_reply_for(client, BST_PULL, BST_RECORD, 1);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "Neo4j Kernel", 12);
	bolt_reply_list(client, 1);
	bolt_reply_string(client, "5.14.0", 6);
	bolt_reply_string(client, "community", 9);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the SHOW PROCEDURES yield name, description, signature query
void BoltShowProcedures
(
	bolt_client_t *client  // the client that sent the message
) {
	// "fields":["name","versions","edition"],"qid":{"low":0,"high":0}}]}
	bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "t_first", 7);
	bolt_reply_int(client, 0);
	bolt_reply_string(client, "fields", 6);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "name", 4);
	bolt_reply_string(client, "description", 11);
	bolt_reply_string(client, "signature", 9);
	bolt_reply_string(client, "qid", 3);
	bolt_reply_int(client, 0);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
}

// handle the SHOW FUNCTIONS yield name, description, signature query
void BoltShowFunctions
(
	bolt_client_t *client  // the client that sent the message
) {
	// "fields":["name","versions","edition"],"qid":{"low":0,"high":0}}]}
	bolt_client_reply_for(client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(client, 3);
	bolt_reply_string(client, "t_first", 7);
	bolt_reply_int(client, 0);
	bolt_reply_string(client, "fields", 6);
	bolt_reply_list(client, 3);
	bolt_reply_string(client, "name", 4);
	bolt_reply_string(client, "description", 11);
	bolt_reply_string(client, "signature", 9);
	bolt_reply_string(client, "qid", 3);
	bolt_reply_int(client, 0);
	bolt_client_end_message(client);

	// SUCCESS {}
	bolt_client_reply_for(client, BST_PULL, BST_SUCCESS, 1);
	bolt_reply_map(client, 1);
	bolt_reply_string(client, "t_last", 6);
	bolt_reply_int8(client, 1);
	bolt_client_end_message(client);
	bolt_client_finish_write(client);
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
	RedisModuleString *query;
	if(!get_query(ctx, client, &query)) {
		bolt_client_free(client);
		return;
	}
	RedisModuleString *graph_name;
	if(!get_graph_name(ctx, client, &graph_name)) {
		RedisModule_FreeString(ctx, query);
		bolt_client_free(client);
		return;
	}
 
	const char *q = RedisModule_StringPtrLen(query, NULL);
	if(strcmp(q, "SHOW DATABASES") == 0) {
		BoltShowDatabases(client);
	} else if(strcmp(q, "CALL dbms.clientConfig()") == 0) {
		BoltDbmsClientConfig(client);
	} else if(strcmp(q, "CALL dbms.security.showCurrentUser()") == 0 || strcmp(q, "CALL dbms.showCurrentUser()") == 0) {
		BoltDbmsSecurityShowCurremtUser(client);
	} else if(strcmp(q, "CALL dbms.components() YIELD name, versions, edition") == 0) {
		BoltDbmsComponents(client);
	} else if(strcmp(q, "SHOW PROCEDURES yield name, description, signature") == 0) {
		BoltShowProcedures(client);
	} else if(strcmp(q, "SHOW FUNCTIONS yield name, description, signature") == 0) {
		BoltShowFunctions(client);
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

static bool _BoltProcessRawMessage
(
	bolt_client_t *client
) {
	int len = array_len(client->read_messages);
	buffer_index_t current_read = client->read_buf.read;

	while(buffer_index_length(&current_read) > 0) {
		uint16_t size;
		bolt_message_t msg;
		msg.bolt_header = current_read;

		// skip to next message
		if(!buffer_read(&current_read, &size)) break;
		msg.start = current_read;

		while(size > 0) {
			size = ntohs(size);
			if(!buffer_index_advance(&current_read, size)) break;

			if(!buffer_read(&current_read, &size)) break;
		}

		if(size > 0) break;

		msg.end = current_read;
		array_append(client->read_messages, msg);
		client->read_buf.read = current_read;
	}

	for (int i = len; i < array_len(client->read_messages); i++) {
		bolt_message_t *msg = client->read_messages + i;
		bolt_structure_type request_type;
		buffer_index_t current_request = msg->start;
		if(!bolt_read_structure_type(&current_request, &request_type)) {
			raxRemove(clients, (unsigned char *)&client->socket,
					sizeof(client->socket), NULL);
			bolt_client_free(client);
			return false;
		}

		if(request_type == BST_RESET) {
			client->reset = true;
			array_del(client->read_messages, i);
			bolt_client_finish_write(client);
			return false;
		} 
	}

	return array_len(client->read_messages) > 0;
}

// process next message from the client
void BoltRequestHandler
(
	bolt_client_t *client // the client that sent the message
) {
	ASSERT(client != NULL);

	// if there is a message already in process or
	// not enough data to read the message
	if(client->processing || !_BoltProcessRawMessage(client)) {
		return;
	}

	//--------------------------------------------------------------------------
	// read chunked message
	//--------------------------------------------------------------------------

	// reset message buffer
	buffer_index_set(&client->msg_buf.read, &client->msg_buf, 0);
	buffer_index_set(&client->msg_buf.write, &client->msg_buf, 0);

	bolt_message_t msg = client->read_messages[0];
	array_del(client->read_messages, 0);
	buffer_index_t current_read = msg.bolt_header;
	if(client->ws && buffer_index_diff(&client->ws_frame, &current_read) == 0) {
		uint64_t payload_length;
		if(!ws_read_frame(&current_read, &payload_length)) {
			return;
		}
		client->ws_frame = current_read;
	}

	uint16_t size;
	if(!buffer_read(&current_read, &size)) {
		return;
	}

	size = ntohs(size);
	ASSERT(size > 0);

	//--------------------------------------------------------------------------
	// read message from READ buffer
	//--------------------------------------------------------------------------

	while(size > 0) {
		// copy message from READ buffer into the message buffer
		if(!buffer_copy(&current_read, &client->msg_buf.write, size)) {
			return;
		}

		// see if there is more data in the READ buffer
		if(!buffer_read(&current_read, &size)) {
			return;
		}

		// a message ends with 0x00 0x00
		size = ntohs(size);
	}

	client->processing = true;

	bolt_structure_type type;
	if(!bolt_read_structure_type(&client->msg_buf.read, &type)) {
		raxRemove(clients, (unsigned char *)&client->socket,
				sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	// handle message
	switch(type)
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
			RedisModule_EventLoopDel(client->socket, REDISMODULE_EVENTLOOP_READABLE);
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

	//--------------------------------------------------------------------------
	// read data from socket to client buffer
	//--------------------------------------------------------------------------

	if(!buffer_socket_read(&client->read_buf, client->socket)) {
		// client disconnected
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket,
				sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	uint64_t payload_length;
	if(client->ws && (!ws_read_frame(&client->read_buf.read, &payload_length) ||
				payload_length != 20)) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket,
				sizeof(client->socket), NULL);
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
			raxRemove(clients, (unsigned char *)&client->socket,
					sizeof(client->socket), NULL);
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

	bolt_version_t version;
	if(!bolt_read_supported_version(client, &version)) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket,
				sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	if(version.major != 5 || version.minor < 1) {
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		raxRemove(clients, (unsigned char *)&client->socket,
				sizeof(client->socket), NULL);
		bolt_client_free(client);
		return;
	}

	buffer_index_t write;
	buffer_index_set(&write, &client->write_buf, 0);
	buffer_index_t start = write;

	if(client->ws) {
		buffer_write_uint16_t(&write, htons(0x8204));
	}

	buffer_write_uint16_t(&write, 0x0000);
	buffer_write_uint8_t(&write, version.minor);
	buffer_write_uint8_t(&write, version.major);
	buffer_socket_write(&start, &write, client->socket);

	RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
	RedisModule_EventLoopAdd(fd, REDISMODULE_EVENTLOOP_READABLE, BoltReadHandler, client);

	// process interrupt message
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

	socket_t socket = socket_accept(fd);
	if(socket == -1) return;

	if(!socket_set_non_blocking(socket)) {
		close(socket);
		return;
	}

	bolt_client_t *client = bolt_client_new(socket, BoltResponseHandler);

	raxInsert(clients, (unsigned char *)&socket, sizeof(socket), client, NULL);

	// add client socket to event loop
	RedisModule_EventLoopAdd(socket, REDISMODULE_EVENTLOOP_READABLE,
			BoltHandshakeHandler, client);
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

	// bind to Bolt port
    socket_t bolt = socket_bind(port);
	if(bolt == -1) {
		RedisModule_Log(ctx, "warning", "Failed to bind to port %d", port);
		return REDISMODULE_ERR;
	}

	// add Bolt socket to event loop
	if(RedisModule_EventLoopAdd(bolt, REDISMODULE_EVENTLOOP_READABLE,
				BoltAcceptHandler, NULL) == REDISMODULE_ERR) {
		RedisModule_Log(ctx, "warning",
				"Failed to register socket accept handler");
		return REDISMODULE_ERR;
	}

	RedisModule_Log(ctx, "notice", "Bolt protocol initialized. Port: %d", port);

	BOLT    = RedisModule_CreateString(ctx, "--bolt", 6);
	COMMAND = RedisModule_CreateString(ctx, "graph.QUERY", 11);

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

