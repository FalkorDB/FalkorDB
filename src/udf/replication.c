/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "replication.h"
#include "../util/rmalloc.h"

// unique type for UDF messages
#define UDF_CLUSTER_MSG 1  // UDF cluster message ID

typedef enum {
	UDF_LOAD_CMD   = 1,  // UDF LOAD   command ID
	UDF_FLUSH_CMD  = 2,  // UDF FLUSH  command ID
	UDF_DELETE_CMD = 3,  // UDF DELETE command ID
} UDF_REPLICATION_CMD ;

// UDF cluster message callback receiver
static void UDF_ClusterMessageReceiver
(
	RedisModuleCtx *ctx,           // redis module context
	const char *sender_id,         // sender id
	uint8_t type,                  // message type
	const unsigned char *payload,  // message payload
	uint32_t len                   // payload length
) {
	// expecting only UDF_CLUSTER_MSG
	ASSERT (type == (uint8_t)UDF_CLUSTER_MSG) ;
	ASSERT (len >= sizeof (UDF_REPLICATION_CMD)) ;

	// parse the message and perform requested action
	// possible actions:
	// 1. UDF flush
	// 2. UDF delete library
	// 3. UDF load library

	FILE *f = fmemopen ((void*)payload, len, "rb") ;

	// read UDF action
	UDF_REPLICATION_CMD cmd ;
	fread (&cmd, sizeof(UDF_REPLICATION_CMD), 1, f) ;

	long idx           = 0 ; 
	char *err          = NULL ;
	size_t lib_len     = 0 ;
	size_t read        = 0 ;
	const char *lib    = NULL ;
	size_t script_len  = 0 ;
	const char *script = NULL ;

	switch (cmd) {
		case UDF_LOAD_CMD:
			// message format:
			//  UDF lib len         | 4 bytes
			//  UDF lib name        | len bytes
			//  UDF lib script len  | 4 bytes
			//  UDF lib script      | len bytes

			read += fread (&lib_len, sizeof (size_t), 1, f) ;
			idx = ftell (f) ;
			lib = (const char*) payload + idx ;
			fseek (f, lib_len, SEEK_CUR) ;  // skip over lib

			read += fread (&script_len, sizeof(size_t), 1, f) ;  // read script len
			idx = ftell (f) ;
			script = (const char*) payload + idx ;
			fseek (f, script_len, SEEK_CUR) ;  // skip over script

			ASSERT (lib_len    > 0) ;
			ASSERT (script_len > 0) ;

			UDF_Load (script,        // script to load
					script_len - 1,  // do not count NULL
					lib,             // library name
					lib_len - 1,     // do not count NULL
					true,            // always replace (no harm)
					&err             // error, expecting no errors
			) ;
			ASSERT (err == NULL) ;

			break ;

		case UDF_DELETE_CMD:
			// message format:
			//  UDF lib to delete len
			//  UDF lib to delete name

			fread (&lib_len, sizeof(size_t), 1, f) ;
			idx = ftell (f) ;
			lib = (const char*) payload + idx ;
			fseek (f, lib_len, SEEK_CUR) ;  // skip over lib

			ASSERT (lib_len > 0) ;

			UDF_Delete (lib, NULL, &err) ;

			break ;

		case UDF_FLUSH_CMD:
			// message format:
			//  UDF action

			ASSERT (len == sizeof (UDF_REPLICATION_CMD)) ;

			UDF_Flush() ;
			break ;

		default:
			assert (false && "unknown UDF replication command") ;
	}

	fclose (f) ;
}

// register callback receivers for cluster UDF messages 
void UDF_ReplicationRegisterReceiver
(
	RedisModuleCtx *ctx  // redis module context
) {
	RedisModule_RegisterClusterMessageReceiver (ctx, UDF_CLUSTER_MSG,
			UDF_ClusterMessageReceiver) ;
}

// counter part of UDF_ReplicationRegisterReceiver
// unregister callback receivers
void UDF_ReplicationUnRegisterReceiver
(
	RedisModuleCtx *ctx  // redis module context
) {
	// unregister receiver
	RedisModule_RegisterClusterMessageReceiver (ctx, UDF_CLUSTER_MSG, NULL) ;
}

// replicate a UDF command to the cluster
void UDF_ReplicationSendCmd
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	int argc                   // number of arguments
) {
	// get nodes in cluster
	size_t numnodes = 0 ;
	char **node_ids = RedisModule_GetClusterNodesList (ctx, &numnodes);

	// return if there are no cluster / empty cluster
	if (node_ids == NULL) {
		return ;	
	}

	RedisModuleString *rm_sub_cmd = argv[1] ;
	const char *sub_cmd = RedisModule_StringPtrLen (rm_sub_cmd, NULL) ;

	FILE *f          = NULL ;             // memory stream
	uint8_t type     = UDF_CLUSTER_MSG ;  // cluster replication message type
	char *msg        = NULL ;             // message
	size_t written   = 0 ;                // number of objects written
	uint32_t msg_len = 0 ;                // message length
	UDF_REPLICATION_CMD cmd_code ;        // replicated command

	if (strcasecmp (sub_cmd, "load") == 0) {
		// message format:
		//  UDF action          | 4 bytes
		//  UDF lib len         | 4 bytes
		//  UDF lib name        | len bytes
		//  UDF lib script len  | 4 bytes
		//  UDF lib script      | len bytes

		// GRAPH.UDF LOAD [REPLACE] <lib> <script>
		bool replace = argc == 5 ;
		int  offset  = (replace) ? 1 : 0 ;

		cmd_code = UDF_LOAD_CMD ;

		size_t lib_len = 0;
		const char *lib =
			RedisModule_StringPtrLen (argv[2 + offset], &lib_len) ;
		lib_len++ ; // include null terminator

		size_t script_len = 0;
		const char *script =
			RedisModule_StringPtrLen (argv[3 + offset], &script_len) ;
		script_len++ ; //include null terminator

		ASSERT (lib_len    > 0) ;
		ASSERT (script_len > 0) ;
		ASSERT (lib        != NULL) ;
		ASSERT (script     != NULL) ;

		msg_len = sizeof (UDF_REPLICATION_CMD) +
				  sizeof (lib_len)             +
				  lib_len                      +
				  sizeof (script_len)          +
				  script_len                   ;

		msg = rm_malloc (sizeof (char) * msg_len) ;

		f = fmemopen ((void*)msg, msg_len, "wb") ;

		written += fwrite (&cmd_code, sizeof (UDF_REPLICATION_CMD), 1, f); // action
		written += fwrite (&lib_len, sizeof (lib_len), 1, f) ;             // lib len
		written += fwrite (lib, lib_len, 1, f);                            // lib name
		written += fwrite (&script_len, sizeof (script_len), 1, f) ;       // script len
		written += fwrite (script, script_len, 1, f) ;                     // script
		ASSERT (written == 5) ;
	}

	else if (strcasecmp (sub_cmd, "delete") == 0) {
		// message format:
		//  UDF action             | 4 bytes
		//  UDF lib to delete len  | 4 bytes
		//  UDF lib to delete name | len bytes

		cmd_code = UDF_DELETE_CMD ;
		size_t lib_len = 0;
		const char *lib = RedisModule_StringPtrLen (argv[2], &lib_len) ;

		ASSERT (lib_len > 0) ;
		ASSERT (lib     != NULL) ;

		msg_len = sizeof (UDF_REPLICATION_CMD) +
				  sizeof (lib_len)             +
				  lib_len                      ;

		msg = rm_malloc (sizeof (char) * msg_len) ;

		f = fmemopen ((void*)msg, msg_len, "wb") ;

		written += fwrite (&cmd_code, sizeof (UDF_REPLICATION_CMD), 1, f);  // action
		written += fwrite (&lib_len, sizeof (lib_len), 1, f) ;              // lib len
		written += fwrite (lib, lib_len, 1, f);                             // lib name
		ASSERT (written == 3) ;
	}

	else if (strcasecmp (sub_cmd, "flush") == 0) {
		// message format:
		//  UDF action | 4 bytes

		cmd_code = UDF_FLUSH_CMD ;
		msg_len = sizeof (UDF_REPLICATION_CMD) ;

		msg = rm_malloc (sizeof (char) * msg_len) ;

		f = fmemopen ((void*)msg, msg_len, "wb") ;

		written += fwrite (&cmd_code, sizeof (UDF_REPLICATION_CMD), 1, f);  // action
		ASSERT (written == 1) ;
	}

	else {
		// unknown sub command
		assert (false && "unknown UDF sub-command") ;
		return ;
	}

	// flush and close the memory stream
	fflush (f) ;
	fclose (f) ;

	// send command to each master node in the cluster
	for (size_t i = 0 ; i < numnodes ; i++) {
		int flags ;
		RedisModule_GetClusterNodeInfo (ctx, node_ids[i], NULL, NULL, NULL,
				&flags) ;

		// skip non-master nodes and self
		if (flags & REDISMODULE_NODE_MYSELF ||       // self
				!(flags & REDISMODULE_NODE_MASTER))  // not master
		{
			continue ;
		}

		// send command to master node
		RedisModule_SendClusterMessage (ctx, node_ids[i], UDF_CLUSTER_MSG, msg,
				msg_len) ;
	}

	if (msg != NULL) {
		rm_free (msg) ;
	}

	RedisModule_FreeClusterNodesList (node_ids) ;
}

