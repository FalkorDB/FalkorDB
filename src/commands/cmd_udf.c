/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../util/arr.h"
#include "../udf/utils.h"
#include "../udf/udf_ctx.h"
#include "../udf/repository.h"
#include "../udf/replication.h"
#include "../arithmetic/func_desc.h"
#include "../arithmetic/udf_funcs/udf_funcs.h"

// forward declarations
int Graph_UDF_List
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) ;

int Graph_UDF_Load
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	bool *success
) ;

int Graph_UDF_Flush
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	bool *success
) ;

int Graph_UDF_Delete
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	bool *success
) ;

// GRAPH.UDF * command handler
// sub commands:
// GRAPH.UDF LOAD [REPLACE] <lib> <script>
// GRAPH.UDF LIST [LIBRARYNAME] [WITHCODE]
// GRAPH.UDF FLUSH
// GRAPH.UDF DELETE <lib>
int Graph_UDF
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	if (argc < 2) {
		RedisModule_WrongArity (ctx) ;
		return REDISMODULE_OK ;
	}

	int res;
	bool modify  = false ;
	bool success = false ;
	RedisModuleString *rm_sub_cmd = argv[1] ;
	const char *sub_cmd = RedisModule_StringPtrLen (rm_sub_cmd, NULL) ;

	if (strcasecmp (sub_cmd, "load") == 0) {
		modify = true ;
		res = Graph_UDF_Load (ctx, argv+2, argc-2, &success) ;
	}

	else if (strcasecmp (sub_cmd, "delete") == 0) {
		modify = true ;
		res = Graph_UDF_Delete (ctx, argv+2, argc-2, &success) ;
	}

	else if (strcasecmp (sub_cmd, "flush") == 0) {
		modify = true ;
		res = Graph_UDF_Flush (ctx, argv+2, argc-2, &success) ;
	}

	else if (strcasecmp (sub_cmd, "list") == 0) {
		res = Graph_UDF_List (ctx, argv+2, argc-2) ;
	}

	else {
		RedisModule_ReplyWithErrorFormat (ctx,
				"Unknown GRAPH.UDF sub command %s", sub_cmd) ;
	}

	if (modify == true && success == true) {
		RedisModule_ReplicateVerbatim (ctx) ;

		// replicate command to the rest of the cluster
		UDF_ReplicationSendCmd (ctx, argv, argc) ;
	}

	return res ;
}

