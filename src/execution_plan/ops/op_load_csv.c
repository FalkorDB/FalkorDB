/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_load_csv.h"
#include "../../datatypes/map.h"
#include "../../datatypes/array.h"
#include "../../configuration/config.h"
#include "../../util/path_utils.h"

// forward declarations
static OpResult LoadCSVInit(OpBase *opBase);
static Record LoadCSVConsume(OpBase *opBase);
static Record LoadCSVConsumeDepleted(OpBase *opBase);
static Record LoadCSVConsumeFromChild(OpBase *opBase);
static OpBase *LoadCSVClone(const ExecutionPlan *plan, const OpBase *opBase);
static OpResult LoadCSVReset(OpBase *opBase);
static void LoadCSVFree(OpBase *opBase);

// evaluate URI expression
// expression must evaluate to string representing a valid URI
// if that's not the case an exception is raised
static bool _computeURI
(
	OpLoadCSV *op,
	Record r
) {
	ASSERT(op != NULL);

	op->uri = AR_EXP_Evaluate(op->exp, r);

	// check uri type
	if(!(SI_TYPE(op->uri) & T_STRING)) {
		ErrorCtx_SetError(EMSG_INVALID_CSV_URI);
		return false;
	}

	// supported URIs: file:// & https://
	const char *csv_uri = op->uri.stringval;
	static const char* URIS[2] = {"file://", "https://"};

	// make sure uri is supported
	for(int i = 0; i < 2; i++) {
		const char *uri = URIS[i];
		if(strncasecmp(csv_uri, uri, strlen(uri)) == 0) {
			return true;
		}
	}

	// unsupported CSV URI
	ErrorCtx_SetError(EMSG_UNSUPPORTED_CSV_URI);
	return false;
}

// initialize CSV reader from https URI
static FILE *_getRemoteURIReadStream
(
	OpLoadCSV *op  // load CSV operation
) {
	int pipefd[2];  // pipe ends
	const char *uri = op->uri.stringval;

	// create pipe from which to read remote file
	if(pipe(pipefd) == -1) {
		ErrorCtx_RaiseRuntimeException("Error creating pipe");
		return NULL;
	}

	// download remote file
	// file content will be written to pipe
	FILE *f = fdopen(pipefd[1], "wb");
	op->curl = Curl_Download(uri, &f);
	if(op->curl == NULL) {
		// close pipe read end, write-end closed by Curl_Download
		close(pipefd[0]);

		ErrorCtx_RaiseRuntimeException("Error downloading file: %s", uri);
		return NULL;
	}

	// return file descriptor from read end of pipe
	return fdopen(pipefd[0], "r");
}

// initialize CSV reader from a local file URI file://
static FILE *_getLocalURIReadStream
(
	OpLoadCSV *op  // load CSV operation
) {
	const char *uri = op->uri.stringval + 7;  // skip file://

    char full_path[PATH_MAX];

	// read import folder path from configuration
	const char *import_folder = NULL;
	Config_Option_get(Config_IMPORT_FOLDER, &import_folder);

    // construct the full path
    snprintf(full_path, sizeof(full_path), "%s%s", import_folder, uri);

	if(!is_safe_path(import_folder, full_path)) {
		// log file access
		RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING,
				"attempt to access unauthorized path %s", full_path);
		return NULL;
	}

	// log file access
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_NOTICE, "opening %s", full_path);

	// open local file
	return fopen(full_path, "r");
}

// initialize CSV reader
static bool _Init_CSVReader
(
	OpLoadCSV *op  // load CSV operation
) {
	ASSERT(op != NULL);

	// free old downloader & reader
	if(op->curl != NULL) {
		Curl_Free(&op->curl);
	}

	if(op->reader != NULL) {
		CSVReader_Free(&op->reader);
	}

	// initialize a new CSV reader
	FILE *stream = NULL;
	const char *uri = op->uri.stringval;

	// get CSV URI read stream
	if(strncasecmp(uri, "file://", 7) == 0) {
		stream = _getLocalURIReadStream(op);
	} else {
		stream = _getRemoteURIReadStream(op);
	}

	if(stream == NULL) {
		ErrorCtx_RaiseRuntimeException("Error opening CSV URI: %s", uri);
		return false;
	}

	op->reader = CSVReader_New(stream, op->with_headers, op->delimiter);

	return (op->reader != NULL);
}

// get a single CSV row
static bool _CSV_GetRow
(
	OpLoadCSV *op,  // load CSV operation
	SIValue *row    // row to populate
) {
	ASSERT(op  != NULL);
	ASSERT(row != NULL);

	// try to get a new row from CSV
	*row = CSVReader_GetRow(op->reader);
	return (!SIValue_IsNull(*row));
}

// create a new load CSV operation
OpBase *NewLoadCSVOp
(
	const ExecutionPlan *plan,  // execution plan
	AR_ExpNode *exp,            // CSV URI expression
	const char *alias,          // CSV row alias
	bool with_headers,          // CSV contains header row
	char delimiter              // field delimiter
) {
	ASSERT(exp   != NULL);
	ASSERT(plan  != NULL);
	ASSERT(alias != NULL);

	OpLoadCSV *op = rm_calloc(1, sizeof(OpLoadCSV));

	op->exp          = exp;
	op->uri          = SI_NullVal();
	op->alias        = rm_strdup(alias);
	op->delimiter    = delimiter;
	op->with_headers = with_headers;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_LOAD_CSV, "Load CSV", LoadCSVInit,
			LoadCSVConsume, NULL, NULL, LoadCSVClone, LoadCSVFree, false, plan);

	op->recIdx = OpBase_Modifies((OpBase *)op, alias);

	return (OpBase *)op;
}

// initialize operation
static OpResult LoadCSVInit
(
	OpBase *opBase
) {
	// set operation's consume function

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	// update consume function in case operation has a child
	if(OpBase_ChildCount(opBase) > 0) {
		ASSERT(OpBase_ChildCount(opBase) == 1);

		op->child = OpBase_GetChild(opBase, 0);
		OpBase_UpdateConsume(opBase, LoadCSVConsumeFromChild);	

		return OP_OK;
	}

	//--------------------------------------------------------------------------
	// no child operation evaluate URI expression
	//--------------------------------------------------------------------------

	// evaluate URI expression
	Record r   = OpBase_CreateRecord(opBase);
	bool   res = _computeURI(op, r);

	OpBase_DeleteRecord(&r);

	if(!res) {
		// failed to evaluate CSV URI
		// update consume function
		OpBase_UpdateConsume(opBase, LoadCSVConsumeDepleted);
		return OP_ERR;
	}

	if(!_Init_CSVReader(op)) {
		// failed to init CSV
		// update consume function
		OpBase_UpdateConsume(opBase, LoadCSVConsumeDepleted);
	}

	return OP_OK;
}

// simply return NULL indicating operation depleted
static Record LoadCSVConsumeDepleted
(
	OpBase *opBase
) {
	return NULL;
}

// load CSV consume function in case this operation in not a tap
static Record LoadCSVConsumeFromChild
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

pull_from_child:

	// in case a record is missing ask child to provide one
	if(op->child_record == NULL) {
		op->child_record = OpBase_Consume(op->child);

		// child failed to provide record, depleted
		if(op->child_record == NULL) {
			return NULL;
		}

		// first call, evaluate CSV URI
		if(!_computeURI(op, op->child_record)) {
			// failed to evaluate CSV URI, quickly return
			return NULL;
		}

		// create a new CSV reader
		if(!_Init_CSVReader(op)) {
			return NULL;
		}
	}

	// must have a record at this point
	ASSERT(op->reader       != NULL);
	ASSERT(op->child_record != NULL);

	// get a new CSV row
	SIValue row;
	if(!_CSV_GetRow(op, &row)) {
		// failed to get a CSV row
		// reset CSV reader and free current child record
		OpBase_DeleteRecord(&op->child_record);
		op->child_record = NULL;

		// free CSV URI, just in case it relies on record data
		SIValue_Free(op->uri);
		op->uri = SI_NullVal();

		// try to get a new record from child
		goto pull_from_child;
	}

	// managed to get a new CSV row
	// update record and return to caller
	Record r = OpBase_CloneRecord(op->child_record);
	Record_AddScalar(r, op->recIdx, row);

	return r;
}

// load CSV consume function in the case this operation is a tap
static Record LoadCSVConsume
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue row;
	Record r = NULL;

	if(_CSV_GetRow(op, &row)) {
		r = OpBase_CreateRecord(opBase);
		Record_AddScalar(r, op->recIdx, row);
	}

	return r;
}

static inline OpBase *LoadCSVClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_LOAD_CSV);

	OpLoadCSV *op = (OpLoadCSV*)opBase;
	return NewLoadCSVOp(plan, AR_EXP_Clone(op->exp), op->alias,
			op->with_headers, op->delimiter);
}

static OpResult LoadCSVReset (
	OpBase *opBase
) {
	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue_Free(op->uri);
	op->uri = SI_NullVal();

	if(op->child_record != NULL) {
		OpBase_DeleteRecord(&op->child_record);
		op->child_record = NULL;
	}

	if(op->curl != NULL) {
		// aborts in-progress download
		// must be called before csv reader is freed
		// due to pipe read end being closed before write end
		Curl_Free(&op->curl);
	}

	if(op->reader != NULL) {
		CSVReader_Free(&op->reader);
	}

	return OP_OK;
}

// free Load CSV operation
static void LoadCSVFree
(
	OpBase *opBase
) {
	ASSERT(opBase != NULL);

	OpLoadCSV *op = (OpLoadCSV*)opBase;

	SIValue_Free(op->uri);

	if(op->curl != NULL) {
		// aborts in-progress download
		// must be called before csv reader is freed
		// due to pipe read end being closed before write end
		Curl_Free(&op->curl);
	}

	if(op->exp != NULL) {
		AR_EXP_Free(op->exp);
		op->exp = NULL;
	}

	if(op->alias != NULL) {
		rm_free(op->alias);
		op->alias = NULL;
	}
	
	if(op->child_record != NULL) {
		OpBase_DeleteRecord(&op->child_record);
		op->child_record = NULL;
	}

	if(op->reader != NULL) {
		// must be called after curl_free
		ASSERT(op->curl == NULL);

		CSVReader_Free(&op->reader);
	}
}

