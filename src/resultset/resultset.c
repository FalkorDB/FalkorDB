/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "resultset.h"
#include "RG.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../bolt/bolt.h"
#include "../bolt/socket.h"
#include "../globals.h"
#include "../errors/errors.h"
#include "../commands/cmd_context.h"

static void _ResultSet_ReplyWithPreamble
(
	ResultSet *set
) {
	set->formatter->EmitHeader(set);
}

static void _ResultSet_SetColumns
(
	ResultSet *set
) {
	ASSERT(set->columns == NULL);

	AST *ast = QueryCtx_GetAST();
	const cypher_astnode_type_t root_type = cypher_astnode_type(ast->root);

	if(root_type == CYPHER_AST_QUERY) {
		uint clause_count = cypher_ast_query_nclauses(ast->root);

		const cypher_astnode_t *last_clause =
			cypher_ast_query_get_clause(ast->root, clause_count - 1);

		cypher_astnode_type_t last_clause_type =
			cypher_astnode_type(last_clause);

		if(last_clause_type == CYPHER_AST_RETURN) {
			set->columns = AST_BuildReturnColumnNames(last_clause);
		} else if(last_clause_type == CYPHER_AST_CALL) {
			set->columns = AST_BuildCallColumnNames(last_clause);
		}

		set->column_count = array_len(set->columns);
	}
}

// create a new result set
ResultSet *NewResultSet
(
	RedisModuleCtx *ctx,
	bolt_client_t *bolt_client,
	ResultSetFormatterType format  // resultset format
) {
	ResultSet *set = rm_malloc(sizeof(ResultSet));

	set->gc                  =  QueryCtx_GetGraphCtx();
	set->ctx                 =  ctx;
	set->cells               =  NULL;
	set->format              =  format;
	set->columns             =  NULL;
	set->formatter           =  ResultSetFormatter_GetFormatter(format);
	set->bolt_client         =  bolt_client;
	set->column_count        =  0;
	set->cells_allocation    =  M_NONE;
	set->columns_record_map  =  NULL;

	// init resultset statistics
	ResultSetStat_init(&set->stats);

	// create resultset columns
	if(set->format != FORMATTER_NOP) {
		_ResultSet_SetColumns(set);
	}

	// allocate space for resultset entries only if data is expected
	if(set->column_count > 0) {
		// none empty result-set
		// allocate enough space for at least 10 rows
		uint64_t nrows = set->column_count * 10;
		set->cells = DataBlock_New(16384, nrows, sizeof(SIValue), NULL);
	}

	return set;
}

// map each column to a record index
// such that when resolving resultset row i column j we'll extract
// data from record at position columns_record_map[j]
void ResultSet_MapProjection
(
	ResultSet *set,  // resultset to init mappings for
	rax *mapping     // mapping
) {
	ASSERT(set     != NULL);
	ASSERT(mapping != NULL);

	if(set->columns_record_map == NULL) {
		set->columns_record_map = rm_malloc(sizeof(uint) * set->column_count);
	}

	for(uint i = 0; i < set->column_count; i++) {
		const char *column = set->columns[i];
		void *idx = raxFind(mapping, (unsigned char *)column, strlen(column));
		ASSERT(idx != raxNotFound);
		set->columns_record_map[i] = (intptr_t)idx;
	}
}

// returns number of rows in result-set
uint64_t ResultSet_RowCount
(
	const ResultSet *set  // resultset to inquery
) {
	ASSERT(set != NULL);

	if(set->column_count == 0) return 0;
	return DataBlock_ItemCount(set->cells) / set->column_count;
}

// add a new row to resultset
int ResultSet_AddRecord
(
	ResultSet *set,  // resultset to extend
	Record r         // record containing projected data
) {
	ASSERT(r   != NULL);
	ASSERT(set != NULL);

	// copy projected values from record to resultset
	for(int i = 0; i < set->column_count; i++) {
		int idx = set->columns_record_map[i];
		SIValue *cell = DataBlock_AllocateItem(set->cells, NULL);
		*cell = Record_Get(r, idx);
		SIValue_Persist(cell);
		set->cells_allocation |= SI_ALLOCATION(cell);
	}

	// remove entry from record in a second pass
	// this will ensure duplicated projections are not removed
	// too early, consider: MATCH (a) RETURN max(a.val), max(a.val)
	for(int i = 0; i < set->column_count; i++) {
		int idx = set->columns_record_map[i];
		Record_Remove(r, idx);
	}

	return RESULTSET_OK;
}

// update resultset index creation statistics
void ResultSet_IndexCreated
(
	ResultSet *set,  // resultset to update
	int status_code  // index creation status code
) {
	ASSERT(set != NULL);

	set->stats.index_creation = true;
	if(status_code == INDEX_OK) {
		set->stats.indices_created += 1;
	}
}

// update resultset index deleted statistics
void ResultSet_IndexDeleted
(
	ResultSet *set,  // resultset to update
	int status_code  // index deletion status code
) {
	ASSERT(set != NULL);

	set->stats.index_deletion = true;
	if(status_code == INDEX_OK) {
		set->stats.indices_deleted += 1;
	}
}

// update resultset constraint creation statistics
void ResultSet_ConstraintCreated
(
	ResultSet *set,  // resultset to update
	int status_code  // index creation status code
) {
	ASSERT(set != NULL);

	set->stats.constraint_creation = true;
	if(status_code == INDEX_OK) {
		set->stats.constraints_created += 1;
	}
}

// update resultset constraint deleted statistics
void ResultSet_ConstraintDeleted
(
	ResultSet *set,  // resultset to update
	int status_code  // index deletion status code
) {
	ASSERT(set != NULL);

	set->stats.constraint_deletion = true;
	if(status_code == INDEX_OK) {
		set->stats.constraints_deleted += 1;
	}
}

// update resultset cache execution statistics
void ResultSet_CachedExecution
(
	ResultSet *set  // resultset to update
) {
	ASSERT(set != NULL);
	set->stats.cached = true;
}

// flush resultset to network
void ResultSet_Reply
(
	ResultSet *set  // resultset to reply with
) {
	ASSERT(set != NULL);

	uint64_t row_count = ResultSet_RowCount(set);

	// check to see if we've encountered a run-time error
	// if so, emit it as the only response
	if(ErrorCtx_EncounteredError()) {
		ErrorCtx_EmitException();
		return;
	}

	// set up the results array and emit the header if the query requires one
	_ResultSet_ReplyWithPreamble(set);

	// emit resultset
	if(set->column_count > 0) {
		RedisModule_ReplyWithArray(set->ctx, row_count);
		SIValue *row[set->column_count];
		uint64_t cells = DataBlock_ItemCount(set->cells);
		// for each row
		for(uint64_t i = 0; i < cells; i += set->column_count) {
			// for each column
			for(uint j = 0; j < set->column_count; j++) {
				row[j] = DataBlock_GetItem(set->cells, i + j);
			}

			set->formatter->EmitRow(set, row);
		}
	}

	set->formatter->EmitStats(set);
}

void ResultSet_Clear(ResultSet *set) {
	ASSERT(set != NULL);
	ResultSetStat_Clear(&set->stats);
}

// free resultset
void ResultSet_Free
(
	ResultSet *set  // resultset to free
) {
	if(set == NULL) return;

	if(set->columns) {
		array_free(set->columns);
	}

	if(set->columns_record_map) {
		rm_free(set->columns_record_map);
	}

	// free resultset cells
	// NOTE: for large result-set containing only NONE heap allocated values
	// the following is a bit of a waste as there's no real memory to free
	// at the moment we can't tell rather or not
	// calling SIValue_Free is required
	if(set->cells) {
		// free individual cells if resultset encountered a heap allocated value
		if(set->cells_allocation & M_SELF) {
			uint64_t n = DataBlock_ItemCount(set->cells);
			for(uint64_t i = 0; i < n; i++) {
				SIValue *v = DataBlock_GetItem(set->cells, i);
				SIValue_Free(*v);
			}
		}
		DataBlock_Free (&set->cells) ;
	}

	rm_free (set) ;
}
