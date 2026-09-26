/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "proc_list_indexes.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/cch_index.h"
#include "../schema/schema.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"

#include <assert.h>
#include <string.h>

typedef struct {
	SIValue out[9];             // outputs
	Index *indices;             // indicies to emit
	CCHIndex **cch_indices;     // CCH path indices to emit
	GraphContext *gc;           // graph context
	SIValue *yield_label;       // yield index label
	SIValue *yield_types;       // yield index fields types
	SIValue *yield_fields;      // yield index fields
	SIValue *yield_options;     // yield index options
	SIValue *yield_language;    // yield index language
	SIValue *yield_stopwords;   // yield index stopwords
	SIValue *yield_entity_type; // yield index entity type
	SIValue *yield_status;      // yield index status
	SIValue *yield_info;        // yield info
} IndexesContext;

static void _process_yield
(
	IndexesContext *ctx,
	const char **yield
) {
	ctx->yield_info        = NULL;
	ctx->yield_label       = NULL;
	ctx->yield_types       = NULL;
	ctx->yield_fields      = NULL;
	ctx->yield_status      = NULL;
	ctx->yield_options     = NULL;
	ctx->yield_language    = NULL;
	ctx->yield_stopwords   = NULL;
	ctx->yield_entity_type = NULL;

	int idx = 0;
	for(uint i = 0; i < arr_len(yield); i++) {
		if(strcasecmp("label", yield[i]) == 0) {
			ctx->yield_label = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("types", yield[i]) == 0) {
			ctx->yield_types = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("options", yield[i]) == 0) {
			ctx->yield_options = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("properties", yield[i]) == 0) {
			ctx->yield_fields = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("language", yield[i]) == 0) {
			ctx->yield_language = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("stopwords", yield[i]) == 0) {
			ctx->yield_stopwords = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("entitytype", yield[i]) == 0) {
			ctx->yield_entity_type = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("status", yield[i]) == 0) {
			ctx->yield_status = ctx->out + idx;
			idx++;
			continue;
		}

		if(strcasecmp("info", yield[i]) == 0) {
			ctx->yield_info = ctx->out + idx;
			idx++;
			continue;
		}
	}
}

// CALL db.indexes()
ProcedureResult Proc_IndexesInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {

	ASSERT(ctx   != NULL);
	ASSERT(args  != NULL);
	ASSERT(yield != NULL);

	// TODO: introduce invoke validation, similar to arithmetic expressions
	// expecting no arguments
	uint arg_count = arr_len((SIValue *)args);
	if(arg_count != 0) return PROCEDURE_ERR;

	GraphContext *gc = QueryCtx_GetGraphCtx();

	IndexesContext *pdata = rm_malloc(sizeof(IndexesContext));

	pdata->gc           = gc;
	pdata->indices      = arr_new(Index, 0);
	pdata->cch_indices  = arr_new(CCHIndex *, 0);

	//--------------------------------------------------------------------------
	// collect all indices
	//--------------------------------------------------------------------------

	unsigned short n;            // number of schemas
	Schema         *s;           // current schema
	unsigned short idx_count;    // number of indicies in schema
	Index          indicies[4];  // schema indicies

	// collect indices from node schemas
	n = GraphContext_SchemaCount(gc, SCHEMA_NODE);
	for(unsigned short i = 0; i < n; i++) {
		s = GraphContext_GetSchemaByID(gc, i, SCHEMA_NODE);
		idx_count = Schema_GetIndicies(s, indicies);
		for(unsigned short j = 0; j < idx_count; j++) {
			arr_append(pdata->indices, indicies[j]);
		}
	}

	// collect indices from edge schemas
	n = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
	for(unsigned short i = 0; i < n; i++) {
		s = GraphContext_GetSchemaByID(gc, i, SCHEMA_EDGE);
		idx_count = Schema_GetIndicies(s, indicies);
		for(uint j = 0; j < idx_count; j++) {
			arr_append(pdata->indices, indicies[j]);
		}
	}

	// collect graph-level CCH path indices
	uint cch_count = GraphContext_CCHIndexCount(gc);
	for(uint i = 0; i < cch_count; i++) {
		arr_append(pdata->cch_indices, GraphContext_GetCCHIndexAt(gc, i));
	}

	_process_yield(pdata, yield);

	ctx->privateData = pdata;

	return PROCEDURE_OK;
}

static bool _EmitIndex
(
	IndexesContext *ctx,
	Index idx
) {
	GraphContext *gc = ctx->gc;
	Graph        *g = GraphContext_GetGraph(gc);

	// get index info
	RSIdxInfo info = { .version = RS_INFO_CURRENT_VERSION };
	RSIndex *rsIdx = Index_RSIndex(idx);
	RediSearch_IndexInfo(rsIdx, &info);

	//--------------------------------------------------------------------------
	// index entity type
	//--------------------------------------------------------------------------

	if(ctx->yield_entity_type != NULL) {
		if(Index_GraphEntityType(idx) == GETYPE_NODE) {
			*ctx->yield_entity_type = SI_ConstStringVal("NODE");
		} else {
			*ctx->yield_entity_type = SI_ConstStringVal("RELATIONSHIP");
		}
	}

	//--------------------------------------------------------------------------
	// index status
	//--------------------------------------------------------------------------

	if(ctx->yield_status != NULL) {
		if(Index_Enabled(idx)) {
			// index is operational
			*ctx->yield_status = SI_ConstStringVal("OPERATIONAL");
		} else {
			// report index construction progress
			// determine number of entities being indexed
			size_t n;                               // total number of entities
			Schema *s;                              // entities schema
			const char *lbl = Index_GetLabel(idx);  // entities label

			// get total number of entities from the graph
			if(Index_GraphEntityType(idx) == GETYPE_NODE) {
				s = GraphContext_GetSchema(gc, lbl, SCHEMA_NODE);
				n = Graph_LabeledNodeCount(g, Schema_GetID(s));
			} else {
				s = GraphContext_GetSchema(gc, lbl, SCHEMA_EDGE);
				n = Graph_RelationEdgeCount(g, Schema_GetID(s));
			}

			// report progress
			int res ;
			char *status;
			size_t m = info.numDocuments;
			res = asprintf(&status, "[Indexing] %zu/%zu: UNDER CONSTRUCTION",
					m, n);
			assert (res >= 34) ;

			*ctx->yield_status = SI_DuplicateStringVal(status);
			free(status);
		}
	}

	//--------------------------------------------------------------------------
	// index label
	//--------------------------------------------------------------------------

	if(ctx->yield_label) {
		*ctx->yield_label = SI_ConstStringVal((char *)Index_GetLabel(idx));
	}

	//--------------------------------------------------------------------------
	// index fields
	//--------------------------------------------------------------------------

	if(ctx->yield_fields) {
		uint fields_count        = Index_FieldsCount(idx);
		const IndexField *fields = Index_GetFields(idx);
		*ctx->yield_fields       = SI_Array(fields_count);

		for(uint i = 0; i < fields_count; i++) {
			const IndexField *field = fields + i;
			SIArray_Append(ctx->yield_fields,
					SI_ConstStringVal(IndexField_GetName(field)));
		}
	}

	//--------------------------------------------------------------------------
	// index fields types
	//--------------------------------------------------------------------------

	if(ctx->yield_types) {
		// {field_name, [field_types]}
		// e.g.
		// {
		//  'name': ['range', 'fulltext']
		//  'image' ['vector']
		// }

		uint fields_count        = Index_FieldsCount(idx);
		const IndexField *fields = Index_GetFields(idx);
		*ctx->yield_types        = SI_Map(fields_count);

		for(uint i = 0; i < fields_count; i++) {
			const IndexField *field = fields + i;
			IndexFieldType types = IndexField_GetType(field);
			int n = IndexField_TypeCount(field);  // number of types

			SIValue prop_types = SI_Array(n);

			if(types & INDEX_FLD_RANGE) {
				SIArray_Append(&prop_types, SI_ConstStringVal("RANGE"));
			}
			if(types & INDEX_FLD_VECTOR) {
				SIArray_Append(&prop_types, SI_ConstStringVal("VECTOR"));
			}
			if(types & INDEX_FLD_FULLTEXT) {
				SIArray_Append(&prop_types, SI_ConstStringVal("FULLTEXT"));
			}

			Map_Add(ctx->yield_types,
					SI_ConstStringVal(IndexField_GetName(field)), prop_types);

			SIValue_Free(prop_types);
		}
	}

	//--------------------------------------------------------------------------
	// index fields options
	//--------------------------------------------------------------------------

	if(ctx->yield_options) {
		// {field_name, {field_options}
		// e.g.
		// {
		//  'embedding': {'dimension': 100, 'M': 16, 'efConstruction': 200, 'efRuntime': 100}
		// }

		uint fields_count        = Index_FieldsCount(idx);
		const IndexField *fields = Index_GetFields(idx);
		*ctx->yield_options      = SI_Map(fields_count);

		for(uint i = 0; i < fields_count; i++) {
			const IndexField *field = fields + i;
			IndexFieldType types = IndexField_GetType(field);

			SIValue prop_types = SI_Map(0);

			if(types & INDEX_FLD_VECTOR) {
				Map_Add(&prop_types, SI_ConstStringVal("dimension"),
						SI_LongVal(IndexField_OptionsGetDimension(field)));
				Map_Add(&prop_types, SI_ConstStringVal("similarityFunction"),
						SI_ConstStringVal(IndexField_OptionsGetSimFunc(field) == VecSimMetric_L2 ? "euclidean" : "cosine"));
				Map_Add(&prop_types, SI_ConstStringVal("M"),
						SI_LongVal(IndexField_OptionsGetM(field)));
				Map_Add(&prop_types, SI_ConstStringVal("efConstruction"),
						SI_LongVal(IndexField_OptionsGetEfConstruction(field)));
				Map_Add(&prop_types, SI_ConstStringVal("efRuntime"),
						SI_LongVal(IndexField_OptionsGetEfRuntime(field)));
			}

			Map_Add(ctx->yield_options,
					SI_ConstStringVal(IndexField_GetName(field)), prop_types);

			SIValue_Free(prop_types);
		}
	}

	//--------------------------------------------------------------------------
	// index language
	//--------------------------------------------------------------------------

	if(ctx->yield_language) {
		*ctx->yield_language =
			SI_ConstStringVal((char *)Index_GetLanguage(idx));
	}

	//--------------------------------------------------------------------------
	// index stopwords
	//--------------------------------------------------------------------------

	if(ctx->yield_stopwords) {
		size_t stopwords_count;
		char **stopwords = Index_GetStopwords(idx, &stopwords_count);
		if(stopwords) {
			*ctx->yield_stopwords = SI_Array(stopwords_count);
			for(size_t i = 0; i < stopwords_count; i++) {
				SIValue value = SI_ConstStringVal(stopwords[i]);
				SIArray_Append(ctx->yield_stopwords, value);
				rm_free(stopwords[i]);
			}
		} else {
			*ctx->yield_stopwords = SI_Array(0);
		}
		rm_free(stopwords);
	}

	//--------------------------------------------------------------------------
	// index info
	//--------------------------------------------------------------------------

	if(ctx->yield_info) {
		SIValue map = SI_Map(23);

		Map_Add(&map, SI_ConstStringVal("gcPolicy"), SI_LongVal(info.gcPolicy));
		Map_Add(&map, SI_ConstStringVal("score"),    SI_DoubleVal(info.score));
		Map_Add(&map, SI_ConstStringVal("lang"),     SI_ConstStringVal(info.lang));

		SIValue fields = SIArray_New(info.numFields);
		for (uint i = 0; i < info.numFields; i++) {
			struct RSIdxField f = info.fields[i];
			SIValue field = SI_Map(6);
			Map_Add(&field, SI_ConstStringVal("path"),             SI_ConstStringVal(f.path));
			Map_Add(&field, SI_ConstStringVal("name"),             SI_ConstStringVal(f.name));
			Map_Add(&field, SI_ConstStringVal("options"),          SI_LongVal(f.options));
			Map_Add(&field, SI_ConstStringVal("textWeight"),       SI_DoubleVal(f.textWeight));
			Map_Add(&field, SI_ConstStringVal("tagCaseSensitive"), SI_BoolVal(f.tagCaseSensitive));
			SIArray_Append(&fields, field);
			SIValue_Free(field);
		}
		Map_Add(&map, SI_ConstStringVal("fields"), fields);
		SIValue_Free(fields);

		Map_Add(&map, SI_ConstStringVal("numDocuments"),     SI_LongVal(info.numDocuments));
		Map_Add(&map, SI_ConstStringVal("maxDocId"),         SI_LongVal(info.maxDocId));
		Map_Add(&map, SI_ConstStringVal("docTableSize"),     SI_LongVal(info.docTableSize));
		Map_Add(&map, SI_ConstStringVal("sortablesSize"),    SI_LongVal(info.sortablesSize));
		Map_Add(&map, SI_ConstStringVal("docTrieSize"),      SI_LongVal(info.docTrieSize));
		Map_Add(&map, SI_ConstStringVal("numTerms"),         SI_LongVal(info.numTerms));
		Map_Add(&map, SI_ConstStringVal("numRecords"),       SI_LongVal(info.numRecords));
		Map_Add(&map, SI_ConstStringVal("invertedSize"),     SI_LongVal(info.invertedSize));
		Map_Add(&map, SI_ConstStringVal("invertedCap"),      SI_LongVal(info.invertedCap));
		Map_Add(&map, SI_ConstStringVal("skipIndexesSize"),  SI_LongVal(info.skipIndexesSize));
		Map_Add(&map, SI_ConstStringVal("scoreIndexesSize"), SI_LongVal(info.scoreIndexesSize));
		Map_Add(&map, SI_ConstStringVal("offsetVecsSize"),   SI_LongVal(info.offsetVecsSize));
		Map_Add(&map, SI_ConstStringVal("offsetVecRecords"), SI_LongVal(info.offsetVecRecords));
		Map_Add(&map, SI_ConstStringVal("termsSize"),        SI_LongVal(info.termsSize));
		Map_Add(&map, SI_ConstStringVal("indexingFailures"), SI_LongVal(info.indexingFailures));
		Map_Add(&map, SI_ConstStringVal("totalCollected"),   SI_LongVal(info.totalCollected));
		Map_Add(&map, SI_ConstStringVal("numCycles"),        SI_LongVal(info.numCycles));
		Map_Add(&map, SI_ConstStringVal("totalMSRun"),       SI_LongVal(info.totalMSRun));
		Map_Add(&map, SI_ConstStringVal("lastRunTimeMs"),    SI_LongVal(info.lastRunTimeMs));

		*ctx->yield_info = map;
	}

	// clean up
	RediSearch_IndexInfoFree(&info);

	return true;
}

// emit a graph-level CCH path index. CCH indices are not RediSearch-backed and
// have no label/fields in the usual sense, so the RediSearch-specific columns
// (options/language/stopwords/info) are reported empty. build is synchronous,
// so a registered CCH index is always OPERATIONAL.
static bool _EmitCCHIndex
(
	IndexesContext *ctx,
	const CCHIndex *idx
) {
	GraphContext *gc = ctx->gc;

	uint              n     = CCHIndex_RelTypeCount(idx);
	const RelationID *rels  = CCHIndex_RelTypes(idx);
	const char       *wname = GraphContext_GetAttributeName(gc,
			CCHIndex_WeightAttr(idx));

	if(ctx->yield_entity_type) {
		*ctx->yield_entity_type = SI_ConstStringVal("RELATIONSHIP");
	}

	if(ctx->yield_status) {
		*ctx->yield_status = SI_ConstStringVal("OPERATIONAL");
	}

	// label: the relationship types the index spans, joined by ','
	if(ctx->yield_label) {
		size_t cap = 1;
		for(uint i = 0; i < n; i++) {
			Schema *s = GraphContext_GetSchemaByID(gc, rels[i], SCHEMA_EDGE);
			cap += strlen(Schema_GetName(s)) + 1;
		}
		char *buf = rm_malloc(cap);
		buf[0] = '\0';
		for(uint i = 0; i < n; i++) {
			Schema *s = GraphContext_GetSchemaByID(gc, rels[i], SCHEMA_EDGE);
			if(i > 0) strcat(buf, ",");
			strcat(buf, Schema_GetName(s));
		}
		*ctx->yield_label = SI_DuplicateStringVal(buf);
		rm_free(buf);
	}

	// properties: the single weight attribute
	if(ctx->yield_fields) {
		*ctx->yield_fields = SI_Array(1);
		SIArray_Append(ctx->yield_fields, SI_ConstStringVal((char *)wname));
	}

	// types: { weightProp: ['CCH'] }
	if(ctx->yield_types) {
		*ctx->yield_types = SI_Map(1);
		SIValue t = SI_Array(1);
		SIArray_Append(&t, SI_ConstStringVal("CCH"));
		Map_Add(ctx->yield_types, SI_ConstStringVal((char *)wname), t);
		SIValue_Free(t);
	}

	// RediSearch-specific columns -- not applicable to a CCH index
	if(ctx->yield_options)   *ctx->yield_options   = SI_Map(0);
	if(ctx->yield_language)  *ctx->yield_language  = SI_ConstStringVal("");
	if(ctx->yield_stopwords) *ctx->yield_stopwords = SI_Array(0);
	if(ctx->yield_info)      *ctx->yield_info      = SI_Map(0);

	return true;
}

SIValue *Proc_IndexesStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	IndexesContext *pdata = ctx->privateData;

	// emit RediSearch-backed indices first
	if(arr_len(pdata->indices) > 0) {
		Index idx = arr_pop(pdata->indices);
		_EmitIndex(pdata, idx);
		return pdata->out;
	}

	// then graph-level CCH path indices
	if(arr_len(pdata->cch_indices) > 0) {
		CCHIndex *idx = arr_pop(pdata->cch_indices);
		_EmitCCHIndex(pdata, idx);
		return pdata->out;
	}

	// no more indices to emit
	return NULL;
}

ProcedureResult Proc_IndexesFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		IndexesContext *pdata = ctx->privateData;
		arr_free(pdata->indices);
		arr_free(pdata->cch_indices);
		rm_free(pdata);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_IndexesCtx(void) {
	void *privateData = NULL;
	ProcedureOutput output;
	ProcedureOutput *outputs = arr_new(ProcedureOutput, 9);

	// indexed label
	output = (ProcedureOutput) {
		.name = "label", .type = T_STRING
	};
	arr_append(outputs, output);

	// indexed properties
	output = (ProcedureOutput) {
		.name = "properties", .type = T_ARRAY
	};
	arr_append(outputs, output);

	// indexed fields types
	output = (ProcedureOutput) {
		.name = "types", .type = T_MAP
	};
	arr_append(outputs, output);

	// indexed fields options
	output = (ProcedureOutput) {
		.name = "options", .type = T_MAP
	};
	arr_append(outputs, output);

	// indexed language
	output = (ProcedureOutput) {
		.name = "language", .type = T_STRING
	};
	arr_append(outputs, output);

	// indexed stopwords
	output = (ProcedureOutput) {
		.name = "stopwords", .type = T_ARRAY
	};
	arr_append(outputs, output);

	// index entity type (node / relationship)
	output = (ProcedureOutput) {
		.name = "entitytype", .type = T_STRING
	};
	arr_append(outputs, output);

	// index status (operational / under construction)
	output = (ProcedureOutput) {
		.name = "status", .type = T_STRING
	};
	arr_append(outputs, output);

	// index info
	output = (ProcedureOutput) {
		.name = "info", .type = T_MAP
	};
	arr_append(outputs, output);

	ProcedureCtx *ctx = ProcCtxNew("db.indexes",
								   0,
								   outputs,
								   Proc_IndexesStep,
								   Proc_IndexesInvoke,
								   Proc_IndexesFree,
								   privateData,
								   true);
	return ctx;
}

