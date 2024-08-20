/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/point.h"
#include "../datatypes/vector.h"

#include <stdatomic.h>

// gets type aware index field name
void Index_RangeFieldName
(
	char *type_aware_name,  // [out] type aware name
	const char *name        // field name
) {
	ASSERT(name != NULL);
	ASSERT(type_aware_name != NULL);

	// prefix range field name with "range:"
	sprintf(type_aware_name, "range:%s", name);
}

// gets type aware index field name
void Index_FulltextxFieldName
(
	char *type_aware_name,  // [out] type aware name
	const char *name        // field name
) {
	ASSERT(name != NULL);
	ASSERT(type_aware_name != NULL);

	// maintain original name for full text fields
	strcpy(type_aware_name, name);
}

// gets type aware index field name
void Index_VectorFieldName
(
	char *type_aware_name,  // [out] type aware name
	const char *name        // field name
) {
	ASSERT(name != NULL);
	ASSERT(type_aware_name != NULL);

	// prefix vector field name with "vector:"
	sprintf(type_aware_name, "vector:%s", name);
}

// index structure
struct _Index {
	char *label;                   // indexed label
	int label_id;                  // indexed label ID
	IndexField *fields;            // indexed fields
	char *language;                // language
	char **stopwords;              // stopwords
	GraphEntityType entity_type;   // entity type (node/edge) indexed
	RSIndex *rsIdx;                // RediSearch index
	uint _Atomic pending_changes;  // number of pending changes
};

// merge field 'b' into 'a'
static void _Index_MergeFields
(
	IndexField *a,
	const IndexField *b
) {
	ASSERT(a != NULL);
	ASSERT(b != NULL);
	ASSERT((a->type & b->type) == 0);

	// merge type
	a->type |= b->type;

	// merge options
	if(b->type & INDEX_FLD_FULLTEXT) {
		a->options.weight   = b->options.weight;
		a->options.nostem   = b->options.nostem;
		a->fulltext_name    = a->name;
		IndexField_OptionsSetPhonetic(a, b->options.phonetic);
	} else if(b->type & INDEX_FLD_RANGE) {
		a->range_name        = rm_strdup(b->range_name);
	} else if(b->type & INDEX_FLD_VECTOR) {
		a->hnsw_options.dimension = b->hnsw_options.dimension;
		a->vector_name = rm_strdup(b->vector_name);
	} else {
		assert(false && "unexpected field type");
	}
}

static void _Index_ConstructStructure
(
	Index idx,
	RSIndex *rsIdx
) {
	ASSERT(idx != NULL);
	ASSERT(rsIdx != NULL);

	uint fields_count = array_len(idx->fields);

	for(uint i = 0; i < fields_count; i++) {
		IndexField *field = idx->fields + i;

		//----------------------------------------------------------------------
		// fulltext field
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_FULLTEXT) {
			// introduce text field
			unsigned options = RSFLDOPT_NONE;

			if(field->options.nostem) {
				options |= RSFLDOPT_TXTNOSTEM;
			}

			if(strcmp(field->options.phonetic,
						INDEX_FIELD_DEFAULT_PHONETIC) != 0) {
				options |= RSFLDOPT_TXTPHONETIC;
			}

			RSFieldID fieldID = RediSearch_CreateField(rsIdx,
					field->fulltext_name, RSFLDTYPE_FULLTEXT, options);

			RediSearch_TextFieldSetWeight(rsIdx, fieldID, field->options.weight);
		}

		//----------------------------------------------------------------------
		// vector field
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_VECTOR) {
			RSFieldID fieldID = RediSearch_CreateVectorField(rsIdx,
					field->vector_name);

			RediSearch_VectorFieldSetDim(rsIdx, fieldID, field->hnsw_options.dimension);
			RediSearch_VectorFieldSetHNSWParams(rsIdx, fieldID, IndexField_OptionsGetM(field), IndexField_OptionsGetEfConstruction(field), IndexField_OptionsGetEfRuntime(field), IndexField_OptionsGetSimFunc(field));
		}

		//----------------------------------------------------------------------
		// numeric, string and geo fields
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_RANGE) {
			// introduce both text, numeric and geo fields
			unsigned types = RSFLDTYPE_NUMERIC | RSFLDTYPE_GEO | RSFLDTYPE_TAG;

			RSFieldID fieldID = RediSearch_CreateField(rsIdx, field->range_name,
					types, RSFLDOPT_NONE);

			RediSearch_TagFieldSetSeparator(rsIdx, fieldID, INDEX_SEPARATOR);
			RediSearch_TagFieldSetCaseSensitive(rsIdx, fieldID, 1);
		}
	}

	//--------------------------------------------------------------------------
	// none indexable types
	//--------------------------------------------------------------------------

	// for none indexable types e.g. Array introduce an additional field
	// "none_indexable_fields" which will hold a list of attribute names
	// that were not indexed
	RSFieldID fieldID = RediSearch_CreateField(rsIdx, INDEX_FIELD_NONE_INDEXED,
			RSFLDTYPE_TAG, RSFLDOPT_NONE);
	RediSearch_TagFieldSetSeparator(rsIdx, fieldID, INDEX_SEPARATOR);
	RediSearch_TagFieldSetCaseSensitive(rsIdx, fieldID, 1);

	//--------------------------------------------------------------------------
	// edge index specifics
	//--------------------------------------------------------------------------

	// introduce edge src and dest node ids as additional index fields
	if(idx->entity_type == GETYPE_EDGE) {
		RediSearch_CreateField(rsIdx, "range:_src_id", RSFLDTYPE_NUMERIC,
				RSFLDOPT_NONE);
		RediSearch_CreateField(rsIdx, "range:_dest_id", RSFLDTYPE_NUMERIC,
				RSFLDOPT_NONE);
	}
}

// responsible for creating the index structure only!
// e.g. fields, stopwords, language
void Index_ConstructStructure
(
	Index idx
) {
	ASSERT(idx != NULL);
	ASSERT(idx->rsIdx == NULL);

	RSIndex *rsIdx = NULL;
	RSIndexOptions *idx_options = RediSearch_CreateIndexOptions();
	RediSearch_IndexOptionsSetLanguage(idx_options, idx->language);
	// TODO: Remove this comment when https://github.com/RediSearch/RediSearch/issues/1100 is closed
	// RediSearch_IndexOptionsSetGetValueCallback(idx_options, _getNodeAttribute, gc);

	#ifndef MEMCHECK
	// enable GC, every 30 seconds gc will check if there's garbage
	// if there are over 100 docs to remove GC will perform clean up
	RediSearch_IndexOptionsSetGCPolicy(idx_options, GC_POLICY_FORK);
	#endif

	RediSearch_IndexOptionsSetStopwords(idx_options, NULL, 0);
	if(idx->stopwords) {
		RediSearch_IndexOptionsSetStopwords(idx_options,
				(const char**)idx->stopwords, array_len(idx->stopwords));
	}

	rsIdx = RediSearch_CreateIndex(idx->label, idx_options);
	RediSearch_FreeIndexOptions(idx_options);

	// create indexed fields
	_Index_ConstructStructure(idx, rsIdx);

	// set RediSearch index
	ASSERT(idx->rsIdx == NULL);
	idx->rsIdx = rsIdx;
}

// index a graph entity
RSDoc *Index_IndexGraphEntity
(
	Index idx,             // index to populate
	const GraphEntity *e,  // entity to index
	const void *key,       // index document key
	size_t key_len,        // index document key length
	uint *doc_field_count  // number of indexed fields
) {
	ASSERT(e               != NULL);
	ASSERT(idx             != NULL);
	ASSERT(key             != NULL);
	ASSERT(doc_field_count != NULL);
	ASSERT(key_len         >  0);

	double     score       = 1;     // default score
	IndexField *field      = NULL;  // current indexed field
	SIValue    *v          = NULL;  // current indexed value
	uint       field_count = array_len(idx->fields);

	*doc_field_count = 0;  // number of indexed fields

	// list of none indexable fields
	uint none_indexable_fields_count = 0; // number of none indexed fields
	const char *none_indexable_fields[field_count]; // none indexed fields

	// create an empty document
	RSDoc *doc = RediSearch_CreateDocument2(key, key_len, NULL, score,
			idx->language);

	// add document field for each indexed attribute
	for(uint i = 0; i < field_count; i++) {
		field = idx->fields + i;

		// try to get attribute value
		v = GraphEntity_GetProperty(e, field->id);

		// entity does not have this attribute
		if(v == ATTRIBUTE_NOTFOUND) {
			continue;
		}

		SIType t = SI_TYPE(*v);

		//----------------------------------------------------------------------
		// fulltext field
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_FULLTEXT) {
			// value must be of type string
			if(t == T_STRING) {
				*doc_field_count += 1;

				RediSearch_DocumentAddFieldString(doc, field->fulltext_name,
						v->stringval, strlen(v->stringval), RSFLDTYPE_FULLTEXT);
			}
		}

		//----------------------------------------------------------------------
		// range field
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_RANGE) {
			*doc_field_count += 1;
			if(t == T_STRING) {
				RediSearch_DocumentAddFieldString(doc, field->range_name,
						v->stringval, strlen(v->stringval), RSFLDTYPE_TAG);
			} else if(t & (SI_NUMERIC | T_BOOL)) {
				double d = SI_GET_NUMERIC(*v);
				RediSearch_DocumentAddFieldNumber(doc, field->range_name, d,
						RSFLDTYPE_NUMERIC);
			} else if(t == T_POINT) {
				double lat = (double)Point_lat(*v);
				double lon = (double)Point_lon(*v);
				RediSearch_DocumentAddFieldGeo(doc, field->range_name, lat, lon,
						RSFLDTYPE_GEO);
			} else {
				// none indexable field
				none_indexable_fields[none_indexable_fields_count++] =
					field->name;
			}
		}

		//----------------------------------------------------------------------
		// vector field
		//----------------------------------------------------------------------

		if(field->type & INDEX_FLD_VECTOR && (t & T_VECTOR)) {
			// make sure entity vector dimension matches index vector dimension
			if(IndexField_OptionsGetDimension(field) != SIVector_Dim(*v)) {
				// vector dimension mis-match, can't index this vector
				continue;
			}

			*doc_field_count += 1;

			size_t   n        = SIVector_ElementsByteSize(*v);
			uint32_t dim      = SIVector_Dim(*v);
			void*    elements = SIVector_Elements(*v);

			// value must be of type array
			RediSearch_DocumentAddFieldVector(doc, field->vector_name, elements,
					dim, n);
		}
	}

	// index name of none index fields
	if(none_indexable_fields_count > 0) {
		// concat all none indexable field names
		size_t len = none_indexable_fields_count - 1; // seperators
		for(uint i = 0; i < none_indexable_fields_count; i++) {
			len += strlen(none_indexable_fields[i]);
		}

		// add room for \0
		len++;

		char *s = NULL;
		if(len < 512) s = alloca(len); // stack base
		else s = rm_malloc(sizeof(char) * len); // heap base

		// concat
		len = sprintf(s, "%s", none_indexable_fields[0]);
		for(uint i = 1; i < none_indexable_fields_count; i++) {
			len += sprintf(s + len, "%c%s", INDEX_SEPARATOR, none_indexable_fields[i]);
		}

		RediSearch_DocumentAddFieldString(doc, INDEX_FIELD_NONE_INDEXED,
				s, len, RSFLDTYPE_TAG);

		// free if heap based
		if(len >= 512) rm_free(s);
	}

	return doc;
}

// create a new index
Index Index_New
(
	const char *label,           // indexed label
	int label_id,                // indexed label id
	GraphEntityType entity_type  // entity type been indexed
) {
	ASSERT(label != NULL);

	Index idx = rm_malloc(sizeof(_Index));

	idx->label           = rm_strdup(label);
	idx->rsIdx           = NULL;
	idx->fields          = array_new(IndexField, 1);
	idx->label_id        = label_id;
	idx->language        = NULL;
	idx->stopwords       = NULL;
	idx->entity_type     = entity_type;
	idx->pending_changes = ATOMIC_VAR_INIT(0);

	return idx;
}

// clone index
Index Index_Clone
(
	const Index idx  // index to clone
) {
	ASSERT(idx != NULL);
	ASSERT(Index_Enabled(idx));

	//--------------------------------------------------------------------------
	// clone index
	//--------------------------------------------------------------------------

	Index clone = rm_malloc(sizeof(_Index));
	memcpy(clone, idx, sizeof(_Index));

	clone->rsIdx           = NULL;
	clone->label           = rm_strdup(idx->label);
	clone->pending_changes = ATOMIC_VAR_INIT(0);
	
	if(clone->stopwords != NULL) {
		array_clone_with_cb(clone->stopwords, idx->stopwords, rm_strdup);
	}

	if(clone->language != NULL) {
		clone->language = rm_strdup(idx->language);
	}

	//--------------------------------------------------------------------------
	// clone index fields
	//--------------------------------------------------------------------------

	int n = array_len(idx->fields);
	clone->fields = array_new(IndexField, 1);
	for(int i = 0; i < n; i++) {
		IndexField _f;
		IndexField *f = idx->fields + i;
		IndexField_Clone(f, &_f);
		array_append(clone->fields, _f);
	}

	return clone;
}

// returns number of pending changes
int Index_PendingChanges
(
	const Index idx  // index to inquery
) {
	ASSERT(idx != NULL);

	return idx->pending_changes;
}

// disable index by increasing the number of pending changes
// and re-creating the internal RediSearch index
void Index_Disable
(
	Index idx  // index to disable
) {
	ASSERT(idx != NULL);

	idx->pending_changes++;

	// drop index if exists
	if(idx->rsIdx != NULL) {
		RediSearch_DropIndex(idx->rsIdx);
		idx->rsIdx = NULL;
	}

	// construct index structure
	Index_ConstructStructure(idx);
}

// try to enable index by dropping number of pending changes by 1
// the index is enabled once there are no pending changes
void Index_Enable
(
	Index idx
) {
	ASSERT(idx != NULL);
	ASSERT(idx->rsIdx != NULL);
	ASSERT(idx->pending_changes > 0);

	idx->pending_changes--;
}

// adds field to index
int Index_AddField
(
	Index idx,         // index to update
	IndexField *field  // field to add
) {
	ASSERT(idx   != NULL);
	ASSERT(field != NULL);

	// make sure typed field is not already indexed
	ASSERT(Index_ContainsField(idx, field->id, field->type) == false);

	// see if index already contains field
	IndexField *existing_field = Index_GetField(NULL, idx, field->id);

	if(existing_field == NULL) {
		// first time field is introduced
		array_append(idx->fields, *field);
	} else {
		// field exists, merge fields
		_Index_MergeFields(existing_field, field);
		IndexField_Free(field);
	}

	return INDEX_OK;
}

// removes fields from index
void Index_RemoveField
(
	Index idx,            // index modified
	AttributeID attr_id,  // field to remove
	IndexFieldType t      // field type
) {
	ASSERT(idx != NULL);
	ASSERT(t & (INDEX_FLD_RANGE | INDEX_FLD_FULLTEXT | INDEX_FLD_VECTOR));

	int pos = -1;
	IndexField *f = Index_GetField(&pos, idx, attr_id);
	ASSERT(f   != NULL);
	ASSERT(pos != -1);

	// remove type from field
	IndexField_RemoveType(f, t);

	// if field is typeless, remove it from index
	if(f->type == INDEX_FLD_UNKNOWN) {
		// free field
		IndexField_Free(f);
		array_del_fast(idx->fields, pos);
	}

	Index_Disable(idx);
}

// query index
RSResultsIterator *Index_Query
(
	const Index idx,
	const char *query,
	char **err
) {
	ASSERT(idx   != NULL);
	ASSERT(query != NULL);

	return RediSearch_IterateQuery(idx->rsIdx, query, strlen(query), err);
}

// returns index graph entity type
GraphEntityType Index_GraphEntityType
(
	const Index idx
) {
	ASSERT(idx != NULL);

	return idx->entity_type;
}

// returns number of fields indexed
uint Index_FieldsCount
(
	const Index idx
) {
	ASSERT(idx != NULL);

	return array_len(idx->fields);
}

// returns indexed fields
const IndexField *Index_GetFields
(
	const Index idx
) {
	ASSERT(idx != NULL);

	return (const IndexField *)idx->fields;
}

// retrieve field by id
// returns NULL if field does not exist
IndexField *Index_GetField
(
	int *pos,         // [optional out] field index
	const Index idx,  // index to get field from
	AttributeID id    // field attribute id
) {
	ASSERT(idx     != NULL);
	ASSERT(id != ATTRIBUTE_ID_NONE && id != ATTRIBUTE_ID_ALL);

	// set field index to -1
	if(pos != NULL) *pos = -1;

	IndexField *f = NULL;
	uint n = array_len(idx->fields);

	for(uint i = 0; i < n; i++) {
		IndexField *field = idx->fields + i;
		if(field->id == id) {
			f = field;
			// if required, set field index
			if(pos != NULL) *pos = i;
			break;
		}
	}

	return f;
}

// returns indexed field type
// if field is not indexed, INDEX_FLD_UNKNOWN is returned
IndexFieldType Index_GetFieldType
(
	const Index idx,  // index to query
	AttributeID id    // field to retrieve type of
) {
	ASSERT(idx != NULL);

	IndexField *f = Index_GetField(NULL, idx, id);
	return (f == NULL) ? INDEX_FLD_UNKNOWN : f->type;
}

// checks if index contains field
// returns true if field is indexed, false otherwise
bool Index_ContainsField
(
	const Index idx,     // index to query
	AttributeID id,      // field to look for
	IndexFieldType type  // field type to look for
) {
	ASSERT(idx != NULL);

	if(id == ATTRIBUTE_ID_NONE) {
		return false;
	}

	return Index_GetFieldType(idx, id) & type;
}

// returns indexed label
const char *Index_GetLabel
(
	const Index idx  // index to query
) {
	ASSERT(idx != NULL);

	return idx->label;
}

int Index_GetLabelID
(
	const Index idx
) {
	ASSERT(idx != NULL);

	return idx->label_id;
}

const char *Index_GetLanguage
(
	const Index idx
) {
	ASSERT(idx != NULL);

	RSIndex *_idx = Index_RSIndex(idx);
	if(_idx == NULL) return NULL;

	return RediSearch_IndexGetLanguage(_idx);
}

// check if index contains stopwords
bool Index_ContainsStopwords
(
	const Index idx  // index to query
) {
	ASSERT(idx != NULL);

	return idx->stopwords != NULL;
}

char **Index_GetStopwords
(
	const Index idx,
	size_t *size
) {
	ASSERT(idx != NULL);

	RSIndex *_idx = Index_RSIndex(idx);
	if(_idx == NULL) return NULL;

	return RediSearch_IndexGetStopwords(_idx, size);
}

// set indexed language
bool Index_SetLanguage
(
	Index idx,
	const char *language
) {
	ASSERT(idx      != NULL);
	ASSERT(language != NULL);

	// fail if index already has language
	if(idx->language != NULL && strcasecmp(idx->language, language) != 0) {
		ErrorCtx_SetError(EMSG_INDEX_CANT_RECONFIG);
		return false;
	}

	idx->language = rm_strdup(language);
	return true;
}

// set indexed stopwords
bool Index_SetStopwords
(
	Index idx,
	char ***stopwords
) {
	ASSERT(idx != NULL);
	ASSERT(stopwords != NULL && *stopwords != NULL);

	// fail if index already has stopwords
	if(idx->stopwords != NULL) {
		ErrorCtx_SetError(EMSG_INDEX_CANT_RECONFIG);
		return false;
	}

	idx->stopwords = *stopwords;
	*stopwords = NULL;

	return true;
}

// returns true if index doesn't contains any pending changes
bool Index_Enabled
(
	const Index idx  // index to get state of
) {
	ASSERT(idx != NULL);

	return idx->pending_changes == 0;
}

// returns RediSearch index
RSIndex *Index_RSIndex
(
	const Index idx  // index to get internal RediSearch index from
) {
	ASSERT(idx != NULL);
	
	return idx->rsIdx;
}

// free index
void Index_Free
(
	Index idx
) {
	ASSERT(idx != NULL);

	if(idx->rsIdx) {
		RediSearch_DropIndex(idx->rsIdx);
	}

	if(idx->language != NULL) {
		rm_free(idx->language);
	}

	uint fields_count = array_len(idx->fields);
	for(uint i = 0; i < fields_count; i++) {
		IndexField_Free(idx->fields + i);
	}
	array_free(idx->fields);

	if(idx->stopwords != NULL) {
		array_free_cb(idx->stopwords, rm_free);
	}

	rm_free(idx->label);
	rm_free(idx);
}

