/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "index.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/point.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/node.h"
#include "../graph/rg_matrix/rg_matrix_iter.h"

#include <stdatomic.h>

struct _Index {
	char *label;                   // indexed label
	int label_id;                  // indexed label ID
	IndexField *fields;            // indexed fields
	char *language;                // language
	char **stopwords;              // stopwords
	GraphEntityType entity_type;   // entity type (node/edge) indexed
	IndexType type;                // index type exact-match / fulltext
	RSIndex *rsIdx;                // RediSearch index
	uint _Atomic pending_changes;  // number of pending changes
};

static void _Index_ConstructFullTextStructure
(
	Index idx,
	RSIndex *rsIdx
) {
	ASSERT(idx != NULL);
	ASSERT(rsIdx != NULL);

	uint fields_count = array_len(idx->fields);

	for(uint i = 0; i < fields_count; i++) {
		IndexField *field = idx->fields + i;
		// introduce text field
		unsigned options = RSFLDOPT_NONE;
		if(field->nostem) {
			options |= RSFLDOPT_TXTNOSTEM;
		}

		if(strcmp(field->phonetic, INDEX_FIELD_DEFAULT_PHONETIC) != 0) {
			options |= RSFLDOPT_TXTPHONETIC;
		}

		RSFieldID fieldID = RediSearch_CreateField(rsIdx, field->name,
				RSFLDTYPE_FULLTEXT, options);
		RediSearch_TextFieldSetWeight(rsIdx, fieldID, field->weight);
	}
}

static void _Index_ConstructExactMatchStructure
(
	Index idx,
	RSIndex *rsIdx
) {
	ASSERT(idx != NULL);
	ASSERT(rsIdx != NULL);

	RSFieldID fieldID;
	uint fields_count = array_len(idx->fields);
	unsigned types = RSFLDTYPE_NUMERIC | RSFLDTYPE_GEO | RSFLDTYPE_TAG;

	for(uint i = 0; i < fields_count; i++) {
		IndexField *field = idx->fields + i;
		// introduce both text, numeric and geo fields
		fieldID = RediSearch_CreateField(rsIdx, field->name, types,
				RSFLDOPT_NONE);
		RediSearch_TagFieldSetSeparator(rsIdx, fieldID, INDEX_SEPARATOR);
		RediSearch_TagFieldSetCaseSensitive(rsIdx, fieldID, 1);
	}

	// introduce edge src and dest node ids as additional index fields
	if(idx->entity_type == GETYPE_EDGE) {
		RediSearch_CreateField(rsIdx, "_src_id", RSFLDTYPE_NUMERIC,
				RSFLDOPT_NONE);
		RediSearch_CreateField(rsIdx, "_dest_id", RSFLDTYPE_NUMERIC,
				RSFLDOPT_NONE);
	}

	// for none indexable types e.g. Array introduce an additional field
	// "none_indexable_fields" which will hold a list of attribute names
	// that were not indexed
	fieldID = RediSearch_CreateField(rsIdx, INDEX_FIELD_NONE_INDEXED,
			RSFLDTYPE_TAG, RSFLDOPT_NONE);
	RediSearch_TagFieldSetSeparator(rsIdx, fieldID, INDEX_SEPARATOR);
	RediSearch_TagFieldSetCaseSensitive(rsIdx, fieldID, 1);
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

	if(idx->stopwords) {
		RediSearch_IndexOptionsSetStopwords(idx_options,
				(const char**)idx->stopwords, array_len(idx->stopwords));
	} else if(idx->type == IDX_EXACT_MATCH) {
		RediSearch_IndexOptionsSetStopwords(idx_options, NULL, 0);
	}

	rsIdx = RediSearch_CreateIndex(idx->label, idx_options);
	RediSearch_FreeIndexOptions(idx_options);

	// create indexed fields
	if(idx->type == IDX_FULLTEXT) {
		_Index_ConstructFullTextStructure(idx, rsIdx);
	} else {
		_Index_ConstructExactMatchStructure(idx, rsIdx);
	}

	// set RediSearch index
	ASSERT(idx->rsIdx == NULL);
	idx->rsIdx = rsIdx;
}

RSDoc *Index_IndexGraphEntity
(
	Index idx,
	const GraphEntity *e,
	const void *key,
	size_t key_len,
	uint *doc_field_count
) {
	ASSERT(idx              !=  NULL);
	ASSERT(e                !=  NULL);
	ASSERT(key              !=  NULL);
	ASSERT(doc_field_count  !=  NULL);
	ASSERT(key_len          >   0);

	double     score       = 1;     // default score
	IndexField *field      = NULL;  // current indexed field
	SIValue    *v          = NULL;  // current indexed value
	EntityID   id          = ENTITY_GET_ID(e);
	uint       field_count = array_len(idx->fields);

	*doc_field_count = 0;

	// list of none indexable fields
	uint none_indexable_fields_count = 0; // number of none indexed fields
	const char *none_indexable_fields[field_count]; // none indexed fields

	// create an empty document
	RSDoc *doc = RediSearch_CreateDocument2(key, key_len, NULL, score,
			idx->language);

	// add document field for each indexed property
	if(idx->type == IDX_FULLTEXT) {
		for(uint i = 0; i < field_count; i++) {
			field = idx->fields + i;
			const char *field_name = field->name;
			v = GraphEntity_GetProperty(e, field->id);
			if(v == ATTRIBUTE_NOTFOUND) continue;

			SIType t = SI_TYPE(*v);

			// value must be of type string
			if(t == T_STRING) {
				*doc_field_count += 1;
				RediSearch_DocumentAddFieldString(doc, field_name, v->stringval,
						strlen(v->stringval), RSFLDTYPE_FULLTEXT);
			}
		}
	} else {
		for(uint i = 0; i < field_count; i++) {
			field = idx->fields + i;
			const char *field_name = field->name;
			v = GraphEntity_GetProperty(e, field->id);
			if(v == ATTRIBUTE_NOTFOUND) continue;

			SIType t = SI_TYPE(*v);

			*doc_field_count += 1;
			if(t == T_STRING) {
				RediSearch_DocumentAddFieldString(doc, field_name, v->stringval,
						strlen(v->stringval), RSFLDTYPE_TAG);
			} else if(t & (SI_NUMERIC | T_BOOL)) {
				double d = SI_GET_NUMERIC(*v);
				RediSearch_DocumentAddFieldNumber(doc, field_name, d,
						RSFLDTYPE_NUMERIC);
			} else if(t == T_POINT) {
				double lat = (double)Point_lat(*v);
				double lon = (double)Point_lon(*v);
				RediSearch_DocumentAddFieldGeo(doc, field_name, lat, lon,
						RSFLDTYPE_GEO);
			} else {
				// none indexable field
				none_indexable_fields[none_indexable_fields_count++] =
					field_name;
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
	}

	return doc;
}

void IndexField_New
(
	IndexField *field,    // field to initialize
	Attribute_ID id,      // attribute ID
	const char *name,     // field name
	double weight,        // field weight
	bool nostem,          // no stemming
	const char *phonetic  // phonetic
) {
	ASSERT(name     != NULL);
	ASSERT(field    != NULL);
	ASSERT(phonetic != NULL);

	field->id       = id;
	field->name     = rm_strdup(name);
	field->weight   = weight;
	field->nostem   = nostem;
	field->phonetic = rm_strdup(phonetic);
}

void IndexField_Free
(
	IndexField *field
) {
	ASSERT(field != NULL);

	rm_free(field->name);
	rm_free(field->phonetic);
}

// create a new index
Index Index_New
(
	const char *label,           // indexed label
	int label_id,                // indexed label id
	IndexType type,              // exact match or full text
	GraphEntityType entity_type  // entity type been indexed
) {
	ASSERT(label != NULL);

	Index idx = rm_malloc(sizeof(_Index));

	idx->type            = type;
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

	//--------------------------------------------------------------------------
	// clone index fields
	//--------------------------------------------------------------------------

	int n = array_len(idx->fields);
	clone->fields = array_new(IndexField, 1);
	for(int i = 0; i < n; i++) {
		IndexField _f;
		IndexField *f = idx->fields + i;
		IndexField_New(&_f, f->id, f->name, f->weight, f->nostem, f->phonetic);
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
void Index_AddField
(
	Index idx,         // index to update
	IndexField *field  // field to add
) {
	ASSERT(idx   != NULL);
	ASSERT(field != NULL);
	ASSERT(!Index_ContainsAttribute(idx, field->id));

	array_append(idx->fields, *field);
}

// removes fields from index
void Index_RemoveField
(
	Index idx,            // index modified
	Attribute_ID attr_id  // field to remove
) {
	ASSERT(idx != NULL);

	uint fields_count = array_len(idx->fields);
	for(uint i = 0; i < fields_count; i++) {
		IndexField *field = idx->fields + i;
		if(field->id == attr_id) {
			// free field
			IndexField_Free(field);
			array_del_fast(idx->fields, i);

			Index_Disable(idx);
			break;
		}
	}
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

// returns index type
IndexType Index_Type
(
	const Index idx
) {
	ASSERT(idx != NULL);

	return idx->type;
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

bool Index_ContainsAttribute
(
	const Index idx,
	Attribute_ID attribute_id
) {
	ASSERT(idx != NULL);

	if(attribute_id == ATTRIBUTE_ID_NONE) return false;
	
	uint fields_count = array_len(idx->fields);
	for(uint i = 0; i < fields_count; i++) {
		IndexField *field = idx->fields + i;
		if(field->id == attribute_id) return true;
	}

	return false;
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
	ASSERT(_idx != NULL);

	return RediSearch_IndexGetLanguage(_idx);
}

char **Index_GetStopwords
(
	const Index idx,
	size_t *size
) {
	ASSERT(idx != NULL);

	RSIndex *_idx = Index_RSIndex(idx);
	ASSERT(_idx != NULL);

	if(idx->type == IDX_FULLTEXT) {
		return RediSearch_IndexGetStopwords(_idx, size);
	}

	return NULL;
}

// set indexed language
void Index_SetLanguage
(
	Index idx,
	const char *language
) {
	ASSERT(idx != NULL);
	ASSERT(language != NULL);
	ASSERT(idx->language == NULL);

	idx->language = rm_strdup(language);
}

// set indexed stopwords
void Index_SetStopwords
(
	Index idx,
	char **stopwords
) {
	ASSERT(idx != NULL);
	ASSERT(stopwords != NULL);
	ASSERT(idx->stopwords == NULL);

	idx->stopwords = stopwords;
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

	if(idx->language) {
		rm_free(idx->language);
	}

	uint fields_count = array_len(idx->fields);
	for(uint i = 0; i < fields_count; i++) {
		IndexField_Free(idx->fields + i);
	}
	array_free(idx->fields);

	if(idx->stopwords) {
		uint stopwords_count = array_len(idx->stopwords);
		for(uint i = 0; i < stopwords_count; i++) {
			rm_free(idx->stopwords[i]);
		}
		array_free(idx->stopwords);
	}

	rm_free(idx->label);
	rm_free(idx);
}

