/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "record.h"
#include "../errors/errors.h"
#include "../util/rmalloc.h"

Record Record_New
(
	rax *mapping
) {
	ASSERT(mapping != NULL);

	// determine record size
	uint entries_count = raxSize(mapping);
	uint rec_size = sizeof(_Record) + sizeof(Entry) * entries_count;

	Record r = rm_calloc(1, rec_size);
	r->mapping = mapping;

	return r;
}

// returns the number of entries held by record
uint Record_length
(
	const Record r
) {
	ASSERT(r);
	return raxSize(r->mapping);
}

bool Record_ContainsEntry
(
	const Record r,
	uint idx
) {
	ASSERT(idx < Record_length(r));
	return r->entries[idx].type != REC_TYPE_UNKNOWN;
}

// retrieve the offset into the Record of the given alias
uint Record_GetEntryIdx (
	Record r,
	const char *alias,
	size_t len
) {
	ASSERT(r && alias);

	void *idx = raxFind(r->mapping, (unsigned char *)alias, len);

	return idx != raxNotFound ? (intptr_t)idx : INVALID_INDEX;
}

void Record_Clone
(
	const restrict Record r,
	restrict Record clone
) {
	// r and clone share the same record mapping
	if(likely(r->owner == clone->owner)) {
		ASSERT(Record_length(r) <= Record_length(clone));
		size_t required_record_size = sizeof(Entry) * Record_length(r);
		memcpy(clone->entries, r->entries, required_record_size);

		// foreach scalar entry in cloned record, make sure it is not freed
		// it is the original record owner responsibility to free the record
		// and its internal scalar as a result
		//
	} else {
		// r and clone don't share the same mappings
		// scan through each entry within r
		// locate coresponding entry in clone
		// if such exists shallow clone it
		raxIterator it;
		raxStart(&it, clone->mapping);
		raxSeek(&it, "^", NULL, 0);

		while(raxNext(&it)) {
			uint src_idx = Record_GetEntryIdx(r, (const char*)it.key, it.key_len);

			if(src_idx == INVALID_INDEX) continue;
			if(Record_GetType(r, src_idx) == REC_TYPE_UNKNOWN) continue;

			intptr_t target_idx = (intptr_t)it.data;
			Record_Add(clone, target_idx, Record_Get(r, src_idx));
		}

		raxStop(&it);
	}

	// TODO: i wish we wouldn't have to perform this loop
	// as it is a major performance hit
	// with the introduction of a garbage collection this should be removed
	int entry_count = Record_length(clone);
	for(int i = 0; i < entry_count; i++) {
		if(Record_GetType(clone, i) == REC_TYPE_SCALAR) {
			SIValue_MakeVolatile(&clone->entries[i].value.s);
		}
	}
}

// merge src record into dest
void Record_Merge
(
	restrict Record dest,      // dest record
	const restrict Record src  // src record
) {
	ASSERT(src->owner == dest->owner);

	uint len = Record_length(src);
	for(uint i = 0; i < len; i++) {
		RecordEntryType src_type  = src->entries[i].type;
		RecordEntryType dest_type = dest->entries[i].type;

		if(src_type != REC_TYPE_UNKNOWN && dest_type == REC_TYPE_UNKNOWN) {
			// copy entry from src to dest
			dest->entries[i] = src->entries[i];

			// protect heap allocated values
			if(src->entries[i].type == REC_TYPE_SCALAR) {
				if(SI_ALLOCATION(&(src->entries[i].value.s)) == M_SELF) {
					SIValue_MakeVolatile(&(src->entries[i].value.s));
				} else {
					SIValue_Persist(&(dest->entries[i].value.s));
				}
			}
		}
	}
}

// duplicates entries from `src` into `dest`
void Record_DuplicateEntries
(
	restrict Record dest,      // destination record
	restrict const Record src  // src record
) {
	ASSERT(src->owner == dest->owner);

	uint len = Record_length(src);
	for(uint i = 0; i < len; i++) {
		if(src->entries[i].type != REC_TYPE_UNKNOWN && 
		   dest->entries[i].type == REC_TYPE_UNKNOWN) {
			// copy the entry
			dest->entries[i] = src->entries[i];
			if(dest->entries[i].type == REC_TYPE_SCALAR) {
				dest->entries[i].value.s =
					SI_CloneValue(dest->entries[i].value.s);
			}
		}
	}
}

RecordEntryType Record_GetType
(
	const Record r,
	uint idx
) {
	ASSERT(Record_length(r) > idx);
	return r->entries[idx].type;
}

Node *Record_GetNode
(
	const Record r,
	uint idx
) {
	switch(r->entries[idx].type) {
		case REC_TYPE_NODE:
			return &(r->entries[idx].value.n);
		case REC_TYPE_UNKNOWN:
			return NULL;
		case REC_TYPE_SCALAR:
			// Null scalar values are expected here; otherwise fall through.
			if(SIValue_IsNull(r->entries[idx].value.s)) return NULL;
		default:
			ErrorCtx_RaiseRuntimeException("encountered unexpected type in Record; expected Node");
			return NULL;
	}
}

Edge *Record_GetEdge
(
	const Record r,
	uint idx
) {
	switch(r->entries[idx].type) {
		case REC_TYPE_EDGE:
			return &(r->entries[idx].value.e);
		case REC_TYPE_UNKNOWN:
			return NULL;
		case REC_TYPE_SCALAR:
			// Null scalar values are expected here; otherwise fall through.
			if(SIValue_IsNull(r->entries[idx].value.s)) return NULL;
		default:
			ErrorCtx_RaiseRuntimeException("encountered unexpected type in Record; expected Edge");
			return NULL;
	}
}

SIValue Record_Get
(
	Record r,
	uint idx
) {
	ASSERT(Record_length(r) > idx);

	Entry e = r->entries[idx];
	switch(e.type) {
		case REC_TYPE_NODE:
			return SI_Node(Record_GetNode(r, idx));
		case REC_TYPE_EDGE:
			return SI_Edge(Record_GetEdge(r, idx));
		case REC_TYPE_SCALAR:
			return r->entries[idx].value.s;
		case REC_TYPE_UNKNOWN:
			return SI_NullVal();
		default:
			ASSERT(false);
			return SI_NullVal();
	}
}

// get a node from record
// if record[idx] is unset, set its type to REC_TYPE_NODE
Node *Record_GetSetNode
(
	Record r,  // record
	uint idx   // entry index
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r));

	Entry *e = r->entries + idx ;
	if (e->type == REC_TYPE_UNKNOWN) {
		e->type = REC_TYPE_NODE ;
	}

	if (unlikely (e->type != REC_TYPE_NODE)) {
		ErrorCtx_RaiseRuntimeException (
				"encountered unexpected record type when trying to retrieve node") ;
		return NULL ;
	}

	return &(e->value.n) ;
}

void Record_Remove
(
	Record r,
	uint idx
) {
	r->entries[idx].type = REC_TYPE_UNKNOWN;
}

GraphEntity *Record_GetGraphEntity
(
	const Record r,
	uint idx
) {
	Entry e = r->entries[idx];
	switch(e.type) {
		case REC_TYPE_NODE:
			return (GraphEntity *)Record_GetNode(r, idx);
		case REC_TYPE_EDGE:
			return (GraphEntity *)Record_GetEdge(r, idx);
		default:
			ErrorCtx_RaiseRuntimeException("encountered unexpected type when trying to retrieve graph entity");
	}
	return NULL;
}

void Record_Add
(
	Record r,
	uint idx,
	SIValue v
) {
	ASSERT(idx < Record_length(r));
	switch(SI_TYPE(v)) {
		case T_NODE:
			Record_AddNode(r, idx, *(Node *)v.ptrval);
			break;
		case T_EDGE:
			Record_AddEdge(r, idx, *(Edge *)v.ptrval);
			break;
		default:
			Record_AddScalar(r, idx, v);
			break;
	}
}

SIValue *Record_AddScalar
(
	Record r,
	uint idx,
	SIValue v
) {
	r->entries[idx].value.s = v;
	r->entries[idx].type = REC_TYPE_SCALAR;
	return &(r->entries[idx].value.s);
}

Node *Record_AddNode
(
	Record r,
	uint idx,
	Node node
) {
	r->entries[idx].value.n = node;
	r->entries[idx].type = REC_TYPE_NODE;
	return &(r->entries[idx].value.n);
}

Edge *Record_AddEdge
(
	Record r,
	uint idx,
	Edge edge
) {
	r->entries[idx].value.e = edge;
	r->entries[idx].type = REC_TYPE_EDGE;
	return &(r->entries[idx].value.e);
}

size_t Record_ToString
(
	const Record r,
	char **buf,
	size_t *buf_cap
) {
	uint rLen = Record_length(r);
	SIValue values[rLen];
	for(int i = 0; i < rLen; i++) {
		if(Record_GetType(r, i) == REC_TYPE_UNKNOWN) {
			values[i] = SI_ConstStringVal("UNKNOWN");
		} else {
			values[i] = Record_Get(r, i);
		}
	}

	size_t required_len = SIValue_StringJoinLen(values, rLen, ",");

	if(*buf_cap < required_len) {
		*buf = rm_realloc(*buf, sizeof(char) * required_len);
		*buf_cap = required_len;
	}

	size_t bytesWritten = 0;
	SIValue_StringJoin(values, rLen, ",", buf, buf_cap, &bytesWritten);
	return bytesWritten;
}

inline rax *Record_GetMappings
(
	const Record r
) {
	ASSERT(r != NULL);
	return r->mapping;
}

inline void Record_FreeEntry
(
	Record r,
	int idx
) {
	if(r->entries[idx].type == REC_TYPE_SCALAR) SIValue_Free(r->entries[idx].value.s);
	r->entries[idx].type = REC_TYPE_UNKNOWN;
}

void Record_FreeEntries
(
	Record r
) {
	uint length = Record_length(r);
	for(uint i = 0; i < length; i++) {
		// free any allocations held by this Record
		Record_FreeEntry(r, i);
	}
}

// increase record's reference count
void Record_IncRefCount
(
	Record r
) {
	ASSERT (r != NULL) ;
	ASSERT (r->ref_count > 0) ;  // can't revive a freed record

	r->ref_count++ ;
}

void Record_Free
(
	Record r
) {
	ASSERT(r != NULL);
	ASSERT(r->ref_count == 0);

	Record_FreeEntries(r);
	rm_free(r);
}

