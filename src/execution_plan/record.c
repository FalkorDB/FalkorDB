/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "record.h"
#include "../errors/errors.h"
#include "../util/rmalloc.h"

// migrate the entry at the given index in the source Record to the same index
// in the destination. Ownership is transferred according to transfer_ownership
static void _RecordPropagateEntry
(
	Record dest,
	Record src,
	uint idx,
	bool transfer_ownership
) {
	// copy the entry
	dest->entries[idx] = src->entries[idx];

	// ownership is determined according to transfer_ownership
	if(src->entries[idx].type == REC_TYPE_SCALAR) {
		if(transfer_ownership) {
			SIValue_MakeVolatile(&src->entries[idx].value.s);
		} else {
			SIValue_MakeVolatile(&dest->entries[idx].value.s);
		}
	}
}

// this function is currently unused
Record Record_New
(
	rax *mapping
) {
	ASSERT(mapping);
	// determine record size
	uint entries_count = raxSize(mapping);
	uint rec_size = sizeof(_Record);
	rec_size += sizeof(Entry) * entries_count;

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
uint Record_GetEntryIdx
(
	Record r,
	const char *alias
) {
	ASSERT(r && alias);

	void *idx = raxFind(r->mapping, (unsigned char *)alias, strlen(alias));

	return idx != raxNotFound ? (intptr_t)idx : INVALID_INDEX;
}

void Record_Clone
(
	const Record r,
	Record clone
) {
	int entry_count = Record_length(r);
	size_t required_record_size = sizeof(Entry) * entry_count;

	memcpy(clone->entries, r->entries, required_record_size);

	/* Foreach scalar entry in cloned record, make sure it is not freed.
	 * it is the original record owner responsibility to free the record
	 * and its internal scalar as a result.
	 *
	 * TODO: I wish we wouldn't have to perform this loop as it is a major performance hit
	 * with the introduction of a garbage collection this should be removed. */
	for(int i = 0; i < entry_count; i++) {
		if(Record_GetType(clone, i) == REC_TYPE_SCALAR) {
			SIValue_MakeVolatile(&clone->entries[i].value.s);
		}
	}
}

void Record_DeepClone
(
		const Record r,
		Record clone
) {
		int entry_count = Record_length(r);
		size_t required_record_size = sizeof(Entry) * entry_count;

		memcpy(clone->entries, r->entries, required_record_size);

		// Deep copy scalars
		for(uint i = 0; i < entry_count; i++) {
				if(r->entries[i].type == REC_TYPE_SCALAR) {
						clone->entries[i].value.s = SI_CloneValue(r->entries[i].value.s);
				}
		}
}

void Record_Merge
(
	Record a,
	const Record b
) {
	ASSERT(a->owner == b->owner);
	uint len = Record_length(a);

	for(uint i = 0; i < len; i++) {
		RecordEntryType a_type = a->entries[i].type;
		RecordEntryType b_type = b->entries[i].type;

		if(a_type == REC_TYPE_UNKNOWN && b_type != REC_TYPE_UNKNOWN) {
			_RecordPropagateEntry(a, b, i, true);
		}
	}
}

// merge entries from `from` into `to`, transfer ownership if transfer_ownership
// is on
void Record_TransferEntries
(
	Record *to,              // destination record
	Record from,             // src record
	bool transfer_ownership  // transfer ownership of the record to dest or not
) {
	uint len = Record_length(from);
	for(uint i = 0; i < len; i++) {
		if(from->entries[i].type != REC_TYPE_UNKNOWN) {
			_RecordPropagateEntry(*to, from, i, transfer_ownership);
		}
	}
}

RecordEntryType Record_GetType
(
	const Record r,
	uint idx
) {
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

void Record_PersistScalars
(
	Record r
) {
	uint len = Record_length(r);
	for(uint i = 0; i < len; i++) {
		if(r->entries[i].type == REC_TYPE_SCALAR) {
			SIValue_Persist(&r->entries[i].value.s);
		}
	}
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

// this function is currently unused
void Record_Free
(
	Record r
) {
	Record_FreeEntries(r);
	rm_free(r);
}

