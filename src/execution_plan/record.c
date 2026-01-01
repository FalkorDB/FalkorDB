/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "record.h"
#include "../errors/errors.h"
#include "../util/rmalloc.h"

// access to the record's entries types
static inline RecordEntryType* Record_GetTypes
(
	const Record r
) {
    return (RecordEntryType*)r->data ;
}

// access to the record's entries
static inline Entry* Record_GetEntries
(
	const Record r
) {
    // skip the types and the padding
    size_t types_padded = (r->num_entries + 7) & ~7 ;
    return (Entry*)(r->data + types_padded) ;
}

// return record's byte size
static inline size_t Record_ByteSize
(
	const Record r
) {
	ASSERT (r != NULL) ;

	size_t types_padded = (r->num_entries + 7) & ~7 ;  // 8 bytes alignment
	return sizeof (_Record) + types_padded + r->num_entries * sizeof (Entry) ;
}

// create a new record sized to accommodate all entries in the given map
Record Record_New
(
	rax *mapping  // record's mapping
) {
	ASSERT (mapping != NULL) ;

	// determine record size
	uint8_t num_entries  = raxSize (mapping) ;
	size_t  types_padded = (num_entries + 7) & ~7 ;  // 8 bytes alignment
	size_t  n = sizeof (_Record) + types_padded + num_entries * sizeof (Entry) ;

	Record r = rm_calloc (1, n) ;

	r->mapping     = mapping ;
	r->num_entries = num_entries ;

	return r ;
}

// returns the number of entries held by record
uint Record_length
(
	const Record r
) {
	ASSERT (r != NULL) ;
	return r->num_entries ;
}

// get entry type
RecordEntryType Record_GetType
(
	const Record r,  // record
	uint idx         // entry index
) {
	ASSERT (Record_length (r) > idx) ;
	return Record_GetTypes (r)[idx] ;
}

bool Record_ContainsEntry
(
	const Record r,
	uint8_t idx
) {
	ASSERT (idx < Record_length (r)) ;
	return (Record_GetTypes (r)[idx] != REC_TYPE_UNKNOWN) ;
}

// retrieve the offset into the Record of the given alias
uint8_t Record_GetEntryIdx (
	Record r,
	const char *alias,
	size_t len
) {
	ASSERT (r     != NULL) ;
	ASSERT (len   >  0) ;
	ASSERT (alias != NULL) ;

	void *idx = raxFind (r->mapping, (unsigned char *)alias, len) ;
	return idx != raxNotFound ? (intptr_t)idx : INVALID_INDEX ;
}

// clone entries in `r` to `clone`
void Record_Clone
(
	const restrict Record r,  // src record
	restrict Record clone     // clone record
) {
	// r and clone share the same record mapping
	if (likely (r->owner == clone->owner)) {
		ASSERT (Record_length (r) <= Record_length (clone)) ;

		size_t required_data_arr_size = Record_ByteSize (r) - sizeof (_Record) ;
		memcpy (clone->data, r->data, required_data_arr_size) ;

		// foreach scalar entry in cloned record, make sure it is not freed
		// it is the original record owner responsibility to free the record
		// and its internal scalar as a result
	} else {
		// TODO: improve by introducing a recordbatch clone function
		// which takes a mapping array
		//
		// r and clone don't share the same mappings
		// scan through each entry within r
		// locate coresponding entry in clone
		// if such exists shallow clone it

		//----------------------------------------------------------------------
		// scan the shorter record
		//----------------------------------------------------------------------

		raxIterator it ;

		if (r->num_entries < clone->num_entries) {
			//------------------------------------------------------------------
			// scanning r
			//------------------------------------------------------------------

			raxStart (&it, r->mapping) ;
			raxSeek  (&it, "^", NULL, 0) ;

			while (raxNext(&it)) {
				uint8_t r_idx = (intptr_t)it.data ;

				// entry not set
				if (Record_GetType (r, r_idx) == REC_TYPE_UNKNOWN) {
					continue ;
				}

				// see if clone tracks entry
				uint8_t clone_idx =
					Record_GetEntryIdx (clone, (const char*)it.key, it.key_len) ;

				// clone does not track entry
				if (clone_idx == INVALID_INDEX) {
					continue ;
				}

				Record_Add (clone, clone_idx, Record_Get (r, r_idx)) ;
			}
		} else {
			//------------------------------------------------------------------
			// scanning clone
			//------------------------------------------------------------------
			raxStart (&it, clone->mapping) ;
			raxSeek  (&it, "^", NULL, 0) ;

			while (raxNext(&it)) {
				uint8_t clone_idx = (intptr_t)it.data ;

				// see if r tracks entry
				uint8_t r_idx =
					Record_GetEntryIdx (r, (const char*)it.key, it.key_len) ;

				// r does not track entry
				if (r_idx == INVALID_INDEX) {
					continue ;
				}

				// entry not set
				if (Record_GetType (r, r_idx) == REC_TYPE_UNKNOWN) {
					continue ;
				}

				Record_Add (clone, clone_idx, Record_Get (r, r_idx)) ;
			}
		}

		raxStop (&it) ;
	}

	// TODO: i wish we wouldn't have to perform this loop
	// as it is a major performance hit
	// with the introduction of a garbage collection this should be removed
	Entry           *entries = Record_GetEntries (clone) ;
	RecordEntryType *types   = Record_GetTypes   (clone) ;

	for (int i = 0 ; i < clone->num_entries ; i++) {
		if (types[i] == REC_TYPE_SCALAR) {
			SIValue_MakeVolatile (&entries[i].value.s) ;
		}
	}
}

// merge src record into dest
void Record_Merge
(
	restrict Record dest,      // dest record
	const restrict Record src  // src record
) {
	ASSERT (src->owner == dest->owner) ;

	uint8_t len = src->num_entries ;

	RecordEntryType *src_types    = Record_GetTypes   (src)  ;
	Entry           *src_entries  = Record_GetEntries (src)  ;
	RecordEntryType *dest_types   = Record_GetTypes   (dest) ;
	Entry           *dest_entries = Record_GetEntries (dest) ;

	for (uint i = 0 ; i < len; i++) {
		RecordEntryType src_type  = src_types[i] ;
		if (src_type == REC_TYPE_UNKNOWN) {
			// skip unknown entry
			continue ;
		}

		RecordEntryType dest_type = dest_types[i] ;
		if (dest_type == REC_TYPE_UNKNOWN) {
			// copy entry from src to dest
			dest_types[i]    = src_types[i] ;
			dest_entries[i] = src_entries[i] ;

			// protect heap allocated values
			if (src_type == REC_TYPE_SCALAR) {
				if (SI_ALLOCATION (&(src_entries[i].value.s)) == M_SELF) {
					SIValue_MakeVolatile (&(src_entries[i].value.s));
				} else {
					SIValue_Persist (&(dest_entries[i].value.s));
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
	ASSERT (src->owner == dest->owner) ;

	uint8_t len = src->num_entries ;
	RecordEntryType *src_types    = Record_GetTypes   (src)  ;
	Entry           *src_entries  = Record_GetEntries (src)  ;
	RecordEntryType *dest_types   = Record_GetTypes   (dest) ;
	Entry           *dest_entries = Record_GetEntries (dest) ;

	for (uint i = 0; i < len; i++) {
		// skip early if there's nothing to copy
		if (src_types[i] == REC_TYPE_UNKNOWN) {
			continue ;
		}

		if (dest_types[i] == REC_TYPE_UNKNOWN) {
			// copy the entry
			dest_types[i]   = src_types[i] ;
			dest_entries[i] = src_entries[i] ;

			if (dest_types[i] == REC_TYPE_SCALAR) {
				dest_entries[i].value.s =
					SI_CloneValue (dest_entries[i].value.s);
			}
		}
	}
}

// get a node from record at position idx
Node *Record_GetNode
(
	const Record r,  // record
	uint idx         // entry index
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	RecordEntryType type = Record_GetTypes (r)[idx] ;
	Entry *entries = Record_GetEntries (r) ;

	switch (type) {
		case REC_TYPE_NODE:
			return &(entries[idx].value.n) ;

		case REC_TYPE_UNKNOWN:
			return NULL ;

		case REC_TYPE_SCALAR:
			// Null scalar values are expected here; otherwise fall through
			if (SIValue_IsNull (entries[idx].value.s)) {
				return NULL ;
			}

		default:
			ErrorCtx_RaiseRuntimeException (
					"encountered unexpected type in Record; expected Node") ;
			return NULL ;
	}
}

// get an edge from record at position idx
Edge *Record_GetEdge
(
	const Record r,  // record
	uint idx         // entry index
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	RecordEntryType type = Record_GetTypes (r)[idx] ;
	Entry *entries = Record_GetEntries (r) ;

	switch (type) {
		case REC_TYPE_EDGE:
			return &(entries[idx].value.e) ;

		case REC_TYPE_UNKNOWN:
			return NULL ;

		case REC_TYPE_SCALAR:
			// Null scalar values are expected here; otherwise fall through.
			if (SIValue_IsNull(entries[idx].value.s)) {
				return NULL ;
			}

		default:
			ErrorCtx_RaiseRuntimeException (
					"encountered unexpected type in Record; expected Edge") ;
			return NULL ;
	}
}

// get an SIValue containing the entity at position idx
SIValue Record_Get
(
	Record r,  // record
	uint idx   // entry index
) {
	ASSERT (Record_length (r) > idx) ;

	RecordEntryType type = Record_GetTypes (r)[idx] ;

	switch (type) {
		case REC_TYPE_NODE :
			return SI_Node (Record_GetNode (r, idx)) ;

		case REC_TYPE_EDGE :
			return SI_Edge (Record_GetEdge(r, idx)) ;

		case REC_TYPE_SCALAR :
			return Record_GetEntries (r)[idx].value.s ;

		case REC_TYPE_UNKNOWN :
			return SI_NullVal ();

		default :
			ASSERT (false) ;
			return SI_NullVal () ;
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

	RecordEntryType *types = Record_GetTypes (r) ;
	RecordEntryType type = types[idx] ;

	if (type == REC_TYPE_UNKNOWN) {
		types[idx] = REC_TYPE_NODE ;
	}

	if (unlikely (types[idx] != REC_TYPE_NODE)) {
		ErrorCtx_RaiseRuntimeException (
				"encountered unexpected record type when trying to retrieve node") ;
		return NULL ;
	}

	Entry *e = &Record_GetEntries (r)[idx] ;
	return &(e->value.n) ;
}

// remove item at position idx
void Record_Remove
(
	Record r,  // record
	uint idx   // entry index
) {
	Record_GetTypes (r)[idx] = REC_TYPE_UNKNOWN ;
}

// get a graph entity from record at position idx
GraphEntity *Record_GetGraphEntity
(
	const Record r,  // record
	uint idx         // entry index
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	RecordEntryType t = Record_GetTypes (r)[idx] ;

	switch (t) {
		case REC_TYPE_NODE:
			return (GraphEntity *)Record_GetNode (r, idx) ;

		case REC_TYPE_EDGE:
			return (GraphEntity *)Record_GetEdge (r, idx) ;

		default:
			ErrorCtx_RaiseRuntimeException (
					"encountered unexpected type when trying to retrieve graph entity") ;
	}

	return NULL ;
}

// add a scalar, node, or edge to the record, depending on the SIValue type
void Record_Add
(
	Record r,  // record
	uint idx,  // entry index
	SIValue v  // value
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	switch (SI_TYPE (v)) {
		case T_NODE:
			Record_AddNode (r, idx, *(Node *)v.ptrval) ;
			break ;

		case T_EDGE:
			Record_AddEdge (r, idx, *(Edge *)v.ptrval) ;
			break ;

		default:
			Record_AddScalar (r, idx, v) ;
			break ;
	}
}

// add a scalar to record at position idx and return a reference to it
SIValue *Record_AddScalar
(
	Record r,  // record
	uint idx,  // entry index
	SIValue v  // value
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	Entry *entries = Record_GetEntries (r) ;
	Record_GetTypes (r)[idx] = REC_TYPE_SCALAR ;

	entries[idx].value.s = v ;
	return &(entries[idx].value.s) ;
}

// add a node to record at position idx and return a reference to it
Node *Record_AddNode
(
	Record r,  // record
	uint idx,  // entry index
	Node node  // node
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	Entry *entries = Record_GetEntries (r) ;
	entries[idx].value.n = node ;

	Record_GetTypes (r)[idx] = REC_TYPE_NODE ;

	return &(entries[idx].value.n) ;
}

// add an edge to record at position idx and return a reference to it
Edge *Record_AddEdge
(
	Record r,  // record
	uint idx,  // entry index
	Edge edge  // edge
) {
	ASSERT (r != NULL) ;
	ASSERT (idx < Record_length (r)) ;

	Entry *entries = Record_GetEntries (r) ;
	entries[idx].value.e = edge ;

	Record_GetTypes (r)[idx] = REC_TYPE_EDGE ;

	return &(entries[idx].value.e) ;
}

// string representation of record
size_t Record_ToString
(
	const Record r,  // record
	char **buf,      // buffer
	size_t *buf_cap  // buffer capacity
) {
	ASSERT (r       != NULL) ;
	ASSERT (buf     != NULL) ;
	ASSERT (buf_cap != NULL) ;

	uint rLen = Record_length (r) ;
	SIValue values[rLen] ;

	for (int i = 0; i < rLen; i++) {
		if (Record_GetType (r, i) == REC_TYPE_UNKNOWN) {
			values[i] = SI_ConstStringVal ("UNKNOWN") ;
		} else {
			values[i] = Record_Get (r, i) ;
		}
	}

	size_t required_len = SIValue_StringJoinLen (values, rLen, ",") ;

	if (*buf_cap < required_len) {
		*buf = rm_realloc (*buf, sizeof (char) * required_len) ;
		*buf_cap = required_len ;
	}

	size_t bytesWritten = 0 ;
	SIValue_StringJoin (values, rLen, ",", buf, buf_cap, &bytesWritten) ;
	return bytesWritten ;
}

// retrieves mapping associated with record
inline rax *Record_GetMappings
(
	const Record r  // record
) {
	ASSERT (r != NULL) ;
	return r->mapping ;
}

// remove and free entry at position idx
inline void Record_FreeEntry
(
	Record r,  // record
	int idx    // entry index
) {
	RecordEntryType *types = Record_GetTypes (r) ;


	if (types[idx] == REC_TYPE_SCALAR) {
		SIValue_Free (Record_GetEntries (r)[idx].value.s) ;
	}

	types[idx] = REC_TYPE_UNKNOWN ;
}

// free record entries
void Record_FreeEntries
(
	Record r  // record
) {
	ASSERT (r != NULL) ;

	uint n = r->num_entries ;
	Entry *entries = Record_GetEntries (r) ;
	RecordEntryType *types = Record_GetTypes (r) ;

	uint32_t i = 0 ;
	const uint32_t scalar_mask = 0x01010101U ;

    // process in batches of 4
	for (; i + 4 <= n; i+= 4) {
		// load 4 bytes into one 32-bit register
		uint32_t chunk = *(uint32_t*)(types + i) ;

		// if no scalars are present in these 4 entries, skip them entirely
		// this effectively skips Nodes, Edges, and Unknowns in one check
		if (!(chunk & scalar_mask)) {
			continue ;
		}

		// at least one scalar exists in this batch; find and free it
		for (int j = 0; j < 4; j++) {
			if (types[i + j] & REC_TYPE_SCALAR) {
				SIValue_Free (entries[i + j].value.s) ;
			}
		}
	}

	// process remaining tail (0-3 entries)
    for (; i < n; i++) {
        if (types[i] & REC_TYPE_SCALAR) {
            SIValue_Free (entries[i].value.s) ;
        }
    }

	// bulk reset types to UNKNOWN
    memset (types, REC_TYPE_UNKNOWN, n) ;
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
	ASSERT (r != NULL) ;
	ASSERT (r->ref_count == 0) ;

	Record_FreeEntries (r) ;
	rm_free (r) ;
}

