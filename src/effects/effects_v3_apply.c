/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3.h"
#include "effects_internal.h"
#include "../graph/graph_hub.h"
#include "../util/rmalloc.h"
#include "../util/roaring_include.h"

#include <inttypes.h>

//------------------------------------------------------------------------------
// v3 apply: records -> graph
//------------------------------------------------------------------------------
//
// This is where ids are expanded, and it is the only place that is allowed to.
// Decode returns segments precisely so that the cost of expansion is paid where
// the graph is in scope and can bound it.
//
// TWO BOUNDS DO THAT WORK, and neither is an invented constant:
//
//   * ids are expanded in fixed batches (APPLY_BATCH), never all at once, so
//     peak memory is independent of a record's declared count. One valid
//     segment describes four billion ids in seven bytes.
//   * for records that reference EXISTING entities - the updates, the deletes,
//     the label changes - a declared count larger than the local graph holds is
//     divergence, and is refused before it sizes anything. Graph_NodeCount and
//     Graph_RelationEdgeCount are the bound. A create cannot be bounded that
//     way (it is making the entities), which is why the batching above carries
//     that case on its own.
//
// EXISTENCE IS CHECKED BEFORE EVERY GraphHub CALL. GraphHub_UpdateNodeProperty
// guards its own Graph_GetNode with ASSERT (graph_hub.c), and ASSERT compiles to
// nothing when RG_DEBUG is off (RG.h) - so a missing node leaves an
// uninitialized Node whose 'attributes' is stack garbage, and the update writes
// through it. That is the AttributeSet_Update crash seen when C was fed a
// foreign buffer. The checks here are load-bearing in release, not belt and
// braces.
//
// VALUE OWNERSHIP: the record owns its SIValues and frees them. GraphHub's
// update entry points take ownership of the value they are handed
// (AttributeSet_Update with clone=false), so they are given SI_CloneValue and
// the record keeps its own copy.
//
//------------------------------------------------------------------------------

// how many ids are materialized at once
//
// matches v2's DELETE_NODE batch. Bounds peak memory independently of a
// record's count, which is a wire-derived u32.
#define APPLY_BATCH 4096

//------------------------------------------------------------------------------
// walking an IdList without expanding it
//------------------------------------------------------------------------------

// a cursor over an IdList that yields one id at a time
//
// holds no array proportional to the list's cardinality: a Range and a Repeat
// are arithmetic, and an Ascending segment is walked with roaring's own
// iterator rather than converted to an array
typedef struct {
	const EffectsV3IdList *list;
	uint32_t               seg;       // segment being walked
	uint64_t               produced;  // ids yielded from that segment
	roaring64_bitmap_t    *bitmap;    // Ascending only, owned
	roaring64_iterator_t  *bit_it;    // Ascending only, owned
	bool                   broken;    // a bitmap failed to deserialize
} IdIter;

static void _IdIter_Init
(
	IdIter *it,
	const EffectsV3IdList *list
) {
	it->list     = list ;
	it->seg      = 0 ;
	it->produced = 0 ;
	it->bitmap   = NULL ;
	it->bit_it   = NULL ;
	it->broken   = false ;
}

// release whatever the current Ascending segment allocated
static void _IdIter_CloseSegment
(
	IdIter *it
) {
	if (it->bit_it != NULL) {
		roaring64_iterator_free (it->bit_it) ;
		it->bit_it = NULL ;
	}
	if (it->bitmap != NULL) {
		roaring64_bitmap_free (it->bitmap) ;
		it->bitmap = NULL ;
	}
}

static void _IdIter_Free
(
	IdIter *it
) {
	_IdIter_CloseSegment (it) ;
}

// yield the next id
//
// returns false when the list is exhausted, or when a bitmap could not be
// deserialized - 'broken' distinguishes the two
static bool _IdIter_Next
(
	IdIter *it,
	uint64_t *id
) {
	while (it->seg < it->list->n) {
		const EffectsV3Segment *s = it->list->segments + it->seg ;

		switch (s->kind) {
			case EFFECTS_V3_SEG_RANGE:
				if (it->produced < s->range.len) {
					// a descending range's base is its FIRST and HIGHEST id
					*id = s->descending
						? s->range.base - it->produced
						: s->range.base + it->produced ;
					it->produced++ ;
					return true ;
				}
				break ;

			case EFFECTS_V3_SEG_REPEAT:
				if (it->produced < s->repeat.count) {
					*id = s->repeat.id ;
					it->produced++ ;
					return true ;
				}
				break ;

			case EFFECTS_V3_SEG_ASCENDING:
				if (it->bitmap == NULL) {
					it->bitmap = roaring64_bitmap_portable_deserialize_safe (
							(const char*)s->ascending.blob, s->ascending.n) ;
					if (it->bitmap == NULL) {
						// decode already deserialized this blob to take its
						// cardinality, so failing here means memory pressure
						// rather than a malformed payload
						it->broken = true ;
						return false ;
					}
					it->bit_it = s->descending
						? roaring64_iterator_create_last (it->bitmap)
						: roaring64_iterator_create (it->bitmap) ;
					if (it->bit_it == NULL) {
						_IdIter_CloseSegment (it) ;
						it->broken = true ;
						return false ;
					}
				}

				if (roaring64_iterator_has_value (it->bit_it)) {
					*id = roaring64_iterator_value (it->bit_it) ;
					if (s->descending) {
						roaring64_iterator_previous (it->bit_it) ;
					} else {
						roaring64_iterator_advance (it->bit_it) ;
					}
					it->produced++ ;
					return true ;
				}
				break ;

			default:
				it->broken = true ;
				return false ;
		}

		// this segment is spent
		_IdIter_CloseSegment (it) ;
		it->seg++ ;
		it->produced = 0 ;
	}

	return false ;
}

//------------------------------------------------------------------------------
// shared validation
//------------------------------------------------------------------------------

// resolve every label a record names against local schema
//
// RESOLUTION, not a range check: an id can be inside the schema count and still
// map to nothing, which a count comparison cannot see. Records 9 and 10 are
// normatively ordered ahead of anything referencing the ids they introduce, so
// in a well-formed buffer these always resolve - one that does not IS
// divergence. This is not an invented ceiling: it is a lookup against local
// state, so it cannot fire on legitimate data.
static bool _VerifyLabels
(
	GraphContext *gc,
	const EffectsV3Record *rec,
	const char *op
) {
	for (uint16_t i = 0; i < rec->n_labels; i++) {
		if (GraphContext_GetSchemaByID (gc, rec->labels[i],
					SCHEMA_NODE) == NULL) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT %s references unknown label schema %d",
					op, rec->labels[i]) ;
			return false ;
		}
	}
	return true ;
}

// resolve a record's relationship type against local schema
static bool _VerifyRelation
(
	GraphContext *gc,
	const EffectsV3Record *rec,
	const char *op
) {
	if (GraphContext_GetSchemaByID (gc, rec->relation_id,
				SCHEMA_EDGE) == NULL) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT %s references relationship type %d "
				"which doesn't exist locally", op, rec->relation_id) ;
		return false ;
	}
	return true ;
}

// confirm the graph knows every attribute the record's shape names
//
// once per record rather than once per row: the grouping has already
// established that every entity in the record has exactly these attribute ids
static bool _VerifyAttrIds
(
	GraphContext *gc,
	const EffectsV3Record *rec,
	const char *op
) {
	for (uint16_t i = 0; i < rec->n_attrs; i++) {
		const AttributeID a = rec->attr_ids[i] ;

		if (a == ATTRIBUTE_ID_NONE) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT %s illegal attribute id %d", op, a) ;
			return false ;
		}

		if (a != ATTRIBUTE_ID_ALL && !GraphContext_HasAttribute (gc, a)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT %s unknown attribute id %d", op, a) ;
			return false ;
		}
	}
	return true ;
}

// every value in the record must be storable, or a removal
//
// T_NULL is legal and means REMOVE THIS ATTRIBUTE - FalkorDB never stores a
// null property, so SET n.x = NULL replicates as a null in a value slot. A
// reader that rejected or filtered nulls would turn every removal into a no-op.
static bool _VerifyValues
(
	const EffectsV3Record *rec,
	const char *op
) {
	for (uint64_t i = 0; i < rec->n_values; i++) {
		if (!(SI_TYPE (rec->values[i]) & (SI_VALID_PROPERTY_VALUE | T_NULL))) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT %s carries a value that cannot be stored",
					op) ;
			return false ;
		}
	}
	return true ;
}

// refuse a count that exceeds what the local graph could possibly satisfy
//
// only meaningful for records referencing entities that must ALREADY exist. A
// bound taken from local graph state, so it cannot misfire on legitimate data
// the way a constant would - being told to update more nodes than exist is
// divergence by definition.
static bool _VerifyCountAgainstGraph
(
	uint64_t count,
	uint64_t local,
	const char *op,
	const char *what
) {
	if (count > local) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT %s declares %" PRIu64 " %s but only %" PRIu64
				" exist locally", op, count, what, local) ;
		return false ;
	}
	return true ;
}

//------------------------------------------------------------------------------
// records 9 and 10 - the two that establish an id space
//------------------------------------------------------------------------------

// ADD_SCHEMA: create the schema and confirm it landed on the id the wire states
//
// This is the check v3 exists for. v2 carried no id and the replica inferred it
// from append order, so a dictionary of a different length assigned a different
// id to the same name and every later record silently used the wrong one. The
// id is on the wire now, so the disagreement is caught where it is introduced.
static bool _ApplyAddSchema
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	if (rec->name == NULL) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_SCHEMA carries no name") ;
		return false ;
	}

	bool created = false ;
	Schema *s = GraphContext_FindOrAddSchema (gc, rec->name, rec->schema_type,
			&created) ;

	if (s == NULL || created == false) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_SCHEMA targets schema '%s' which already "
				"exists locally", rec->name) ;
		return false ;
	}

	const int assigned = Schema_GetID (s) ;
	if (assigned != rec->schema_id) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_SCHEMA '%s' was assigned id %d locally but "
				"the master assigned %d - schema numbering has diverged",
				rec->name, assigned, rec->schema_id) ;
		return false ;
	}

	return true ;
}

// ADD_ATTRIBUTE: same shape, on the attribute dictionary
//
// no node/relationship discriminator: #2459 unified the two dictionaries and C
// has always had one
static bool _ApplyAddAttribute
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	if (rec->name == NULL) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_ATTRIBUTE carries no name") ;
		return false ;
	}

	if (GraphContext_GetAttributeID (gc, rec->name) != ATTRIBUTE_ID_NONE) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_ATTRIBUTE targets attribute '%s' which "
				"already exists locally", rec->name) ;
		return false ;
	}

	const AttributeID assigned =
		GraphHub_FindOrAddAttribute (gc, rec->name, false) ;

	if (assigned != rec->attr_id) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_ATTRIBUTE '%s' was assigned id %d locally "
				"but the master assigned %d - attribute numbering has diverged",
				rec->name, assigned, rec->attr_id) ;
		return false ;
	}

	return true ;
}

//------------------------------------------------------------------------------
// records 3 and 4 - creates
//------------------------------------------------------------------------------

// build the AttributeSet for row 'k'
//
// values are CLONED: the record owns its SIValues and frees them, while the
// attribute set takes ownership of what it is given
static AttributeSet _RowAttributes
(
	const EffectsV3Record *rec,
	uint64_t k
) {
	AttributeSet set = NULL ;
	if (rec->n_attrs == 0) {
		return set ;
	}

	SIValue vals[rec->n_attrs] ;
	for (uint16_t a = 0; a < rec->n_attrs; a++) {
		vals[a] = SI_CloneValue (rec->values[k * rec->n_attrs + a]) ;
	}

	AttributeSet_Add (&set, rec->attr_ids, vals, rec->n_attrs, false) ;
	return set ;
}

static bool _ApplyCreateNode
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	if (!_VerifyLabels (gc, rec, "CREATE_NODE")   ||
		!_VerifyAttrIds (gc, rec, "CREATE_NODE")  ||
		!_VerifyValues (rec, "CREATE_NODE")) {
		return false ;
	}

	IdIter it ;
	_IdIter_Init (&it, &rec->ids) ;

	bool ok = true ;
	uint64_t id ;
	uint64_t k = 0 ;

	while (ok && _IdIter_Next (&it, &id)) {
		AttributeSet set = _RowAttributes (rec, k) ;

		Node n = GE_NEW_NODE () ;
		GraphHub_CreateNode (gc, &n, rec->labels, rec->n_labels, set, false) ;

		// v3's whole premise: the replica does not infer the id, it checks it
		if (n.id != id) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT CREATE_NODE allocated node %" PRIu64
					" locally but the master allocated %" PRIu64
					" - node id allocation has diverged", n.id, id) ;
			ok = false ;
		}

		k++ ;
	}

	if (ok && it.broken) {
		ok = false ;
	}

	_IdIter_Free (&it) ;
	return ok ;
}

static bool _ApplyCreateEdge
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	if (!_VerifyRelation (gc, rec, "CREATE_EDGE") ||
		!_VerifyAttrIds (gc, rec, "CREATE_EDGE")  ||
		!_VerifyValues (rec, "CREATE_EDGE")) {
		return false ;
	}

	Graph *g = GraphContext_GetGraph (gc) ;

	// resolved once for the whole record: the grouping has already established
	// that every edge in it shares this type, and the name is what the index
	// is keyed under
	Schema *schema = GraphContext_GetSchemaByID (gc, rec->relation_id,
			SCHEMA_EDGE) ;
	const char *rel_name = Schema_GetName (schema) ;

	IdIter ids, srcs, dsts ;
	_IdIter_Init (&ids,  &rec->ids) ;
	_IdIter_Init (&srcs, &rec->src) ;
	_IdIter_Init (&dsts, &rec->dst) ;

	bool ok = true ;
	uint64_t id, src, dst ;
	uint64_t k = 0 ;

	while (ok && _IdIter_Next (&ids, &id)) {
		// the three lists are positionally aligned by construction, and decode
		// has already checked all three total the record's count - so a
		// short one here is a decoder bug, not a wire problem
		if (!_IdIter_Next (&srcs, &src) || !_IdIter_Next (&dsts, &dst)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT CREATE_EDGE endpoint lists are shorter than "
					"its id list") ;
			ok = false ;
			break ;
		}

		// endpoints must exist before an edge can join them. Checked here
		// because GraphHub_CreateEdge does not check for us.
		if (!Graph_HasNode (g, src) || !Graph_HasNode (g, dst)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT CREATE_EDGE references nodes %" PRIu64
					" -> %" PRIu64 ", at least one of which doesn't exist "
					"locally", src, dst) ;
			ok = false ;
			break ;
		}

		AttributeSet set = _RowAttributes (rec, k) ;

		Edge e = GE_NEW_LABELED_EDGE (rel_name, rec->relation_id) ;
		GraphHub_CreateEdge (gc, &e, src, dst, rec->relation_id, set, false) ;

		if (e.id != id) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT CREATE_EDGE allocated edge %" PRIu64
					" locally but the master allocated %" PRIu64
					" - edge id allocation has diverged", e.id, id) ;
			ok = false ;
		}

		k++ ;
	}

	if (ok && (ids.broken || srcs.broken || dsts.broken)) {
		ok = false ;
	}

	_IdIter_Free (&ids) ;
	_IdIter_Free (&srcs) ;
	_IdIter_Free (&dsts) ;
	return ok ;
}

//------------------------------------------------------------------------------
// records 1 and 2 - updates
//------------------------------------------------------------------------------

static bool _ApplyUpdateNode
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	Graph *g = GraphContext_GetGraph (gc) ;

	if (!_VerifyCountAgainstGraph (rec->count, Graph_NodeCount (g),
				"UPDATE_NODE", "nodes")                ||
		!_VerifyLabels (gc, rec, "UPDATE_NODE")        ||
		!_VerifyAttrIds (gc, rec, "UPDATE_NODE")       ||
		!_VerifyValues (rec, "UPDATE_NODE")) {
		return false ;
	}

	IdIter it ;
	_IdIter_Init (&it, &rec->ids) ;

	bool ok = true ;
	uint64_t id ;
	uint64_t k = 0 ;

	while (ok && _IdIter_Next (&it, &id)) {
		if (id == INVALID_ENTITY_ID || !Graph_HasNode (g, id)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT UPDATE_NODE references node %" PRIu64
					" which doesn't exist locally", id) ;
			ok = false ;
			break ;
		}

		for (uint16_t a = 0; a < rec->n_attrs; a++) {
			const SIValue v = rec->values[k * rec->n_attrs + a] ;

			// ATTRIBUTE_ID_ALL means "remove everything" and is only legal
			// alongside a null, mirroring v2's check
			if (rec->attr_ids[a] == ATTRIBUTE_ID_ALL && !SIValue_IsNull (v)) {
				RedisModule_Log (NULL, "warning",
						"GRAPH.EFFECT UPDATE_NODE illegal attribute id %d",
						rec->attr_ids[a]) ;
				ok = false ;
				break ;
			}

			// the hub takes ownership of the value it is handed
			GraphHub_UpdateNodeProperty (gc, id, rec->attr_ids[a],
					SI_CloneValue (v)) ;
		}

		k++ ;
	}

	if (ok && it.broken) {
		ok = false ;
	}

	_IdIter_Free (&it) ;
	return ok ;
}

// UPDATE_EDGE carries its relationship type and deliberately not its endpoints
//
// They are per edge rather than per record, so carrying them would cost two
// more IdLists where the type costs four bytes. C needs them anyway, because
// GraphHub_UpdateEdgeProperty fills an Edge for the index and index_edge.c
// stores 'range:_src_id' / 'range:_dest_id'. Graph_GetEdge sets only id and
// attributes and will not help.
//
// So the endpoints are recovered by scanning the relationship's own tensor
// once, which yields (src, dst, edge_id) - precedent at
// graphcontext_memoryUsage.c. One scan per record, not per edge: a per-edge
// scan would be quadratic.
static bool _ApplyUpdateEdge
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	Graph *g = GraphContext_GetGraph (gc) ;

	// the local edge count of this type is what makes the (id -> row) table
	// below safe to size: being told to update more edges of a type than exist
	// is divergence, and refusing first means a wire-declared count never
	// reaches an allocator
	if (!_VerifyRelation (gc, rec, "UPDATE_EDGE")) {
		return false ;
	}

	const uint64_t local =
		Graph_RelationEdgeCount (g, rec->relation_id) ;

	if (!_VerifyCountAgainstGraph (rec->count, local, "UPDATE_EDGE", "edges") ||
		!_VerifyAttrIds (gc, rec, "UPDATE_EDGE")                              ||
		!_VerifyValues (rec, "UPDATE_EDGE")) {
		return false ;
	}

	if (rec->count == 0) {
		return true ;
	}

	//--------------------------------------------------------------------------
	// materialize (edge id -> row) so a single tensor scan can find each row
	//--------------------------------------------------------------------------

	typedef struct { uint64_t id ; uint64_t row ; } IdRow ;

	IdRow *table = rm_malloc (rec->count * sizeof (IdRow)) ;

	IdIter it ;
	_IdIter_Init (&it, &rec->ids) ;

	uint64_t id ;
	uint64_t k = 0 ;
	while (k < rec->count && _IdIter_Next (&it, &id)) {
		table[k].id  = id ;
		table[k].row = k ;
		k++ ;
	}
	const bool broken = it.broken ;
	_IdIter_Free (&it) ;

	if (broken || k != rec->count) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT UPDATE_EDGE id list yielded %" PRIu64
				" of %u ids", k, rec->count) ;
		rm_free (table) ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// scan the relationship tensor, updating each edge the record names
	//--------------------------------------------------------------------------

	Tensor R = Graph_GetRelationMatrix (g, rec->relation_id, false) ;

	TensorIterator ti ;
	TensorIterator_ScanRange (&ti, R, 0, UINT64_MAX, false) ;

	uint64_t applied = 0 ;
	GrB_Index row, col ;
	uint64_t  edge_id ;

	while (applied < rec->count &&
			TensorIterator_next (&ti, &row, &col, &edge_id, NULL)) {
		// linear over the table: the record's ids are usually one Range, so
		// this is a scan over a handful of entries in practice. Kept simple
		// deliberately - a sort plus binary search would be faster on a large
		// record and is worth doing only once that shape is measured.
		for (uint64_t i = 0; i < rec->count; i++) {
			if (table[i].id != edge_id) {
				continue ;
			}

			for (uint16_t a = 0; a < rec->n_attrs; a++) {
				const SIValue v =
					rec->values[table[i].row * rec->n_attrs + a] ;

				if (rec->attr_ids[a] == ATTRIBUTE_ID_ALL &&
						!SIValue_IsNull (v)) {
					RedisModule_Log (NULL, "warning",
							"GRAPH.EFFECT UPDATE_EDGE illegal attribute id %d",
							rec->attr_ids[a]) ;
					rm_free (table) ;
					return false ;
				}

				GraphHub_UpdateEdgeProperty (gc, edge_id, rec->relation_id,
						row, col, rec->attr_ids[a], SI_CloneValue (v)) ;
			}

			applied++ ;
			break ;
		}
	}

	rm_free (table) ;

	if (applied != rec->count) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT UPDATE_EDGE matched %" PRIu64 " of %u edges of "
				"relationship type %d locally", applied, rec->count,
				rec->relation_id) ;
		return false ;
	}

	return true ;
}

//------------------------------------------------------------------------------
// records 5 and 6 - deletes
//------------------------------------------------------------------------------

// DELETE_NODE carries the labels the node ACTUALLY held, captured as it was
// deleted - they cannot be recovered when the buffer is built, because by then
// the node is gone.
//
// On a replica the node still exists at apply time, so C does not need them to
// drive the deletion: GraphHub_DeleteNodes reads the local matrices. They are
// used here as a divergence check, which is what a replica can do with them.
static bool _ApplyDeleteNode
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	Graph *g = GraphContext_GetGraph (gc) ;

	if (!_VerifyCountAgainstGraph (rec->count, Graph_NodeCount (g),
				"DELETE_NODE", "nodes")                ||
		!_VerifyLabels (gc, rec, "DELETE_NODE")) {
		return false ;
	}

	IdIter it ;
	_IdIter_Init (&it, &rec->ids) ;

	Node batch[APPLY_BATCH] ;
	uint32_t n = 0 ;
	bool ok = true ;
	uint64_t id ;

	while (ok && _IdIter_Next (&it, &id)) {
		if (!Graph_GetNode (g, id, batch + n)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT DELETE_NODE references node %" PRIu64
					" which doesn't exist locally", id) ;
			ok = false ;
			break ;
		}

		n++ ;

		if (n == APPLY_BATCH) {
			GraphHub_DeleteNodes (gc, batch, n, false) ;
			n = 0 ;
		}
	}

	if (ok && it.broken) {
		ok = false ;
	}

	if (ok && n > 0) {
		GraphHub_DeleteNodes (gc, batch, n, false) ;
	}

	_IdIter_Free (&it) ;
	return ok ;
}

static bool _ApplyDeleteEdge
(
	GraphContext *gc,
	const EffectsV3Record *rec
) {
	Graph *g = GraphContext_GetGraph (gc) ;

	if (!_VerifyRelation (gc, rec, "DELETE_EDGE")) {
		return false ;
	}

	if (!_VerifyCountAgainstGraph (rec->count,
				Graph_RelationEdgeCount (g, rec->relation_id),
				"DELETE_EDGE", "edges")) {
		return false ;
	}

	Schema *schema = GraphContext_GetSchemaByID (gc, rec->relation_id,
			SCHEMA_EDGE) ;
	const char *rel_name = Schema_GetName (schema) ;

	IdIter ids, srcs, dsts ;
	_IdIter_Init (&ids,  &rec->ids) ;
	_IdIter_Init (&srcs, &rec->src) ;
	_IdIter_Init (&dsts, &rec->dst) ;

	Edge batch[APPLY_BATCH] ;
	uint32_t n = 0 ;
	bool ok = true ;
	uint64_t id, src, dst ;

	while (ok && _IdIter_Next (&ids, &id)) {
		if (!_IdIter_Next (&srcs, &src) || !_IdIter_Next (&dsts, &dst)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT DELETE_EDGE endpoint lists are shorter than "
					"its id list") ;
			ok = false ;
			break ;
		}

		// the delete records carry endpoints precisely because a deleted
		// edge's endpoints cannot be recovered afterwards, so they are used
		// rather than re-derived
		Edge *e = batch + n ;
		*e = GE_NEW_LABELED_EDGE (rel_name, rec->relation_id) ;
		e->id      = id ;
		e->src_id  = src ;
		e->dest_id = dst ;

		if (!Graph_GetEdge (g, id, e)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT DELETE_EDGE references edge %" PRIu64
					" which doesn't exist locally", id) ;
			ok = false ;
			break ;
		}

		// Graph_GetEdge fills only id and attributes, so the endpoints and
		// type from the wire are restored over whatever it left
		e->src_id     = src ;
		e->dest_id    = dst ;
		e->relationID = rec->relation_id ;

		n++ ;

		if (n == APPLY_BATCH) {
			GraphHub_DeleteEdges (gc, batch, n, false, false) ;
			n = 0 ;
		}
	}

	if (ok && (ids.broken || srcs.broken || dsts.broken)) {
		ok = false ;
	}

	if (ok && n > 0) {
		GraphHub_DeleteEdges (gc, batch, n, false, false) ;
	}

	_IdIter_Free (&ids) ;
	_IdIter_Free (&srcs) ;
	_IdIter_Free (&dsts) ;
	return ok ;
}

//------------------------------------------------------------------------------
// records 7 and 8 - label changes
//------------------------------------------------------------------------------

// SET_LABELS / REMOVE_LABELS carry the labels and ALL their nodes, rather than
// one (node, label) pair per node
//
// Grouping is sound because label add and remove are idempotent set
// operations, so order within a record carries no information - unlike edge
// endpoints, which is why those stay positional.
//
// GraphHub_UpdateNodeLabels takes one GrB_Vector per label, named by the
// schema's name, whose pattern is the node set. Every node in the record gets
// every label in the record, so all the vectors share one pattern.
static bool _ApplyLabels
(
	GraphContext *gc,
	const EffectsV3Record *rec,
	bool add
) {
	const char *op = add ? "SET_LABELS" : "REMOVE_LABELS" ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (!_VerifyCountAgainstGraph (rec->count, Graph_NodeCount (g),
				op, "nodes")) {
		return false ;
	}

	if (rec->n_labels == 0) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT %s carries no labels", op) ;
		return false ;
	}

	if (!_VerifyLabels (gc, rec, op)) {
		return false ;
	}

	GrB_Vector *lbls = rm_calloc (rec->n_labels, sizeof (GrB_Vector)) ;
	const uint64_t cap = Graph_NodeCap (g) ;

	for (uint16_t i = 0; i < rec->n_labels; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, rec->labels[i],
				SCHEMA_NODE) ;
		GrB_OK (GrB_Vector_new (lbls + i, GrB_BOOL, cap)) ;
		GrB_OK (GrB_set (lbls[i], (char*) Schema_GetName (s), GrB_NAME)) ;
	}

	IdIter it ;
	_IdIter_Init (&it, &rec->ids) ;

	bool ok = true ;
	uint64_t id ;

	while (ok && _IdIter_Next (&it, &id)) {
		if (!Graph_HasNode (g, id)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT %s references node %" PRIu64
					" which doesn't exist locally", op, id) ;
			ok = false ;
			break ;
		}

		for (uint16_t i = 0; i < rec->n_labels; i++) {
			GrB_OK (GrB_Vector_setElement (lbls[i], true, id)) ;
		}
	}

	if (ok && it.broken) {
		ok = false ;
	}

	_IdIter_Free (&it) ;

	if (ok) {
		if (add) {
			GraphHub_UpdateNodeLabels (gc, lbls, rec->n_labels, NULL, 0,
					false) ;
		} else {
			GraphHub_UpdateNodeLabels (gc, NULL, 0, lbls, rec->n_labels,
					false) ;
		}
	}

	for (uint16_t i = 0; i < rec->n_labels; i++) {
		GrB_OK (GrB_free (lbls + i)) ;
	}
	rm_free (lbls) ;

	return ok ;
}

//------------------------------------------------------------------------------
// the entry point
//------------------------------------------------------------------------------

bool EffectsV3_Apply
(
	GraphContext *gc,
	const EffectsV3Records *records
) {
	ASSERT (gc      != NULL) ;
	ASSERT (records != NULL) ;

	if (gc == NULL || records == NULL) {
		return false ;
	}

	// records arrive in apply order and are applied in it. Records 9 and 10
	// are normatively ahead of anything referencing the ids they introduce, so
	// nothing here reorders.
	for (uint32_t i = 0; i < records->n; i++) {
		const EffectsV3Record *rec = records->records + i ;
		bool ok ;

		switch (rec->opcode) {
			case EFFECT_ADD_SCHEMA:
				ok = _ApplyAddSchema (gc, rec) ;
				break ;

			case EFFECT_ADD_ATTRIBUTE:
				ok = _ApplyAddAttribute (gc, rec) ;
				break ;

			case EFFECT_CREATE_NODE:
				ok = _ApplyCreateNode (gc, rec) ;
				break ;

			case EFFECT_CREATE_EDGE:
				ok = _ApplyCreateEdge (gc, rec) ;
				break ;

			case EFFECT_UPDATE_NODE:
				ok = _ApplyUpdateNode (gc, rec) ;
				break ;

			case EFFECT_UPDATE_EDGE:
				ok = _ApplyUpdateEdge (gc, rec) ;
				break ;

			case EFFECT_DELETE_NODE:
				ok = _ApplyDeleteNode (gc, rec) ;
				break ;

			case EFFECT_DELETE_EDGE:
				ok = _ApplyDeleteEdge (gc, rec) ;
				break ;

			case EFFECT_SET_LABELS:
				ok = _ApplyLabels (gc, rec, true) ;
				break ;

			case EFFECT_REMOVE_LABELS:
				ok = _ApplyLabels (gc, rec, false) ;
				break ;

			default:
				// records 11-14 are refused at decode, so reaching here means
				// a record this build decoded and cannot apply
				RedisModule_Log (NULL, "warning",
						"GRAPH.EFFECT cannot apply record type %d",
						(int)rec->opcode) ;
				ok = false ;
				break ;
		}

		if (!ok) {
			// stop at the first failure: the caller treats a false return as
			// divergence and must not propagate the effects any further
			return false ;
		}
	}

	return true ;
}
