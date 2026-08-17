/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "../util/arr.h"
#include "effects_internal.h"
#include "../graph/graph_hub.h"

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

// read effect type from stream
static inline EffectType ReadEffectType
(
	FILE *stream  // effects stream
) {
	EffectType t = EFFECT_UNKNOWN;  // default to unknown effect type

	// read EffectType off of stream
	fread_assert(&t, sizeof(EffectType), stream);

	return t;
}

static AttributeSet ReadAttributeSet
(
	FILE *stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// read attribute count
	//--------------------------------------------------------------------------

	uint16_t attr_count;
	fread_assert(&attr_count, sizeof(attr_count), stream);
	if (attr_count == 0) {
		return NULL ;
	}

	//--------------------------------------------------------------------------
	// read attributes
	//--------------------------------------------------------------------------

	SIValue values[attr_count];
	AttributeID ids[attr_count];

	for(uint16_t i = 0; i < attr_count; i++) {
		// read attribute ID
		fread_assert(ids + i, sizeof(AttributeID), stream);
		
		// read attribute value
		values[i] = SIValue_FromBinary(stream);
	}

	AttributeSet attr_set = NULL;
	AttributeSet_Add (&attr_set, ids, values, attr_count, false) ;

	return attr_set;
}

// validate that an ID read from the effects stream refers to an existing
// schema entry, if it does not the graph is out of sync with the effects
// stream (e.g. a replicated query failed on this side), in which case we
// log the problem and exit instead of dereferencing an invalid matrix
static inline void ValidateSchemaID
(
	const Graph *g,     // graph to validate against
	int id,             // schema ID read from the effects stream
	SchemaType t        // schema type (label / relationship-type)
) {
	int count = (t == SCHEMA_EDGE)
		? Graph_RelationTypeCount (g)
		: Graph_LabelTypeCount (g) ;

	if (unlikely (id < 0 || id >= count)) {
		// graph/effects-stream out of sync
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT refers to a missing %s (id: %d), graph is out of sync with the effects stream",
				(t == SCHEMA_EDGE) ? "relationship-type" : "label", id) ;
		exit (1) ;
	}
}

static void ApplyCreateNode
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// read label count
	//--------------------------------------------------------------------------

	uint16_t lbl_count ;
	fread_assert (&lbl_count, sizeof (lbl_count), stream) ;

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	LabelID labels[lbl_count] ;
	Graph *g = GraphContext_GetGraph (gc) ;
	for (uint16_t i = 0; i < lbl_count; i++) {
		fread_assert (labels + i, sizeof (LabelID), stream) ;
		ValidateSchemaID (g, labels[i], SCHEMA_NODE) ;
	}

	//--------------------------------------------------------------------------
	// read attributes
	//--------------------------------------------------------------------------

	AttributeSet attr_set = ReadAttributeSet (stream) ;

	//--------------------------------------------------------------------------
	// create node
	//--------------------------------------------------------------------------

	Node n = GE_NEW_NODE () ;
	GraphHub_CreateNode (gc, &n, labels, lbl_count, attr_set, false) ;
}

static inline void FlushEdges
(
	GraphContext *gc,
	Edge **batch,
	AttributeSet *sets,
	RelationID r,
	int *i
) {
	ASSERT (i     != NULL) ;
	ASSERT (gc    != NULL) ;
	ASSERT (sets  != NULL) ;
	ASSERT (batch != NULL) ;

	if (*i > 0) {
		ASSERT (arr_len (batch) == *i) ;
		ASSERT (arr_len (batch) == arr_len (sets)) ;

		GraphHub_CreateEdges (gc, batch, r, sets, false) ;
		arr_clear (sets) ;
		arr_clear (batch) ;
		*i = 0 ;
	}
}

// apply "create edge" effects from a serialized stream
//
// the stream encodes a sequence of EFFECT_CREATE_EDGE operations
//  format per edge:
//    - uint16_t   rel_count (must be 1)
//    - RelationID relation
//    - NodeID     src node ID
//    - NodeID     dest node ID
//    - AttributeSet (id, value pairs)
//
// multiple edges of the same relation type are batched together
// for efficient insertion
static void ApplyCreateEdge
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	size_t l           // length of stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// relationship count
	// relationships
	// src node ID
	// dest node ID
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	int i = 0 ;                     // size of current batch
	enum { batch_size = 4096 } ;    // compile-time constant, avoids a VLA
	Edge edges[batch_size] ;        // edges

	Edge **batch = arr_new (Edge *, 1) ;  // batch, points to edges
	AttributeSet *sets = arr_new (AttributeSet, 1) ;  // attribute-sets

	RelationID r      = GRAPH_UNKNOWN_RELATION ;  // current edge relation id
	RelationID prev_r = GRAPH_UNKNOWN_RELATION ;  // last processed relation id

	Graph *g = GraphContext_GetGraph (gc) ;

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		uint16_t rel_count ;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _edge_desc;
	#pragma pack(pop)

	while (true) {
		//----------------------------------------------------------------------
		// read a single edge descriptor in one go
		//----------------------------------------------------------------------

		fread_assert(&_edge_desc, sizeof (_edge_desc), stream);
		ASSERT(_edge_desc.rel_count == 1);
		ValidateSchemaID (g, _edge_desc.r, SCHEMA_EDGE) ;

		if (prev_r == GRAPH_UNKNOWN_RELATION) {
			prev_r = _edge_desc.r ;
		}

		// check if relationship-type changed
		if (_edge_desc.r != prev_r) {
			FlushEdges (gc, batch, sets, prev_r, &i) ;
			prev_r = _edge_desc.r ;
		}

		//----------------------------------------------------------------------
		// read attributes
		//----------------------------------------------------------------------

		arr_append (sets, ReadAttributeSet(stream)) ;

		//----------------------------------------------------------------------
		// add edge to batch
		//----------------------------------------------------------------------

		r = _edge_desc.r ;

		Edge *e = edges + i ;
		Edge_SetSrcNodeID  (e, _edge_desc.src_id) ;
		Edge_SetDestNodeID (e, _edge_desc.dest_id) ;
		Edge_SetRelationID (e, _edge_desc.r) ;

		arr_append (batch, e) ;
		i++ ;

		// check if batch is full
		if (i == batch_size) {
			FlushEdges (gc, batch, sets, r, &i) ;
		}

		// have we reached the end of the stream ?
		if (ftell (stream) >= l) {
			break ;
		}

		// check if the next item in the stream is a EFFECT_CREATE_EDGE effect
		EffectType t = ReadEffectType (stream) ;
		if (t != EFFECT_CREATE_EDGE) {
			// go back sizeof (EffectType) bytes
			fseek (stream, -((long)sizeof (EffectType)), SEEK_CUR) ;
			break ;
		}
	}

	// flush last batch
	FlushEdges (gc, batch, sets, r, &i) ;

	// clean up
	arr_free (sets) ;
	arr_free (batch) ;
}

// returns false if the effect references graph state that doesn't exist
// locally (replica has diverged from the master)
static bool ApplyLabels
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	bool add           // add or remove labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// read node ID
	//--------------------------------------------------------------------------

	EntityID id ;
	fread_assert (&id, sizeof (id), stream) ;

	//--------------------------------------------------------------------------
	// get updated node
	//--------------------------------------------------------------------------

	Node  n ;

	Graph *g = GraphContext_GetGraph (gc) ;
	if (!Graph_GetNode (g, id, &n)) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT SET/REMOVE_LABELS references node %" PRIu64
				" which doesn't exist locally", id) ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// read labels count
	//--------------------------------------------------------------------------

	uint8_t lbl_count ;
	fread_assert (&lbl_count, sizeof (lbl_count), stream) ;
	ASSERT (lbl_count > 0) ;

	GrB_Vector lbls [lbl_count] ;

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	for (uint16_t i = 0; i < lbl_count; i++) {
		LabelID l ;
		fread_assert (&l, sizeof (LabelID), stream) ;
		Schema *s = GraphContext_GetSchemaByID (gc, l, SCHEMA_NODE) ;
		if (s == NULL) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT SET/REMOVE_LABELS references unknown "
					"label schema %d", l) ;
			// clean up label vectors created so far
			for (uint16_t j = 0; j < i; j++) {
				GrB_OK (GrB_free (lbls + j)) ;
			}
			return false ;
		}

		GrB_Vector V = NULL ;
		GrB_OK (GrB_Vector_new (&V, GrB_BOOL, Graph_NodeCap (g))) ;
		GrB_OK (GrB_set (V, (char*) Schema_GetName (s), GrB_NAME)) ;
		GrB_OK (GrB_Vector_setElement (V, true, id)) ;
		lbls [i] = V ;
	}

	//--------------------------------------------------------------------------
	// update node labels
	//--------------------------------------------------------------------------

	if (add) {
		GraphHub_UpdateNodeLabels (gc, lbls, lbl_count, NULL, 0, false) ;
	} else {
		GraphHub_UpdateNodeLabels (gc, NULL, 0, lbls, lbl_count, false) ;
	}

	// clean up
	for (uint16_t i = 0; i < lbl_count; i++) {
		GrB_OK (GrB_free (lbls + i)) ;
	}

	return true ;
}

// effect format:
//   [EffectType]         effect type tag
//   [GxB serialized]     GxB_Vector_serialize blob of the node vector
//
// returns false if the effect's payload is malformed
static bool ApplyLabels_V2
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	bool add           // add or remove labels
) {

	GrB_Index blob_size = 0 ;
	fread_assert (&blob_size, sizeof(blob_size), stream) ;
	if (blob_size == 0) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT SET/REMOVE_LABELS carries an empty payload") ;
		return false ;
	}

	void *blob = rm_malloc (blob_size) ;
	fread_assert (blob, blob_size, stream) ;

	GrB_Vector w = NULL ;
	GrB_OK (GxB_Vector_deserialize (&w, NULL, blob, blob_size, NULL)) ;

	rm_free (blob) ;

	if (add) {
		GraphHub_UpdateNodeLabels (gc, &w, 1, NULL, 0, false) ;
	} else {
		GraphHub_UpdateNodeLabels (gc, NULL, 0, &w, 1, false) ;
	}

	// clean up
	GrB_OK (GrB_free (&w)) ;

	return true ;
}

// returns false if the schema this effect introduces already exists locally
// (replica has diverged from the master)
static bool ApplyAddSchema
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	// read schema type
	SchemaType t;
	fread_assert(&t, sizeof(t), stream);

	// read schema name
	// read string length
	size_t l;
	fread_assert(&l, sizeof(l), stream);

	// read string
	char schema_name[l];
	fread_assert(schema_name, l, stream);

	// create schema
	bool created = false ;
	Schema *s = GraphContext_FindOrAddSchema (gc, schema_name, t, &created) ;
	if (s == NULL || created == false) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_SCHEMA targets schema '%s' which already "
				"exists locally", schema_name) ;
		return false ;
	}

	return true ;
}

// returns false if the attribute this effect introduces already exists
// locally (replica has diverged from the master)
static bool ApplyAddAttribute
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------

	// read attribute name length
	size_t l ;
	fread_assert (&l, sizeof (l), stream) ;

	// read attribute name
	char attr[l] ;
	fread_assert (attr, l, stream) ;

	// attr should not exist
	if (GraphContext_GetAttributeID (gc, attr) != ATTRIBUTE_ID_NONE) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT ADD_ATTRIBUTE targets attribute '%s' which "
				"already exists locally", attr) ;
		return false ;
	}

	// add attribute
	GraphHub_FindOrAddAttribute (gc, attr, false) ;

	return true ;
}

// resolve & verify a schema referenced by an effect via its id+name pair
// (see design principle #2 in the index/constraint effects: the id is
// authoritative - it's only valid because every schema mutation is itself
// an effect, applied in the same order on every replica - the name is a
// cheap cross-check that surfaces divergence instead of silently trusting
// a stale/incorrect id)
//
// returns NULL if the replica has diverged from the master (the id doesn't
// resolve locally, or resolves to a schema with a different name)
Schema *VerifySchema
(
	GraphContext *gc,   // graph to operate on
	SchemaType st,      // schema type (node/edge)
	int label_id,       // expected label/relationship-type id
	const char *label   // expected label/relationship-type name
) {
	Schema *s = GraphContext_GetSchemaByID (gc, label_id, st) ;
	if (s == NULL || strcmp (Schema_GetName (s), label) != 0) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT references label/relationship-type '%s' "
				"(id %d) which doesn't match local schema state",
				label, label_id) ;
		return NULL ;
	}

	return s ;
}

// resolve & verify an attribute referenced by an effect via its id+name pair
// returns false if the replica has diverged from the master
bool VerifyAttribute
(
	GraphContext *gc,     // graph to operate on
	AttributeID attr_id,  // expected attribute id
	const char *attr      // expected attribute name
) {
	const char *local_name = GraphContext_GetAttributeName (gc, attr_id) ;
	if (local_name == NULL || strcmp (local_name, attr) != 0) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT references attribute '%s' (id %u) which "
				"doesn't match local attribute state", attr, attr_id) ;
		return false ;
	}

	return true ;
}

// read (attribute id, attribute name) pairs off of stream, as encoded by
// EffectsBuffer_AddCreateConstraintEffect / EffectsBuffer_AddDropConstraintEffect
//
// returns props attribute-name allocations via 'props' (caller must free each
// entry with rm_free)
void ReadConstraintAttributes
(
	FILE *stream,           // effects stream
	uint8_t n,              // number of attributes
	AttributeID *attr_ids,  // [output] attribute ids
	char **props            // [output] mutable attribute-name view
) {
	for (uint8_t i = 0; i < n; i++) {
		fread_assert (attr_ids + i, sizeof (AttributeID), stream) ;

		size_t l ;
		fread_assert (&l, sizeof (l), stream) ;
		char *name = rm_malloc (l) ;
		fread_assert (name, l, stream) ;
		props [i] = name ;
	}
}

// process Update_Edge effect
static void ApplyUpdateEdge
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    attribute ID
	//    attribute value
	//--------------------------------------------------------------------------
	
	SIValue v;            // updated value
	AttributeID attr_id;  // entity ID

	NodeID     s_id = INVALID_ENTITY_ID;       // edge src node ID
	NodeID     t_id = INVALID_ENTITY_ID;       // edge dest node ID
	RelationID r_id = GRAPH_UNKNOWN_RELATION;  // edge rel-type

	EntityID id = INVALID_ENTITY_ID;

	//--------------------------------------------------------------------------
	// read edge ID
	//--------------------------------------------------------------------------

	fread_assert(&id, sizeof(EntityID), stream);
	ASSERT(id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read relation ID
	//--------------------------------------------------------------------------

	fread_assert(&r_id, sizeof(RelationID), stream);
	ASSERT(r_id >= 0);

	//--------------------------------------------------------------------------
	// read src ID
	//--------------------------------------------------------------------------

	fread_assert(&s_id, sizeof(NodeID), stream);
	ASSERT(s_id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read dest ID
	//--------------------------------------------------------------------------

	fread_assert(&t_id, sizeof(NodeID), stream);
	ASSERT(t_id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert(&attr_id, sizeof(AttributeID), stream);

	//--------------------------------------------------------------------------
	// read attribute value
	//--------------------------------------------------------------------------

	v = SIValue_FromBinary(stream);
	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);

	GraphHub_UpdateEdgeProperty(gc, id, r_id, s_id, t_id, attr_id, v);
}

// process UpdateNode effect
static void ApplyUpdateNode
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    entity ID
	//    attribute ID
	//    attribute value
	//--------------------------------------------------------------------------

	SIValue v;            // updated value
	AttributeID attr_id;  // entity ID

	EntityID id = INVALID_ENTITY_ID;

	//--------------------------------------------------------------------------
	// read node ID
	//--------------------------------------------------------------------------

	fread_assert(&id, sizeof(EntityID), stream);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert(&attr_id, sizeof(AttributeID), stream);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	v = SIValue_FromBinary(stream);
	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);

	GraphHub_UpdateNodeProperty(gc, id, attr_id, v);
}

// process DeleteNode effect
// returns false if the effect references a node that doesn't exist locally
// (replica has diverged from the master); processing stops immediately,
// without flushing the in-progress batch, once that's detected
static bool ApplyDeleteNode
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	size_t l           // length of stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    node ID
	//--------------------------------------------------------------------------

	EntityID id;                             // node ID
	Graph *g = GraphContext_GetGraph (gc) ;  // graph to delete node from

	int i = 0 ;                    // size of batch
	enum { batch_size = 4096 } ;   // compile-time constant, avoids a VLA
	Node nodes[batch_size] ;       // nodes

	while (true) {
		Node *n = nodes + i ;
		// read node ID off of stream
		fread_assert (&n->id, sizeof(EntityID), stream) ;

		if (!Graph_GetNode (g, n->id, nodes + i)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT DELETE_NODE references node %" PRIu64
					" which doesn't exist locally", n->id) ;
			return false ;
		}

		i++ ;

		if (i == batch_size) {
			// flush batch
			GraphHub_DeleteNodes (gc, nodes, i, false) ;
			i = 0 ;
		}

		// have we reached the end of the stream ?
		if (ftell (stream) >= l) {
			break ;
		}

		// check if the next item in the stream is a EFFECT_DELETE_NODE effect
		EffectType t = ReadEffectType (stream) ;
		if (t != EFFECT_DELETE_NODE) {
			// go back sizeof (EffectType) bytes
			fseek (stream, -((long)sizeof (EffectType)), SEEK_CUR) ;
			break ;
		}
	}

	// flush any remaining node deletions
	if (i > 0) {
		// flush batch
		GraphHub_DeleteNodes (gc, nodes, i, false) ;
	}

	return true ;
}

// process DeleteEdge effect
// returns false if the effect references an edge that doesn't exist locally
// (replica has diverged from the master); processing stops immediately,
// without flushing the in-progress batch, once that's detected
static bool ApplyDeleteEdge
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	size_t l           // length of stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	int i = 0 ;                     // size of current batch
	enum { batch_size = 4096 } ;    // compile-time constant, avoids a VLA
	Edge edges[batch_size] ;          // edges

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EntityID id ;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _edge_desc ;
	#pragma pack(pop)

	Graph *g = GraphContext_GetGraph (gc) ;  // graph to delete edge from

	while (true) {
		// read edge description from stream
		fread_assert (&_edge_desc, sizeof (_edge_desc), stream) ;

		Edge *e = edges + i ;

		if (!Graph_GetEdge (g, _edge_desc.id, edges + i)) {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT DELETE_EDGE references edge %" PRIu64
					" which doesn't exist locally", _edge_desc.id) ;
			return false ;
		}

		// set edge relation, src and destination node
		e->id         = _edge_desc.id      ;
		e->src_id     = _edge_desc.src_id  ;
		e->dest_id    = _edge_desc.dest_id ;
		e->relationID = _edge_desc.r       ;

		i++ ;

		// check if batch is full
		if (i == batch_size) {
			// flush batch
			GraphHub_DeleteEdges (gc, edges, i, false, false) ;
			i = 0 ;
		}

		// have we reached the end of the stream ?
		if (ftell (stream) >= l) {
			break ;
		}

		// check if the next item in the stream is a EFFECT_DELETE_EDGE effect
		EffectType t = ReadEffectType (stream) ;
		if (t != EFFECT_DELETE_EDGE) {
			// go back sizeof (EffectType) bytes
			fseek (stream, -((long)sizeof (EffectType)), SEEK_CUR) ;
			break ;
		}
	}

	// flush last batch
	if (i > 0) {
		GraphHub_DeleteEdges (gc, edges, i, false, false) ;
	}

	return true ;
}

// returns false in case of effect encode/decode version mismatch
static bool ValidateVersion
(
	FILE *stream,  // effects stream
	uint8_t *v
) {
	ASSERT (v      != NULL) ;
	ASSERT (stream != NULL) ;

	// read version
	fread_assert (v, sizeof (uint8_t), stream) ;

	if (*v > EFFECTS_VERSION) {
		// unexpected effects version
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT version mismatch expected version <= %d got: %d",
				EFFECTS_VERSION, *v) ;
		return false ;
	}

	return true ;
}

// applies effects encoded in buffer
// returns false if the replica has diverged from the master (an effect
// referenced graph state that doesn't exist locally, or the encoding
// version didn't match); processing stops at the first such failure
bool Effects_Apply
(
	GraphContext *gc,          // graph to operate on
	const char *effects_buff,  // encoded effects
	size_t l                   // size of buffer
) {
	// validations
	ASSERT (l > 0) ;  // buffer can't be empty
	ASSERT (effects_buff != NULL) ;  // buffer can't be NULL

	// read buffer in a stream fashion
	FILE *stream = fmemopen ((void*)effects_buff, l, "r") ;

	// validate effects version
	uint8_t version ;
	if (ValidateVersion (stream, &version) == false) {
		// replica/primary out of sync
		fclose (stream) ;
		return false ;
	}

	bool ok = true ;

	// as long as there's data in stream
	while (ok && ftell (stream) < l) {
		// read effect type
		EffectType t = ReadEffectType (stream) ;
		switch (t) {
			case EFFECT_DELETE_NODE:
				ok = ApplyDeleteNode (stream, gc, l) ;
				break ;

			case EFFECT_DELETE_EDGE:
				ok = ApplyDeleteEdge (stream, gc, l) ;
				break ;

			case EFFECT_UPDATE_NODE:
				ApplyUpdateNode (stream, gc) ;
				break ;

			case EFFECT_UPDATE_EDGE:
				ApplyUpdateEdge (stream, gc) ;
				break ;

			case EFFECT_CREATE_NODE:
				ApplyCreateNode (stream, gc) ;
				break ;

			case EFFECT_CREATE_EDGE:
				ApplyCreateEdge (stream, gc, l) ;
				break ;

			case EFFECT_SET_LABELS:
				if (unlikely (version == 1)) {
					ok = ApplyLabels (stream, gc, true) ;
				} else {
					ok = ApplyLabels_V2 (stream, gc, true) ;
				}
				break ;

			case EFFECT_REMOVE_LABELS:
				if (unlikely (version == 1)) {
					ok = ApplyLabels (stream, gc, false) ;
				} else {
					ok = ApplyLabels_V2 (stream, gc, false) ;
				}
				break ;

			case EFFECT_ADD_SCHEMA:
				ok = ApplyAddSchema (stream, gc) ;
				break ;

			case EFFECT_ADD_ATTRIBUTE:
				ok = ApplyAddAttribute (stream, gc) ;
				break ;

			case EFFECT_CREATE_INDEX:
				ok = ApplyCreateIndex (stream, gc) ;
				break ;

			case EFFECT_DROP_INDEX:
				ok = ApplyDropIndex (stream, gc) ;
				break ;

			case EFFECT_CREATE_CONSTRAINT:
				ok = ApplyCreateConstraint (stream, gc) ;
				break ;

			case EFFECT_DROP_CONSTRAINT:
				ok = ApplyDropConstraint (stream, gc) ;
				break ;

			default:
				RedisModule_Log (NULL, "warning",
						"GRAPH.EFFECT encountered unknown effect type %d", t) ;
				ok = false ;
				break ;
		}
	}

	// close stream
	fclose (stream) ;

	return ok ;
}

