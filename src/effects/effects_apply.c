/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "../graph/graph_hub.h"

#include <stdio.h>

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
	for (uint16_t i = 0; i < lbl_count; i++) {
		fread_assert (labels + i, sizeof (LabelID), stream) ;
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
		ASSERT (array_len (batch) == *i) ;
		ASSERT (array_len (batch) == array_len (sets)) ;

		GraphHub_CreateEdges (gc, batch, r, sets, false) ;
		array_clear (sets) ;
		array_clear (batch) ;
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

	int i = 0 ;                       // size of current batch
	const size_t batch_size = 4096 ;  // max batch size
	Edge edges[batch_size] ;          // edges

	Edge **batch = array_new (Edge *, 1) ;  // batch, points to edges
	AttributeSet *sets = array_new (AttributeSet, 1) ;  // attribute-sets

	RelationID r      = GRAPH_UNKNOWN_RELATION ;  // current edge relation id
	RelationID prev_r = GRAPH_UNKNOWN_RELATION ;  // last processed relation id

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

		array_append (sets, ReadAttributeSet(stream)) ;

		//----------------------------------------------------------------------
		// add edge to batch
		//----------------------------------------------------------------------

		r = _edge_desc.r ;

		Edge *e = edges + i ;
		Edge_SetSrcNodeID  (e, _edge_desc.src_id) ;
		Edge_SetDestNodeID (e, _edge_desc.dest_id) ;
		Edge_SetRelationID (e, _edge_desc.r) ;

		array_append (batch, e) ;
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
	array_free (sets) ;
	array_free (batch) ;
}

static void ApplyLabels
(
	FILE *stream,     // effects stream
	GraphContext *gc, // graph to operate on
	bool add          // add or remove labels
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
	Graph *g = gc->g ;

	bool found = Graph_GetNode (g, id, &n) ;
	ASSERT (found == true) ;

	//--------------------------------------------------------------------------
	// read labels count
	//--------------------------------------------------------------------------

	uint8_t lbl_count ;
	fread_assert (&lbl_count, sizeof (lbl_count), stream) ;
	ASSERT (lbl_count > 0) ;

	// TODO: move to LabelID
	uint n_add_labels          = 0 ;
	uint n_remove_labels       = 0 ;
	const char **add_labels    = NULL ;
	const char **remove_labels = NULL ;
	const char *lbl[lbl_count] ;

	// assign lbl to the appropriate array
	if (add) {
		add_labels = lbl ;
		n_add_labels = lbl_count ;
	} else {
		remove_labels = lbl ;
		n_remove_labels = lbl_count ;
	}

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	for (uint16_t i = 0; i < lbl_count; i++) {
		LabelID l ;
		fread_assert (&l, sizeof (LabelID), stream) ;
		Schema *s = GraphContext_GetSchemaByID (gc, l, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;
		lbl[i] = Schema_GetName (s) ;
	}

	//--------------------------------------------------------------------------
	// update node labels
	//--------------------------------------------------------------------------

	GraphHub_UpdateNodeLabels (gc, &n, add_labels, remove_labels, n_add_labels,
			n_remove_labels, false) ;
}

static void ApplyAddSchema
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
	GraphHub_AddSchema(gc, schema_name, t, false);
}

static void ApplyAddAttribute
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
	const char attr[l] ;
	fread_assert (attr, l, stream) ;

	// attr should not exist
	ASSERT (GraphContext_GetAttributeID (gc, attr) == ATTRIBUTE_ID_NONE) ;

	// add attribute
	GraphHub_FindOrAddAttribute (gc, attr, false) ;
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
static void ApplyDeleteNode
(
	FILE *stream,      // effects stream
	GraphContext *gc,  // graph to operate on
	size_t l           // length of stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    node ID
	//--------------------------------------------------------------------------
	
	EntityID id;       // node ID
	Graph *g = gc->g;  // graph to delete node from

	int i = 0 ;                      // size of batch
	const size_t batch_size = 4096 ; // max batch size
	Node nodes[batch_size] ;         // nodes

	while (true) {
		Node *n = nodes + i ;
		// read node ID off of stream
		fread_assert (&n->id, sizeof(EntityID), stream) ;

		// debug assert node exists
		ASSERT (Graph_GetNode (g, n->id, nodes + i)) ;

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
}

// process DeleteNode effect
static void ApplyDeleteEdge
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

	int i = 0 ;                       // size of current batch
	const size_t batch_size = 4096 ;  // max batch size
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

	Graph *g = gc->g ;  // graph to delete edge from

	while (true) {
		// read edge description from stream
		fread_assert (&_edge_desc, sizeof (_edge_desc), stream) ;

		Edge *e = edges + i ;

		// debug assert edge exists
		ASSERT (Graph_GetEdge (g, _edge_desc.id, edges + i) == true) ;

		// set edge relation, src and destination node
		e->id         = _edge_desc.id      ;
		e->src_id     = _edge_desc.src_id  ;
		e->dest_id    = _edge_desc.dest_id ;
		e->relationID = _edge_desc.r       ;

		i++ ;

		// check if batch is full
		if (i == batch_size) {
			// flush batch
			GraphHub_DeleteEdges (gc, edges, i, false) ;
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
		GraphHub_DeleteEdges (gc, edges, i, false) ;
	}
}

// returns false in case of effect encode/decode version mismatch
static bool ValidateVersion
(
	FILE *stream  // effects stream
) {
	ASSERT(stream != NULL);

	// read version
	uint8_t v;
	fread_assert(&v, sizeof(uint8_t), stream);

	if(v != EFFECTS_VERSION) {
		// unexpected effects version
		RedisModule_Log(NULL, "warning",
				"GRAPH.EFFECT version mismatch expected: %d got: %d",
				EFFECTS_VERSION, v);
		return false;
	}

	return true;
}

// applys effects encoded in buffer
void Effects_Apply
(
	GraphContext *gc,          // graph to operate on
	const char *effects_buff,  // encoded effects
	size_t l                   // size of buffer
) {
	// validations
	ASSERT(l > 0);  // buffer can't be empty
	ASSERT(effects_buff != NULL);  // buffer can't be NULL

	// read buffer in a stream fashion
	FILE *stream = fmemopen((void*)effects_buff, l, "r");

	// validate effects version
	if(ValidateVersion(stream) == false) {
		// replica/primary out of sync
		exit(1);
	}

	// lock graph for writing
	Graph *g = GraphContext_GetGraph(gc);
	Graph_AcquireWriteLock(g);

	// update graph sync policy
	MATRIX_POLICY policy = Graph_SetMatrixPolicy(g, SYNC_POLICY_RESIZE);

	// as long as there's data in stream
	while (ftell (stream) < l) {
		// read effect type
		EffectType t = ReadEffectType (stream) ;
		switch (t) {
			case EFFECT_DELETE_NODE:
				ApplyDeleteNode (stream, gc, l) ;
				break ;
			case EFFECT_DELETE_EDGE:
				ApplyDeleteEdge (stream, gc, l) ;
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
				ApplyLabels (stream, gc, true) ;
				break ;
			case EFFECT_REMOVE_LABELS: 
				ApplyLabels (stream, gc, false) ;
				break ;
			case EFFECT_ADD_SCHEMA:
				ApplyAddSchema (stream, gc) ;
				break ;
			case EFFECT_ADD_ATTRIBUTE:
				ApplyAddAttribute (stream, gc) ;
				break ;
			default:
				ASSERT (false && "unknown effect type") ;
				break ;
		}
	}

	// restore graph sync policy
	Graph_SetMatrixPolicy(g, policy);

	// release write lock
	Graph_ReleaseLock(g);

	// close stream
	fclose(stream);
}

