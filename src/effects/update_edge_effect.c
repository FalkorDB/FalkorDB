/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "effects.h"
#include "../graph/graph_hub.h"

#include <stdio.h>

// process Update_Edge effect
// returns false if the effect references an edge, or a relationship-type,
// that doesn't exist locally (replica has diverged from the master)
bool ApplyUpdateEdge
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
	Graph *g = GraphContext_GetGraph (gc) ;

	EdgeID     id   = INVALID_ENTITY_ID ;
	NodeID     s_id = INVALID_ENTITY_ID ;       // edge src node ID
	NodeID     t_id = INVALID_ENTITY_ID ;       // edge dest node ID
	RelationID r_id = GRAPH_UNKNOWN_RELATION ;  // edge rel-type

	//--------------------------------------------------------------------------
	// read edge ID
	//--------------------------------------------------------------------------

	fread_assert (&id, sizeof (EntityID), stream) ;

	//--------------------------------------------------------------------------
	// read relation ID
	//--------------------------------------------------------------------------

	fread_assert (&r_id, sizeof (RelationID), stream) ;

	//--------------------------------------------------------------------------
	// read src ID
	//--------------------------------------------------------------------------

	fread_assert (&s_id, sizeof (NodeID), stream) ;

	//--------------------------------------------------------------------------
	// read dest ID
	//--------------------------------------------------------------------------

	fread_assert (&t_id, sizeof (NodeID), stream) ;

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert (&attr_id, sizeof (AttributeID), stream) ;

	//--------------------------------------------------------------------------
	// read attribute value
	//--------------------------------------------------------------------------

	v = SIValue_FromBinary (stream) ;

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// validate the followings:
	// 1. value
	// 2. attribute id
	// 3. source node
	// 4. target node
	// 5. relationship type
	// 6. edge exists

	// validate value type
	if (!(SI_TYPE (v) & (SI_VALID_PROPERTY_VALUE | T_NULL)))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE invalid value") ;
		goto fail ;
	}

	// validate attribute id
	if (!((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull (v)) &&
		   attr_id != ATTRIBUTE_ID_NONE))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE illegal attribute_id %d", attr_id) ;
		goto fail ;
	}

	// make sure graph is aware of attribute id
	if (attr_id != ATTRIBUTE_ID_ALL && !GraphContext_HasAttribute (gc, attr_id))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE unknown attribute_id %d", attr_id) ;
		goto fail ;
	}

	// make sure graph is aware of relation id
	if (r_id < 0 || r_id >= GraphContext_SchemaCount (gc, SCHEMA_EDGE))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE references relationship type %d"
			" which doesn't exist locally", r_id) ;
		goto fail ;
	}

	if (id   == INVALID_ENTITY_ID ||
		s_id == INVALID_ENTITY_ID ||
		t_id == INVALID_ENTITY_ID)
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE invalid entity id") ;
		goto fail ;
	}

	// make sure source node exists
	if (Graph_HasNode (g, s_id) == false)
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE references node %" PRIu64
			" which doesn't exist locally", s_id) ;
		goto fail ;
	}

	// make sure target node exists
	if (Graph_HasNode (g, t_id) == false)
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE references node %" PRIu64
			" which doesn't exist locally", t_id) ;
		goto fail ;
	}

	// make sure edge exists
	if (Graph_HasEdge (g, id, s_id, t_id, r_id) == false)
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_EDGE references edge %" PRIu64
			" doesn't exist locally", id) ;
		goto fail ;
	}

	GraphHub_UpdateEdgeProperty (gc, id, r_id, s_id, t_id, attr_id, v) ;
	return true ;

fail:
	SIValue_Free (v) ;
	return false ;
}

