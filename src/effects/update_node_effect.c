/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "effects.h"
#include "../graph/graph_hub.h"

#include <stdio.h>

// process UpdateNode effect
// returns false if the effect references a node that doesn't exist locally
// (replica has diverged from the master)
bool ApplyUpdateNode
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
	Graph *g = GraphContext_GetGraph (gc) ;

	EntityID id = INVALID_ENTITY_ID;

	//--------------------------------------------------------------------------
	// read node ID
	//--------------------------------------------------------------------------

	fread_assert (&id, sizeof (EntityID), stream) ;

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert (&attr_id, sizeof (AttributeID), stream) ;

	//--------------------------------------------------------------------------
	// read value
	//--------------------------------------------------------------------------

	v = SIValue_FromBinary (stream) ;

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// validate the followings:
	// 1. value
	// 2. attribute id
	// 3. node id
	// 4. node exists

	// validate value type
	if (!(SI_TYPE (v) & (SI_VALID_PROPERTY_VALUE | T_NULL)))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_NODE invalid value") ;
		goto fail ;
	}

	// validate attribute id
	if (!((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull (v)) &&
		   attr_id != ATTRIBUTE_ID_NONE))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_NODE illegal attribute_id %d", attr_id) ;
		goto fail ;
	}

	// make sure graph is aware of attribute id
	if (attr_id != ATTRIBUTE_ID_ALL && !GraphContext_HasAttribute (gc, attr_id))
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_NODE unknown attribute_id %d", attr_id) ;
		goto fail ;
	}

	if (id == INVALID_ENTITY_ID) {
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_NODE invalid node id") ;
		goto fail ;
	}

	// make sure node exists
	if (Graph_HasNode (g, id) == false)
	{
		RedisModule_Log (NULL, "warning",
			"GRAPH.EFFECT UPDATE_NODE references node %" PRIu64
			" which doesn't exist locally", id) ;
		goto fail ;
	}

	// perform the update
	GraphHub_UpdateNodeProperty (gc, id, attr_id, v) ;
	return true ;

fail:
	SIValue_Free (v) ;
	return false ;
}

