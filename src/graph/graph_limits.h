/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <limits.h>
#include <stdint.h>

#include "entities/attribute_set.h"  // AttributeID, ATTRIBUTE_ID_ALL

// central home for a graph's structural limits and the boundaries the RDB
// decoder enforces on an untrusted payload. a count read from a payload that
// exceeds its limit is treated as corrupt data: the decode fails cleanly
// rather than driving a huge allocation or a long loop

//------------------------------------------------------------------------------
// attributes
//------------------------------------------------------------------------------

// AttributeID is a 16-bit type whose two top values are reserved sentinels
// (ATTRIBUTE_ID_NONE, ATTRIBUTE_ID_ALL), so a graph holds at most
// ATTRIBUTE_ID_ALL distinct attribute names - enforced in
// GraphContext_FindOrAddAttribute (EMSG_MAX_ATTRIBUTES_EXCEEDED). an entity
// can't reference more attributes than the graph defines
#define GRAPH_MAX_ATTRIBUTES  ATTRIBUTE_ID_ALL

//------------------------------------------------------------------------------
// labels & relationship types
//------------------------------------------------------------------------------

// LabelID / RelationID are 'int' and otherwise uncapped; bound a payload-
// supplied count well below INT_MAX so garbage fails fast instead of sizing a
// huge allocation
#define GRAPH_MAX_LABELS     (1 << 16)
#define GRAPH_MAX_RELATIONS  (1 << 16)

//------------------------------------------------------------------------------
// constraints
//------------------------------------------------------------------------------

// a constraint stores its field count in a uint8 (Constraint.n_attr)
#define GRAPH_MAX_CONSTRAINT_FIELDS  UINT8_MAX

//------------------------------------------------------------------------------
// decode payload boundaries (not graph structure, but untrusted-count bounds)
//------------------------------------------------------------------------------

// elements in a single decoded SIArray
#define GRAPH_MAX_SIARRAY_LEN  (1 << 28)

// encode-state payloads described by one (virtual) key's schema
#define GRAPH_MAX_KEY_PAYLOADS  (1 << 16)
