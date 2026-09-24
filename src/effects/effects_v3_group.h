/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_bytes.h"
#include "effects_v3.h"
#include "effects_v3_id_list.h"

// groups a query's mutations into v3 records as they arrive
//
// A v3 record states its count and its shape ahead of its rows, so nothing can
// be serialized until the query stops producing effects. Entities are filed into
// their group on arrival and the records are emitted once at the end.
//
// FOUR RULES LIVE HERE, each a byte-for-byte divergence if missed. None is
// visible in the wire format, so the conformance corpus cannot arbitrate any of
// them: a fixture pins what one record looks like, not which entities are in it.
//
//   1. ONE RECORD PER (opcode, shape). Shape is the attribute IDS and never the
//      values, plus the labels for a node and the relationship type for an edge.
//
//   2. GROUPS ARE SORTED BY KEY before emission. A hash table iterates
//      arbitrarily, so the same query could emit its records in a different
//      order on two runs of one engine, let alone on two engines.
//
//   3. LABEL SETS ARE NORMALISED ASCENDING. Callers keep their own order, so
//      [7,8] and [8,7] arrive for the same set; they are ONE shape.
//
//   4. A NEW ATTRIBUTE IS ANNOUNCED ONCE per payload, not once per group that
//      uses it - the only rule that spans groups, so the one a per-group
//      implementation satisfies locally and violates globally.
typedef struct EffectsV3Grouping EffectsV3Grouping;

// create an accumulator
EffectsV3Grouping *EffectsV3Grouping_New(void);

// free it and everything it holds
void EffectsV3Grouping_Free
(
	EffectsV3Grouping *g  // accumulator
);

// announce a schema, at most once per (type, id)
//
// emitted ahead of every record that references the ids it introduces, because
// a bulk record carries a bare id and the replica has to have seen the name
void EffectsV3Grouping_AddSchema
(
	EffectsV3Grouping *g,  // accumulator
	SchemaType t,          // node or edge
	int id,                // the id being introduced
	const char *name       // its name
);

// announce an attribute, at most ONCE per payload
//
// v3's ADD_ATTRIBUTE carries no node/relationship discriminator - correctly,
// since C has a single dictionary - so announcing per entity kind would
// introduce the same id twice
void EffectsV3Grouping_AddAttribute
(
	EffectsV3Grouping *g,  // accumulator
	AttributeID id,        // the attribute id
	const char *name       // its name
);

// file a node-shaped entity into its group
//
// 'labels' may arrive in any order; it is normalised before it becomes a key.
// 'values' is n_attrs values in the order 'attr_ids' states, and is copied
void EffectsV3Grouping_AddNode
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // CREATE_NODE, UPDATE_NODE, DELETE_NODE,
	                             // SET_LABELS or REMOVE_LABELS
	const LabelID *labels,       // the node's labels, any order
	uint16_t n_labels,           // how many
	uint64_t id,                 // the entity id
	const AttributeID *attr_ids, // attribute ids, ascending; NULL if none
	const SIValue *values,       // one value per attribute id; NULL if none
	uint16_t n_attrs             // how many attributes
);

// file an edge-shaped entity into its group
//
// src and dst are ignored for UPDATE_EDGE, which does not carry endpoints
void EffectsV3Grouping_AddEdge
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // CREATE_EDGE, UPDATE_EDGE or DELETE_EDGE
	RelationID relation_id,      // the relationship type
	uint64_t id,                 // the edge id
	uint64_t src,                // source node id
	uint64_t dst,                // destination node id
	const AttributeID *attr_ids, // attribute ids, ascending; NULL if none
	const SIValue *values,       // one value per attribute id; NULL if none
	uint16_t n_attrs             // how many attributes
);

// stage ONE attribute of an entity's update
//
// v2's write API hands over one (entity, attribute, value) at a time, but a v3
// record's shape is the entity's WHOLE updated attribute set - so an update is
// staged per entity and folded into a group at emission, once the query stops
// producing attributes for it.
//
// The value is encoded IMMEDIATELY, because the SIValue belongs to the caller
// and will not outlive the call. The tagged blobs are concatenated in
// attribute-id order at the flush, which is what lets attributes arrive in any
// order and still produce one canonical shape.
//
// A T_NULL value is a removal and is staged like any other: the attribute is
// part of the shape and the null is what instructs the replica to drop it.
void EffectsV3Grouping_StageUpdate
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // UPDATE_NODE or UPDATE_EDGE
	uint64_t id,            // entity id
	const LabelID *labels,  // the node's labels, any order; NULL for an edge
	uint16_t n_labels,      // how many
	RelationID relation_id, // relationship type; ignored for a node
	AttributeID attr_id,    // the attribute being set
	SIValue value           // its new value, or a null to remove it
);


// file a constraint DDL record
//
// Singular like a schema announcement - one statement, one record, no count and
// no statement-level accumulation, because C hands over the whole property list
// in one call.
//
// 'status' is the one field v3 states that C's own writer never received: a
// replica never validates, so the announcement is the only thing that can tell
// an enforcing constraint from one still building, and it is what makes the
// second announcement converge on the first rather than duplicate it. DROP
// ignores it.
void EffectsV3Grouping_AddConstraint
(
	EffectsV3Grouping *g,         // accumulator
	EffectType opcode,            // CREATE_CONSTRAINT or DROP_CONSTRAINT
	uint32_t constraint_type,     // unique or mandatory
	uint32_t entity_type,         // 1-BASED - a node is 1, not 0
	uint32_t status,              // ConstraintStatus; CREATE only
	int label_id,                 // label/relationship-type id
	const char *label,            // its name, the cross-check
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attr_names,      // their names
	uint8_t n                     // how many
);

// file ONE FIELD of an index statement
//
// The per-attribute problem again. C emits one effect per FIELD, while a v3
// CREATE_INDEX is one record per STATEMENT with a counted field list - so
// `CREATE INDEX FOR (n:P) ON (n.a, n.b)` arrives as two calls that must become
// one record, and the field list is not known until the statement stops
// producing fields.
//
// Fields fold into an existing record when they agree on EVERY component of its
// key: opcode, schema, field type AND the options. The options are part of the
// key because a record carries ONE options block that apply hands to every field
// in it, so two fields disagreeing about weight cannot share a record without
// one silently taking the other's. C reaches that case through
// db.idx.fulltext.createNodeIndex, which rebuilds weight, phonetic and nostem
// per field (proc_fulltext_create_index.c:320-327); CREATE INDEX ... OPTIONS
// passes one map to every field (index_operations.c:457) and so always makes
// exactly one record.
//
// INDEX-LEVEL OPTIONS ARE STATED ONCE PER INDEX PER PAYLOAD. language and
// stopwords belong to the index rather than the field, and their setters are
// asymmetric: Index_SetLanguage objects only if the language DIFFERS, but
// Index_SetStopwords objects if stopwords are set AT ALL - so a statement that
// splits into two records would fail on the second. They are dropped from every
// record after the first naming the same index.
//
// 'options' is C's own options map, where a key PRESENT is exactly an option the
// statement STATED - the conversion is a lookup per key and never a default.
// DROP_INDEX ignores it.
void EffectsV3Grouping_AddIndexField
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // CREATE_INDEX or DROP_INDEX
	SchemaType schema_type, // node or edge
	int schema_id,          // label/relationship-type id
	const char *label,      // its name, the cross-check
	uint32_t field_type,    // IndexFieldType, a BIT SET - test with &
	AttributeID attr_id,    // the field being indexed
	const char *attr_name,  // its name
	SIValue options         // C's options map; ignored for a drop
);

// how many records the accumulator would emit
//
// NOT const: staged updates are folded into their groups here if they have not
// been already. Counting without folding would report fewer records than the
// payload contains, and a caller deciding whether to send would act on it.
uint32_t EffectsV3Grouping_RecordCount
(
	EffectsV3Grouping *g  // accumulator
);

// write every record, schema and attribute announcements first, then the
// groups in key order
void EffectsV3Grouping_Encode
(
	EffectsV3Grouping *g,  // accumulator
	EffectsBytes *out      // sink
);
