/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_group.h"
#include "effects_v3_encode.h"
#include "effects_internal.h"
#include "../util/rmalloc.h"

#include <stdlib.h>
#include <string.h>

// one (opcode, shape) group, with its rows accumulating
typedef struct {
	EffectType opcode;

	// the shape, whichever half of it this opcode uses
	LabelID   *labels;       // owned, ascending
	uint16_t   n_labels;
	RelationID relation_id;
	AttributeID *attr_ids;   // owned
	uint16_t     n_attrs;

	// the rows
	EffectsV3IdListBuilder *ids;
	EffectsV3IdListBuilder *src;  // edge create/delete only
	EffectsV3IdListBuilder *dst;
	EffectsBytes *values;         // count * n_attrs SIValues, row-major
	uint32_t count;               // entities filed here
} Group;

// a schema or attribute announcement, kept in arrival order
typedef struct {
	EffectType  opcode;  // ADD_SCHEMA or ADD_ATTRIBUTE
	SchemaType  schema_type;
	int         schema_id;
	AttributeID attr_id;
	char       *name;    // owned
} Announcement;

struct EffectsV3Grouping {
	Group *groups;
	uint32_t n_groups;
	uint32_t cap_groups;

	Announcement *announcements;
	uint32_t n_announcements;
	uint32_t cap_announcements;
};

//------------------------------------------------------------------------------
// shape keys
//------------------------------------------------------------------------------

static int _cmp_u32(uint32_t a, uint32_t b) {
	return (a < b) ? -1 : (a > b) ? 1 : 0;
}

// total order over (opcode, shape)
//
// Any total order would keep one engine self-consistent; this one has to match
// the other engine's, so it compares the shape's components in the order the
// wire states them: the opcode, then the schema half (labels or relationship
// type), then the attribute ids.
static int _cmp_group(const void *va, const void *vb) {
	const Group *a = va, *b = vb;

	int c = _cmp_u32((uint32_t)a->opcode, (uint32_t)b->opcode);
	if(c != 0) return c;

	c = _cmp_u32(a->n_labels, b->n_labels);
	if(c != 0) return c;
	for(uint16_t i = 0; i < a->n_labels; i++) {
		c = _cmp_u32((uint32_t)a->labels[i], (uint32_t)b->labels[i]);
		if(c != 0) return c;
	}

	c = _cmp_u32((uint32_t)a->relation_id, (uint32_t)b->relation_id);
	if(c != 0) return c;

	c = _cmp_u32(a->n_attrs, b->n_attrs);
	if(c != 0) return c;
	for(uint16_t i = 0; i < a->n_attrs; i++) {
		c = _cmp_u32(a->attr_ids[i], b->attr_ids[i]);
		if(c != 0) return c;
	}

	return 0;
}

static int _cmp_label(const void *a, const void *b) {
	LabelID x = *(const LabelID *)a, y = *(const LabelID *)b;
	return (x < y) ? -1 : (x > y) ? 1 : 0;
}

// whether a record would say nothing about any entity it names
//
// The rule, which covers three categories without a per-record table:
//
//   A record is vacuous if removing an empty block leaves it saying nothing
//   about any entity it names. Vacuous records must not be emitted. An empty
//   block that DESCRIBES the entities the record names is information and is
//   legal. A schema announcement is a binding rather than an instruction and is
//   legal regardless of whether anything references it.
//
// So the distinction is which block is the record's SUBJECT, not whether a
// block is empty:
//
//   DELETE_NODE with no labels    LEGAL - says these nodes carried no labels
//   CREATE_NODE with no attrs     LEGAL - says these nodes have no properties
//   SET_LABELS with no labels     VACUOUS - a label record's whole payload IS
//                                 its label set, so with none it is an
//                                 instruction to do nothing
//   any record naming no entities VACUOUS - it cannot say anything about any
//
// Readers deliberately TOLERATE these rather than refusing them. Rejecting
// count = 0 at the header removes parse surface, because every block would
// otherwise need a zero-length path both engines agree on; rejecting a
// zero-label record removes nothing, since DELETE_NODE and CREATE_NODE require
// the zero-length LabelSet path anyway. So rejection buys no safety and costs a
// resync loop against any peer still emitting one. Tolerate on read, refuse to
// emit.
static bool _vacuous(const Group *grp) {
	// names no entities, so it says nothing about any
	if(grp->count == 0) {
		return true;
	}

	// a label record's payload is its label set
	if((grp->opcode == EFFECT_SET_LABELS
				|| grp->opcode == EFFECT_REMOVE_LABELS)
			&& grp->n_labels == 0) {
		return true;
	}

	return false;
}

//------------------------------------------------------------------------------
// the accumulator
//------------------------------------------------------------------------------

EffectsV3Grouping *EffectsV3Grouping_New(void) {
	EffectsV3Grouping *g = rm_malloc(sizeof(EffectsV3Grouping));

	g->cap_groups        = 4;
	g->groups            = rm_calloc(g->cap_groups, sizeof(Group));
	g->n_groups          = 0;
	g->cap_announcements = 4;
	g->announcements     = rm_calloc(g->cap_announcements, sizeof(Announcement));
	g->n_announcements   = 0;

	return g;
}

// find the group matching this shape, or open one
//
// A linear scan rather than a hash table, deliberately: the number of distinct
// shapes in a query is small - a label set is parsed rather than computed, so
// it is bounded by the query text - and a scan has no iteration order to get
// wrong. The sort before emission is what makes the order normative either way.
static Group *_group_for
(
	EffectsV3Grouping *g,         // accumulator
	EffectType opcode,            // record type
	const LabelID *labels,        // labels, already normalised
	uint16_t n_labels,            // how many
	RelationID relation_id,       // relationship type, or 0
	const AttributeID *attr_ids,  // attribute ids
	uint16_t n_attrs              // how many
) {
	Group probe = {
		.opcode      = opcode,
		.labels      = (LabelID *)labels,
		.n_labels    = n_labels,
		.relation_id = relation_id,
		.attr_ids    = (AttributeID *)attr_ids,
		.n_attrs     = n_attrs,
	};

	for(uint32_t i = 0; i < g->n_groups; i++) {
		if(_cmp_group(&probe, g->groups + i) == 0) {
			return g->groups + i;
		}
	}

	if(g->n_groups == g->cap_groups) {
		g->cap_groups *= 2;
		g->groups = rm_realloc(g->groups, g->cap_groups * sizeof(Group));
	}

	Group *grp = g->groups + g->n_groups++;
	memset(grp, 0, sizeof(*grp));

	grp->opcode      = opcode;
	grp->relation_id = relation_id;
	grp->n_labels    = n_labels;
	grp->n_attrs     = n_attrs;
	grp->ids         = EffectsV3IdListBuilder_New();
	grp->values      = EffectsBytes_New(1024);

	if(n_labels > 0) {
		grp->labels = rm_malloc(sizeof(LabelID) * n_labels);
		memcpy(grp->labels, labels, sizeof(LabelID) * n_labels);
	}

	if(n_attrs > 0) {
		grp->attr_ids = rm_malloc(sizeof(AttributeID) * n_attrs);
		memcpy(grp->attr_ids, attr_ids, sizeof(AttributeID) * n_attrs);
	}

	if(opcode == EFFECT_CREATE_EDGE || opcode == EFFECT_DELETE_EDGE) {
		grp->src = EffectsV3IdListBuilder_New();
		grp->dst = EffectsV3IdListBuilder_New();
	}

	return grp;
}

// append a row's values through the shared SIValue codec
static void _append_values
(
	Group *grp,             // group to append to
	const SIValue *values,  // one value per attribute id
	uint16_t n_attrs        // how many
) {
	if(n_attrs == 0) {
		return;
	}

	EffectsBuffer *wrapper = EffectsBuffer_Wrap(grp->values);
	for(uint16_t i = 0; i < n_attrs; i++) {
		EffectsBuffer_WriteSIValue(values + i, wrapper);
	}
	EffectsBuffer_Free(wrapper);
}

void EffectsV3Grouping_AddNode
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // record type
	const LabelID *labels,       // labels, any order
	uint16_t n_labels,           // how many
	uint64_t id,                 // entity id
	const AttributeID *attr_ids, // attribute ids
	const SIValue *values,       // values
	uint16_t n_attrs             // how many
) {
	// RULE 3: normalise before the labels become a key, so [7,8] and [8,7] are
	// one shape rather than two records describing the same set
	LabelID sorted[64];
	LabelID *norm = (n_labels <= 64)
		? sorted
		: rm_malloc(sizeof(LabelID) * n_labels);

	if(n_labels > 0) {
		memcpy(norm, labels, sizeof(LabelID) * n_labels);
		qsort(norm, n_labels, sizeof(LabelID), _cmp_label);
	}

	Group *grp = _group_for(g, opcode, norm, n_labels, 0, attr_ids, n_attrs);

	EffectsV3IdListBuilder_Push(grp->ids, id);
	_append_values(grp, values, n_attrs);
	grp->count++;

	if(norm != sorted) {
		rm_free(norm);
	}
}

void EffectsV3Grouping_AddEdge
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // record type
	RelationID relation_id,      // relationship type
	uint64_t id,                 // edge id
	uint64_t src,                // source node
	uint64_t dst,                // destination node
	const AttributeID *attr_ids, // attribute ids
	const SIValue *values,       // values
	uint16_t n_attrs             // how many
) {
	Group *grp = _group_for(g, opcode, NULL, 0, relation_id, attr_ids, n_attrs);

	EffectsV3IdListBuilder_Push(grp->ids, id);

	// an update recovers its endpoints from the graph, so it carries none
	if(grp->src != NULL) {
		EffectsV3IdListBuilder_Push(grp->src, src);
		EffectsV3IdListBuilder_Push(grp->dst, dst);
	}

	_append_values(grp, values, n_attrs);
	grp->count++;
}

static Announcement *_new_announcement(EffectsV3Grouping *g) {
	if(g->n_announcements == g->cap_announcements) {
		g->cap_announcements *= 2;
		g->announcements = rm_realloc(g->announcements,
				g->cap_announcements * sizeof(Announcement));
	}

	Announcement *a = g->announcements + g->n_announcements++;
	memset(a, 0, sizeof(*a));
	return a;
}

void EffectsV3Grouping_AddSchema
(
	EffectsV3Grouping *g,  // accumulator
	SchemaType t,          // node or edge
	int id,                // schema id
	const char *name       // schema name
) {
	for(uint32_t i = 0; i < g->n_announcements; i++) {
		Announcement *a = g->announcements + i;
		if(a->opcode == EFFECT_ADD_SCHEMA && a->schema_type == t
				&& a->schema_id == id) {
			return;
		}
	}

	Announcement *a = _new_announcement(g);
	a->opcode      = EFFECT_ADD_SCHEMA;
	a->schema_type = t;
	a->schema_id   = id;
	a->name        = rm_strdup(name);
}

void EffectsV3Grouping_AddAttribute
(
	EffectsV3Grouping *g,  // accumulator
	AttributeID id,        // attribute id
	const char *name       // attribute name
) {
	// RULE 4: once per PAYLOAD, not once per group. v3's ADD_ATTRIBUTE carries
	// no node/relationship discriminator - correctly, since C has a single
	// dictionary - so announcing per entity kind would introduce the same id
	// twice under the same name
	for(uint32_t i = 0; i < g->n_announcements; i++) {
		Announcement *a = g->announcements + i;
		if(a->opcode == EFFECT_ADD_ATTRIBUTE && a->attr_id == id) {
			return;
		}
	}

	Announcement *a = _new_announcement(g);
	a->opcode  = EFFECT_ADD_ATTRIBUTE;
	a->attr_id = id;
	a->name    = rm_strdup(name);
}

uint32_t EffectsV3Grouping_RecordCount
(
	const EffectsV3Grouping *g  // accumulator
) {
	uint32_t n = g->n_announcements;

	// counts what would be EMITTED, so a vacuous group does not appear here
	// either - a caller deciding whether a payload is worth sending must not
	// be told about records that will not be in it
	for(uint32_t i = 0; i < g->n_groups; i++) {
		if(!_vacuous(g->groups + i)) {
			n++;
		}
	}

	return n;
}

void EffectsV3Grouping_Encode
(
	EffectsV3Grouping *g,  // accumulator
	EffectsBytes *out      // sink
) {
	// announcements first: a bulk record carries a bare id, so the replica has
	// to have seen the name before anything references it
	for(uint32_t i = 0; i < g->n_announcements; i++) {
		const Announcement *a = g->announcements + i;
		EffectsV3Record r = {
			.opcode      = a->opcode,
			.schema_type = a->schema_type,
			.schema_id   = a->schema_id,
			.attr_id     = a->attr_id,
			.name        = a->name,
		};
		EffectsV3_EncodeRecord(&r, out);
	}

	// RULE 2: sorted by key. A scan finds groups in arrival order, which is a
	// property of the query rather than of the ids, so it is sorted here
	qsort(g->groups, g->n_groups, sizeof(Group), _cmp_group);

	for(uint32_t i = 0; i < g->n_groups; i++) {
		Group *grp = g->groups + i;

		if(_vacuous(grp)) {
			continue;
		}

		EffectsV3IdList ids = EffectsV3IdListBuilder_ToIdList(grp->ids);
		EffectsV3IdList src = { 0 };
		EffectsV3IdList dst = { 0 };
		if(grp->src != NULL) {
			src = EffectsV3IdListBuilder_ToIdList(grp->src);
			dst = EffectsV3IdListBuilder_ToIdList(grp->dst);
		}

		EffectsV3Record r = {
			.opcode      = grp->opcode,
			.count       = grp->count,
			.labels      = grp->labels,
			.n_labels    = grp->n_labels,
			.relation_id = grp->relation_id,
			.attr_ids    = grp->attr_ids,
			.n_attrs     = grp->n_attrs,
			.ids         = ids,
			.src         = src,
			.dst         = dst,
		};

		// the values were encoded as they arrived, so they are appended as
		// bytes rather than re-encoded from SIValues the group no longer holds
		EffectsV3_EncodeRecordWithRawValues(&r, grp->values, out);

		EffectsV3IdListBuilder_FreeIdList(&ids);
		if(grp->src != NULL) {
			EffectsV3IdListBuilder_FreeIdList(&src);
			EffectsV3IdListBuilder_FreeIdList(&dst);
		}
	}
}

void EffectsV3Grouping_Free
(
	EffectsV3Grouping *g  // accumulator
) {
	if(g == NULL) {
		return;
	}

	for(uint32_t i = 0; i < g->n_groups; i++) {
		Group *grp = g->groups + i;
		EffectsV3IdListBuilder_Free(grp->ids);
		EffectsV3IdListBuilder_Free(grp->src);
		EffectsV3IdListBuilder_Free(grp->dst);
		EffectsBytes_Free(grp->values);
		rm_free(grp->labels);
		rm_free(grp->attr_ids);
	}

	for(uint32_t i = 0; i < g->n_announcements; i++) {
		rm_free(g->announcements[i].name);
	}

	rm_free(g->groups);
	rm_free(g->announcements);
	rm_free(g);
}
