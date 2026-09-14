/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_group.h"
#include "effects_v3_encode.h"
#include "effects_internal.h"
#include "../util/rmalloc.h"
#include "../../deps/rax/rax.h"

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

// one attribute of a staged update, with its value already encoded
typedef struct {
	AttributeID id;
	// WHERE the encoded SIValue is, not a buffer of its own
	//
	// An offset rather than a pointer because the arena is realloc'd as it
	// grows, and every staged attribute would have to be rewritten otherwise.
	size_t off;
	size_t n;
} StagedAttr;

// an entity's update, accumulating until the query stops producing attributes
typedef struct {
	EffectType opcode;
	uint64_t   id;
	LabelID   *labels;      // owned, ascending
	uint16_t   n_labels;
	RelationID relation_id;
	StagedAttr *attrs;      // owned
	uint32_t    n_attrs;
	uint32_t    cap_attrs;
} PendingUpdate;

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

	PendingUpdate *updates;
	uint32_t n_updates;
	uint32_t cap_updates;

	// every staged value, back to back
	//
	// Was a separate rm_malloc per attribute, on top of a throwaway
	// EffectsBytes and a throwaway EffectsBuffer to encode into - four
	// allocations and four frees for every attribute of every entity, and the
	// same again whenever a query set one twice.
	//
	// Contiguous and grown by realloc, so a staged attribute holds an offset
	// and a length. Reset rather than freed at flush, so a second statement in
	// the same query reuses the allocation.
	//
	// NOT reclaimed on overwrite: setting an attribute twice leaves the first
	// encoding behind as dead bytes. Bounded by the number of stage calls
	// rather than by anything unbounded, and the alternative - a free list, or
	// rewriting later offsets - costs more than the bytes are worth.
	unsigned char *arena;
	size_t arena_len;
	size_t arena_cap;

	// encoded HERE first, then copied into the arena
	//
	// The SIValue codec writes into an EffectsBytes, which is block-chained
	// and so cannot hand out a stable offset - that is why the arena is not
	// simply the sink. Both of these live for the whole accumulator instead of
	// being built and torn down per attribute, which is where the four
	// allocations went.
	EffectsBytes  *scratch;
	EffectsBuffer *scratch_w;

	// (opcode, entity id) -> index into 'updates', plus one
	//
	// The array stays: flush walks it in arrival order, and that order is what
	// the emitted record's rows follow. This is only the index that finds an
	// entity again, which was a linear scan over every entity staged so far.
	//
	// It stores an INDEX rather than a pointer because 'updates' is realloc'd,
	// and index+1 so that the first entry is distinguishable from a miss
	// without leaning on what raxFind returns for a NULL value.
	rax *update_index;
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
	g->cap_updates       = 4;
	g->updates           = rm_calloc(g->cap_updates, sizeof(PendingUpdate));
	g->n_updates         = 0;
	g->update_index      = raxNew();

	g->arena_cap = 4096;
	g->arena_len = 0;
	g->arena     = rm_malloc(g->arena_cap);
	g->scratch   = EffectsBytes_New(256);
	g->scratch_w = EffectsBuffer_Wrap(g->scratch);

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

static int _cmp_staged_attr(const void *a, const void *b) {
	AttributeID x = ((const StagedAttr *)a)->id;
	AttributeID y = ((const StagedAttr *)b)->id;
	return (x < y) ? -1 : (x > y) ? 1 : 0;
}

// encode one value into the arena, returning where it landed
//
// The SIValue belongs to the caller and will not outlive the call, so it has
// to be encoded now rather than retained - that constraint is unchanged. What
// changed is that the scratch sink and its wrapper are reused instead of being
// built and torn down for every attribute.
static size_t _arena_put
(
	EffectsV3Grouping *g,  // accumulator
	SIValue value,         // value to encode
	size_t *n              // [output] how many bytes it took
) {
	// keeps the first block, drops the rest, resets the write offset - so a
	// value larger than one block costs a block here and the next value does
	// not pay for it
	EffectsBytes_Clear(g->scratch);

	EffectsBuffer_WriteSIValue(&value, g->scratch_w);

	const size_t len = EffectsBytes_Len(g->scratch);

	if(g->arena_len + len > g->arena_cap) {
		do {
			g->arena_cap *= 2;
		} while(g->arena_len + len > g->arena_cap);
		g->arena = rm_realloc(g->arena, g->arena_cap);
	}

	EffectsBytes_CopyInto(g->scratch, g->arena + g->arena_len);

	const size_t off = g->arena_len;
	g->arena_len += len;

	*n = len;
	return off;
}

// the index key: opcode then entity id
//
// _update_for matches on both, so both are in the key. The byte order is
// whatever the host uses - this key never leaves the process and is only ever
// compared for equality, so it needs no canonical form.
static inline size_t _update_key
(
	unsigned char *buf,  // at least 12 bytes
	EffectType opcode,   // UPDATE_NODE or UPDATE_EDGE
	uint64_t id          // entity id
) {
	uint32_t op = (uint32_t)opcode;
	memcpy(buf, &op, sizeof(op));
	memcpy(buf + sizeof(op), &id, sizeof(id));
	return sizeof(op) + sizeof(id);
}

// find the staged update for this entity, or open one
//
// WAS A LINEAR SCAN, and it made a single-attribute update O(n^2) over the
// entities in the statement. `MATCH (n:P) SET n.age = n.age + 1` touches each
// entity once, so the scan never hit: it walked everything staged so far,
// found nothing, and appended. Measured at 462,415 instructions per entity at
// n=100,000 against v2's flat 6,588, with 1,456 of 1,471 profile samples on
// the scan line.
//
// The deduplication it performs is real but only multi-attribute writes need
// it, and they were making every single-attribute write pay for it.
static PendingUpdate *_update_for
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // UPDATE_NODE or UPDATE_EDGE
	uint64_t id,            // entity id
	const LabelID *labels,  // normalised labels
	uint16_t n_labels,      // how many
	RelationID relation_id  // relationship type
) {
	unsigned char key[12];
	size_t klen = _update_key(key, opcode, id);

	void *found = raxFind(g->update_index, key, klen);
	if(found != raxNotFound) {
		return g->updates + ((uintptr_t)found - 1);
	}

	if(g->n_updates == g->cap_updates) {
		g->cap_updates *= 2;
		g->updates = rm_realloc(g->updates,
				g->cap_updates * sizeof(PendingUpdate));
	}

	const uint32_t idx = g->n_updates++;
	PendingUpdate *u = g->updates + idx;
	memset(u, 0, sizeof(*u));

	raxInsert(g->update_index, key, klen, (void *)(uintptr_t)(idx + 1), NULL);

	u->opcode      = opcode;
	u->id          = id;
	u->relation_id = relation_id;
	u->n_labels    = n_labels;
	u->cap_attrs   = 4;
	u->attrs       = rm_calloc(u->cap_attrs, sizeof(StagedAttr));

	if(n_labels > 0) {
		u->labels = rm_malloc(sizeof(LabelID) * n_labels);
		memcpy(u->labels, labels, sizeof(LabelID) * n_labels);
	}

	return u;
}

void EffectsV3Grouping_StageUpdate
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // UPDATE_NODE or UPDATE_EDGE
	uint64_t id,            // entity id
	const LabelID *labels,  // labels, any order
	uint16_t n_labels,      // how many
	RelationID relation_id, // relationship type
	AttributeID attr_id,    // the attribute being set
	SIValue value           // its new value
) {
	LabelID sorted[64];
	LabelID *norm = (n_labels <= 64)
		? sorted
		: rm_malloc(sizeof(LabelID) * n_labels);

	if(n_labels > 0) {
		memcpy(norm, labels, sizeof(LabelID) * n_labels);
		qsort(norm, n_labels, sizeof(LabelID), _cmp_label);
	}

	PendingUpdate *u =
		_update_for(g, opcode, id, norm, n_labels, relation_id);

	if(norm != sorted) {
		rm_free(norm);
	}

	// setting the same attribute twice in one query keeps the LAST value: the
	// query's own order decides, and the wire carries one value per attribute
	for(uint32_t i = 0; i < u->n_attrs; i++) {
		if(u->attrs[i].id == attr_id) {
			// the superseded encoding stays in the arena as dead bytes; see
			// the arena's own comment for why reclaiming it is not worth it
			u->attrs[i].off = _arena_put(g, value, &u->attrs[i].n);
			return;
		}
	}

	if(u->n_attrs == u->cap_attrs) {
		u->cap_attrs *= 2;
		u->attrs = rm_realloc(u->attrs, u->cap_attrs * sizeof(StagedAttr));
	}

	StagedAttr *a = u->attrs + u->n_attrs++;
	a->id = attr_id;

	// encoded NOW: the SIValue belongs to the caller and will not outlive this
	a->off = _arena_put(g, value, &a->n);
}

// release everything a staged update owns
//
// Both the flush and the free path go through this, and the flush is the one
// that matters: it sets n_updates to 0, and EffectsV3Grouping_Free frees per
// update by iterating n_updates - so anything still owned at that point is
// unreachable. Every entity in a `SET` leaked its encoded value, its attribute
// array and its label array, once per query.
//
// Idempotent, so freeing an accumulator that was never flushed is still correct.
static void _update_release(PendingUpdate *u) {
	// the encoded values are not freed here - they live in the accumulator's
	// arena, which outlives every individual update and is released once
	rm_free(u->attrs);
	rm_free(u->labels);

	u->attrs   = NULL;
	u->labels  = NULL;
	u->n_attrs = 0;
}

// fold every staged update into its group
//
// deferred to here because an entity's SHAPE is not known until the query stops
// producing attributes for it, and the shape is what selects the group
static void _flush_updates(EffectsV3Grouping *g) {
	for(uint32_t i = 0; i < g->n_updates; i++) {
		PendingUpdate *u = g->updates + i;

		if(u->n_attrs == 0) {
			// still owns the array _update_for allocated for it
			_update_release(u);
			continue;
		}

		// attribute-id order is what makes two entities with the same set land
		// in one group however their attributes happened to arrive
		qsort(u->attrs, u->n_attrs, sizeof(StagedAttr), _cmp_staged_attr);

		AttributeID ids[256];
		AttributeID *attr_ids = (u->n_attrs <= 256)
			? ids
			: rm_malloc(sizeof(AttributeID) * u->n_attrs);

		for(uint32_t k = 0; k < u->n_attrs; k++) {
			attr_ids[k] = u->attrs[k].id;
		}

		Group *grp = _group_for(g, u->opcode, u->labels, u->n_labels,
				u->relation_id, attr_ids, (uint16_t)u->n_attrs);

		EffectsV3IdListBuilder_Push(grp->ids, u->id);
		for(uint32_t k = 0; k < u->n_attrs; k++) {
			EffectsBytes_Write(grp->values, g->arena + u->attrs[k].off,
					u->attrs[k].n);
		}
		grp->count++;

		if(attr_ids != ids) {
			rm_free(attr_ids);
		}

		// the bytes have been copied into the group, so the staging copy goes
		_update_release(u);
	}

	g->n_updates = 0;

	// every offset into the arena belonged to an update that has just been
	// released, so the bytes are unreachable - reset rather than free, so a
	// second statement in the same query reuses the allocation
	g->arena_len = 0;

	// the indices it holds now point past the end of a zero-length array
	raxFree(g->update_index);
	g->update_index = raxNew();
}

size_t EffectsV3Grouping_StagedBytes
(
	const EffectsV3Grouping *g  // accumulator
) {
	ASSERT(g != NULL);

	return g->arena_len;
}

uint32_t EffectsV3Grouping_RecordCount
(
	EffectsV3Grouping *g  // accumulator
) {
	// fold first: a staged update has no group until its shape is complete
	_flush_updates(g);

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
	// staged updates become groups before anything is counted or written
	_flush_updates(g);

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

	// anything still staged - an accumulator freed without being encoded
	for(uint32_t i = 0; i < g->n_updates; i++) {
		_update_release(g->updates + i);
	}
	raxFree(g->update_index);
	rm_free(g->arena);
	EffectsBuffer_Free(g->scratch_w);
	EffectsBytes_Free(g->scratch);
	rm_free(g->updates);

	rm_free(g->groups);
	rm_free(g->announcements);
	rm_free(g);
}
