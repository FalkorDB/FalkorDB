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

// one attribute of a staged update, with its value already encoded
typedef struct {
	AttributeID id;

	// THE VALUE, not its encoding.
	//
	// It used to be an offset into a byte arena, because the SIValue belongs
	// to the caller and does not outlive the call - but encoding it early was
	// only one way to make it outlive, and the expensive one. SIValue_Persist
	// clones a volatile value and leaves an owned one alone, which is what
	// Rust's Pending does by holding Value rather than bytes.
	//
	// Serialising once at emit rather than at stage removes the arena, the
	// scratch sink and its wrapper, both copies per attribute, and the dead
	// bytes a superseded value left behind.
	SIValue v;
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

// one slot of the per-entity index; idx1 == 0 means empty
typedef struct {
	uint64_t   id;
	uint32_t   op;
	uint32_t   idx1;   // index into 'updates', plus one
} UpdateSlot;

// mix an (opcode, id) pair into a slot
//
// splitmix64's finaliser: entity ids are dense and sequential, and the low bits
// alone would put a bulk create into one probe chain
static inline uint64_t _slot_hash(uint32_t op, uint64_t id) {
	uint64_t x = id + 0x9E3779B97F4A7C15ULL * (uint64_t)(op + 1);
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
	return x ^ (x >> 31);
}

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

	// (opcode, entity id) -> index into 'updates', plus one
	//
	// OPEN ADDRESSED, and chosen for its TEARDOWN. This was a rax, which fixed
	// the quadratic scan and then became the emitter's single largest cost:
	// 29.6% at 200,000 entities, of which freeing the tree was 14.2% and
	// building it 11.5%, against 3.9% for the lookups it exists for. On a
	// single-attribute update every lookup misses and nothing is ever merged -
	// a radix tree built and destroyed to discover there was no merging to do.
	//
	// A flat table frees in one call and resets with a memset, which is the
	// half a radix tree does worst. It stores an INDEX rather than a pointer
	// because 'updates' is realloc'd, and index+1 so zero means empty.
	UpdateSlot *index;
	uint32_t    index_cap;   // always a power of two
	uint32_t    index_n;     // occupied slots
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
	g->index_cap = 256;   // power of two
	g->index_n   = 0;
	g->index     = rm_calloc(g->index_cap, sizeof(UpdateSlot));


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
// one (attribute id, value) pair
//
// Exists so that sorting the ids CARRIES THE VALUES WITH THEM. A record's rows
// are row-major - entity k, attribute j is values[k * n + j], paired with
// attr_ids[j] - so sorting the id array alone lands every value on the wrong
// attribute, in a payload that is well-formed, passes every length check and
// passes a receiver's ascending check too. Neither engine would complain and
// nothing would surface until someone read the data.
//
// Pairing makes that unspellable rather than a thing to remember.
typedef struct {
	AttributeID id;
	SIValue     v;
} AttrPair;

static int _cmp_attr_pair(const void *a, const void *b) {
	AttributeID x = ((const AttrPair *)a)->id;
	AttributeID y = ((const AttrPair *)b)->id;
	return (x < y) ? -1 : (x > y) ? 1 : 0;
}

// sort an entity's attributes into ascending id order, values following
//
// WHY THE EMITTER SORTS, rather than leaving it to a receiver: attr_ids is half
// the partition key of a batched record. Without a canonical order {a,b} and
// {b,a} are different shapes, so the same logical write produces a different
// record count and different bytes on two engines - and the conformance corpus
// stops being able to arbitrate anything. Sorting at the receiver cannot fix
// that, because the grouping has already happened by then.
//
// It is a same-engine defect too: _group_for compares the id array as given, so
// those two orders land in different groups on C alone today.
static void _sort_attrs
(
	AttrPair *out,               // n pairs
	const AttributeID *attr_ids, // ids, any order
	const SIValue *values,       // one value per id, positionally
	uint16_t n                   // how many
) {
	for(uint16_t i = 0; i < n; i++) {
		out[i].id = attr_ids[i];
		out[i].v  = values[i];
	}
	qsort(out, n, sizeof(AttrPair), _cmp_attr_pair);
}

// drop repeats from an already-sorted label set, returning what remains
//
// The set is a key, and a repeated label makes one set look like two. Sorting
// alone does not make [7,7] and [7] the same shape.
static uint16_t _dedup_labels(LabelID *labels, uint16_t n) {
	if(n < 2) {
		return n;
	}

	uint16_t w = 1;
	for(uint16_t i = 1; i < n; i++) {
		if(labels[i] != labels[w - 1]) {
			labels[w++] = labels[i];
		}
	}
	return w;
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
		n_labels = _dedup_labels(norm, n_labels);
	}

	// ids ascending, values carried along with them
	AttrPair pairs[64];
	AttrPair *ap = (n_attrs <= 64) ? pairs
		: rm_malloc(sizeof(AttrPair) * n_attrs);
	AttributeID ids[64];
	AttributeID *sorted_ids = (n_attrs <= 64) ? ids
		: rm_malloc(sizeof(AttributeID) * n_attrs);

	if(n_attrs > 0) {
		_sort_attrs(ap, attr_ids, values, n_attrs);
		for(uint16_t i = 0; i < n_attrs; i++) {
			sorted_ids[i] = ap[i].id;
		}
	}

	Group *grp = _group_for(g, opcode, norm, n_labels, 0, sorted_ids, n_attrs);

	EffectsV3IdListBuilder_Push(grp->ids, id);
	if(n_attrs > 0) {
		EffectsBuffer *w = EffectsBuffer_Wrap(grp->values);
		for(uint16_t i = 0; i < n_attrs; i++) {
			EffectsBuffer_WriteSIValue(&ap[i].v, w);
		}
		EffectsBuffer_Free(w);
	}
	grp->count++;

	if(ap != pairs) {
		rm_free(ap);
	}
	if(sorted_ids != ids) {
		rm_free(sorted_ids);
	}

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
	// ids ascending, values carried along - same reason as the node path
	AttrPair pairs[64];
	AttrPair *ap = (n_attrs <= 64) ? pairs
		: rm_malloc(sizeof(AttrPair) * n_attrs);
	AttributeID ids[64];
	AttributeID *sorted_ids = (n_attrs <= 64) ? ids
		: rm_malloc(sizeof(AttributeID) * n_attrs);

	if(n_attrs > 0) {
		_sort_attrs(ap, attr_ids, values, n_attrs);
		for(uint16_t i = 0; i < n_attrs; i++) {
			sorted_ids[i] = ap[i].id;
		}
	}

	Group *grp = _group_for(g, opcode, NULL, 0, relation_id, sorted_ids,
			n_attrs);

	EffectsV3IdListBuilder_Push(grp->ids, id);

	// an update recovers its endpoints from the graph, so it carries none
	if(grp->src != NULL) {
		EffectsV3IdListBuilder_Push(grp->src, src);
		EffectsV3IdListBuilder_Push(grp->dst, dst);
	}

	if(n_attrs > 0) {
		EffectsBuffer *w = EffectsBuffer_Wrap(grp->values);
		for(uint16_t i = 0; i < n_attrs; i++) {
			EffectsBuffer_WriteSIValue(&ap[i].v, w);
		}
		EffectsBuffer_Free(w);
	}
	grp->count++;

	if(ap != pairs) {
		rm_free(ap);
	}
	if(sorted_ids != ids) {
		rm_free(sorted_ids);
	}
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

// find the slot for (op, id): the entry if present, else where it would go
static inline UpdateSlot *_index_probe
(
	UpdateSlot *tbl,   // table
	uint32_t cap,      // capacity, a power of two
	uint32_t op,       // opcode
	uint64_t id        // entity id
) {
	uint32_t m = cap - 1;
	uint32_t i = (uint32_t)_slot_hash(op, id) & m;

	// linear probing: the table never fills past 70%, so this terminates
	while(tbl[i].idx1 != 0) {
		if(tbl[i].id == id && tbl[i].op == op) {
			return tbl + i;
		}
		i = (i + 1) & m;
	}
	return tbl + i;
}

// double the table and reinsert
static void _index_grow(EffectsV3Grouping *g) {
	const uint32_t cap = g->index_cap * 2;
	UpdateSlot *tbl = rm_calloc(cap, sizeof(UpdateSlot));

	for(uint32_t i = 0; i < g->index_cap; i++) {
		if(g->index[i].idx1 == 0) {
			continue;
		}
		UpdateSlot *d = _index_probe(tbl, cap, g->index[i].op, g->index[i].id);
		*d = g->index[i];
	}

	rm_free(g->index);
	g->index     = tbl;
	g->index_cap = cap;
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
	UpdateSlot *slot = _index_probe(g->index, g->index_cap, (uint32_t)opcode, id);
	if(slot->idx1 != 0) {
		return g->updates + (slot->idx1 - 1);
	}

	if(g->n_updates == g->cap_updates) {
		g->cap_updates *= 2;
		g->updates = rm_realloc(g->updates,
				g->cap_updates * sizeof(PendingUpdate));
	}

	const uint32_t idx = g->n_updates++;
	PendingUpdate *u = g->updates + idx;
	memset(u, 0, sizeof(*u));

	// grown BEFORE the insert would push past 70%, and the slot re-probed
	// because growing rehashes everything
	if((g->index_n + 1) * 10 >= g->index_cap * 7) {
		_index_grow(g);
		slot = _index_probe(g->index, g->index_cap, (uint32_t)opcode, id);
	}
	slot->id   = id;
	slot->op   = (uint32_t)opcode;
	slot->idx1 = idx + 1;
	g->index_n++;

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
			// the superseded value is FREED rather than orphaned: the arena
			// used to keep it as dead bytes, and a leak here shows in no byte
			// comparison
			SIValue_Free(u->attrs[i].v);
			u->attrs[i].v = value;
			SIValue_Persist(&u->attrs[i].v);
			return;
		}
	}

	if(u->n_attrs == u->cap_attrs) {
		u->cap_attrs *= 2;
		u->attrs = rm_realloc(u->attrs, u->cap_attrs * sizeof(StagedAttr));
	}

	StagedAttr *a = u->attrs + u->n_attrs++;
	a->id = attr_id;

	// PERSISTED, not encoded. The caller's value does not outlive this call, so
	// it is cloned if volatile and taken as is if it already owns its memory
	a->v = value;
	SIValue_Persist(&a->v);
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
	// each staged value is owned here now rather than living in a shared
	// arena, so each is freed
	for(uint32_t k = 0; k < u->n_attrs; k++) {
		SIValue_Free(u->attrs[k].v);
	}
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
		// SERIALISED HERE, once, straight into the group's byte sequence -
		// the only point at which these values become bytes
		EffectsBuffer *w = EffectsBuffer_Wrap(grp->values);
		for(uint32_t k = 0; k < u->n_attrs; k++) {
			EffectsBuffer_WriteSIValue(&u->attrs[k].v, w);
		}
		EffectsBuffer_Free(w);
		grp->count++;

		if(attr_ids != ids) {
			rm_free(attr_ids);
		}

		// the bytes have been copied into the group, so the staging copy goes
		_update_release(u);
	}

	g->n_updates = 0;


	// the indices it holds now point past the end of a zero-length array.
	// Cleared with a memset and the allocation kept - this is the teardown
	// that cost 14.2% as a radix tree
	memset(g->index, 0, g->index_cap * sizeof(UpdateSlot));
	g->index_n = 0;
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
		// each opcode fills its own arm now; an announcement is one or the
		// other, never both, and the record no longer has a shared slot that
		// would let the wrong one be read
		EffectsV3Record r = { .opcode = a->opcode };
		if(a->opcode == EFFECT_ADD_SCHEMA) {
			r.add_schema.schema_type = a->schema_type;
			r.add_schema.schema_id   = a->schema_id;
			r.add_schema.name        = a->name;
		} else {
			r.add_attribute.attr_id = a->attr_id;
			r.add_attribute.name    = a->name;
		}
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

		// FILLED PER OPCODE, because each now owns its arm. A group knows its
		// shape - which halves it carries - but the arm it fills is selected by
		// the opcode, and writing the wrong one is valid C that encodes zeroes.
		EffectsV3Record r = { .opcode = grp->opcode };
		switch(grp->opcode) {
			case EFFECT_UPDATE_NODE:
				r.update_node.count    = grp->count;
				r.update_node.labels   = grp->labels;
				r.update_node.n_labels = grp->n_labels;
				r.update_node.attr_ids = grp->attr_ids;
				r.update_node.n_attrs  = grp->n_attrs;
				r.update_node.ids      = ids;
				break;

			case EFFECT_UPDATE_EDGE:
				r.update_edge.count       = grp->count;
				r.update_edge.relation_id = grp->relation_id;
				r.update_edge.attr_ids    = grp->attr_ids;
				r.update_edge.n_attrs     = grp->n_attrs;
				r.update_edge.ids         = ids;
				break;

			case EFFECT_CREATE_NODE:
				r.create_node.count    = grp->count;
				r.create_node.labels   = grp->labels;
				r.create_node.n_labels = grp->n_labels;
				r.create_node.attr_ids = grp->attr_ids;
				r.create_node.n_attrs  = grp->n_attrs;
				r.create_node.ids      = ids;
				break;

			case EFFECT_CREATE_EDGE:
				r.create_edge.count       = grp->count;
				r.create_edge.relation_id = grp->relation_id;
				r.create_edge.attr_ids    = grp->attr_ids;
				r.create_edge.n_attrs     = grp->n_attrs;
				r.create_edge.ids         = ids;
				r.create_edge.src         = src;
				r.create_edge.dst         = dst;
				break;

			case EFFECT_DELETE_NODE:
				r.delete_node.count    = grp->count;
				r.delete_node.labels   = grp->labels;
				r.delete_node.n_labels = grp->n_labels;
				r.delete_node.ids      = ids;
				break;

			case EFFECT_DELETE_EDGE:
				r.delete_edge.count       = grp->count;
				r.delete_edge.relation_id = grp->relation_id;
				r.delete_edge.ids         = ids;
				r.delete_edge.src         = src;
				r.delete_edge.dst         = dst;
				break;

			case EFFECT_SET_LABELS:
				r.set_labels.count    = grp->count;
				r.set_labels.labels   = grp->labels;
				r.set_labels.n_labels = grp->n_labels;
				r.set_labels.ids      = ids;
				break;

			case EFFECT_REMOVE_LABELS:
				r.remove_labels.count    = grp->count;
				r.remove_labels.labels   = grp->labels;
				r.remove_labels.n_labels = grp->n_labels;
				r.remove_labels.ids      = ids;
				break;

			default:
				ASSERT(false && "not a batchable record");
				break;
		}

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
	rm_free(g->index);
	rm_free(g->updates);

	rm_free(g->groups);
	rm_free(g->announcements);
	rm_free(g);
}
