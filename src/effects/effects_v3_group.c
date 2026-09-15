/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_v3_group.h"
#include "effects_v3_encode.h"
#include "effects_internal.h"
#include "../util/rmalloc.h"
#include "../util/arr.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../index/index_field.h"

#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// THE FLOW: three ways in, one way out
//------------------------------------------------------------------------------
//
//   1  AddNode / AddEdge -------------------------------.
//      six opcodes, whole shape in one call              |
//                                                        |
//                                                        v
//   2  StageUpdate --> PendingUpdate --------------> _group_for --> Group --.
//      two opcodes, one attribute per call    (at the flush)                |
//                                                                           |
//                                                                           v
//   3  AddSchema / AddAttribute -----------------> Announcement --------> Encode
//      singular records, no grouping                (emitted FIRST)
//
// PATH 1, DIRECT. The caller hands over the whole shape at once, so the entity
// goes straight into its group. Normalisation happens on the way IN: labels
// sorted and deduplicated, attribute ids sorted with their values carried along.
//
// PATH 2, DEFERRED. An update's shape is the entity's WHOLE attribute set, and
// C's write API delivers one attribute at a time - so the entity cannot pick a
// group until the query stops producing attributes for it. It waits in Staging,
// found by a hash index on (opcode, entity id), and the LAST value staged for an
// attribute wins. Normalisation happens at the FLUSH instead, which is the one
// structural consequence of the deferral and the source of most of the asymmetry
// in this file.
//
// PATH 3, ANNOUNCEMENTS. Not batched at all: one statement, one record, no
// count. They are emitted before everything else because a batched record
// carries a bare schema or attribute id, and the replica has to have seen the
// name before anything references it.
//
// THE JOIN is _flush_updates, which runs from both exits and is idempotent - a
// caller normally asks for the count and then encodes. After it, Staging is
// empty and every record-to-be is a Group.
//
// THE EXITS are Encode and RecordCount. RecordCount is the same decision made as
// a dry run: flush, then count exactly what Encode would write, which is why it
// cannot be const and why it consults _has_effect too.
//
// WHERE THE HEADER'S FOUR RULES ARE ENFORCED
//
//   1  one record per (opcode, shape)   _group_for, via _cmp_group
//   2  groups sorted by key             Encode, the qsort - NOT the scan
//   3  label sets normalised ascending  on the way in: AddNode, StageUpdate
//   4  an attribute announced once      AddAttribute, the scan before appending
//
// Rule 2 sits at the exit rather than the entrance deliberately. _group_for
// finds groups in ARRIVAL order, which is a property of the query text rather
// than of the data, so the canonical order is imposed once at emission instead
// of being maintained on every insert.
//
// _cmp_group therefore serves both rule 1 and rule 2 - the lookup and the sort -
// which is why it is a total order and not merely an equality test.

// which arm of a Group's shape union an opcode uses
//
// EXHAUSTIVE OVER EVERY EffectType WITH NO default, which is the price of the
// union below: reading the wrong arm is not a wrong answer, it is a label
// pointer read out of a relationship id. A new opcode has to be classified
// here or the compiler says so, and that is the only thing standing between a
// ninth batchable record and a wild free in EffectsV3Grouping_Free.
typedef enum {
	SHAPE_NODE,  // carries a label set
	SHAPE_EDGE,  // carries a relationship type, and maybe endpoints
	SHAPE_NONE,  // not batchable - never reaches a Group at all
} GroupShape;

static GroupShape _shape_of(EffectType opcode) {
	switch(opcode) {
		case EFFECT_UPDATE_NODE:
		case EFFECT_CREATE_NODE:
		case EFFECT_DELETE_NODE:
		case EFFECT_SET_LABELS:
		case EFFECT_REMOVE_LABELS:
			return SHAPE_NODE;

		case EFFECT_UPDATE_EDGE:
		case EFFECT_CREATE_EDGE:
		case EFFECT_DELETE_EDGE:
			return SHAPE_EDGE;

		// singular records: one statement, one record, no count and nothing to
		// batch. They are announcements or DDL and never reach _group_for
		case EFFECT_UNKNOWN:
		case EFFECT_ADD_SCHEMA:
		case EFFECT_ADD_ATTRIBUTE:
		case EFFECT_CREATE_INDEX:
		case EFFECT_DROP_INDEX:
		case EFFECT_CREATE_CONSTRAINT:
		case EFFECT_DROP_CONSTRAINT:
			break;
	}

	// SHAPE_NONE is the fail-SAFE answer, not a fail-fast one: it frees
	// nothing and compares nothing, so a mistake here leaks rather than
	// freeing an integer as a pointer. ASSERT would be silent in release,
	// which is exactly where that distinction matters
	return SHAPE_NONE;
}

// one (opcode, shape) group, with its rows accumulating
//
// The opcode is the DISCRIMINATOR and is compared first everywhere, so by the
// time any arm is read both sides are known to be the same kind. That is what
// makes the union safe here rather than a hazard: there is no path that picks
// an arm without having established the opcode.
typedef struct {
	EffectType opcode;

	// the half of the shape that depends on the entity kind
	union {
		struct {
			LabelID *labels;   // owned, ascending
			uint16_t n_labels;
		} node;

		struct {
			RelationID relation_id;

			// endpoints, on CREATE_EDGE and DELETE_EDGE only - an update
			// recovers them from the graph, so it carries none. Nested here
			// rather than beside the rows because they are unreachable on a
			// node group by construction instead of by convention
			EffectsV3IdListBuilder *src;
			EffectsV3IdListBuilder *dst;
		} edge;
	} shape;

	// NOT in the union: the four record types that carry attributes are two
	// nodes and two edges, so attributes do not partition on the entity kind.
	// They are simply empty on the four that carry none
	AttributeID *attr_ids;   // owned
	uint16_t     n_attrs;

	// the rows
	EffectsV3IdListBuilder *ids;
	EffectsBytes *values;         // count * n_attrs SIValues, row-major
	uint32_t count;               // entities filed here
} Group;

// one attribute of a staged update, with its value already encoded
typedef struct {
	AttributeID id;

	// THE VALUE, not its encoding.
	//
	// The SIValue belongs to the caller and does not outlive the call, so it has
	// to be made to outlive it somehow. SIValue_Persist is the cheap way: it
	// clones a volatile value and leaves an owned one alone. Encoding it here
	// instead would need an arena to hold the bytes, a scratch sink to write
	// them through, and a second copy per attribute - and would leave dead bytes
	// behind whenever a later write superseded one.
	//
	// So it is serialised once, at the flush, straight into its group's buffer.
	// Rust's Pending holds Value for the same reason.
	SIValue v;
} StagedAttr;

// an entity's update, accumulating until the query stops producing attributes
//
// The same shape union as Group, for the same reason and with the same
// discriminator - UPDATE_NODE and UPDATE_EDGE are one opcode on each side of
// _shape_of's partition. It matters a little more here than there: a Group
// exists once per distinct shape, this exists once per ENTITY, so the arm that
// is not live is dead weight on every entity a query updates.
typedef struct {
	EffectType opcode;
	uint64_t   id;

	union {
		struct {
			LabelID *labels;   // owned, ascending
			uint16_t n_labels;
		} node;

		struct {
			RelationID relation_id;
		} edge;
	} shape;

	// owned. The arr carries its own length, so there is no second count beside
	// it to keep in step - StageUpdate appends or replaces, and the flush reads
	// the length back with arr_len
	StagedAttr *attrs;
} PendingUpdate;

// a singular record, kept in arrival order
//
// schema and attribute announcements and the constraint DDL share this: all are
// one statement to one record with no grouping and no count, so they need a
// list rather than a group table.
typedef struct {
	EffectType  opcode;  // ADD_SCHEMA, ADD_ATTRIBUTE, or the constraint DDL
	SchemaType  schema_type;
	int         schema_id;
	AttributeID attr_id;
	char       *name;    // owned

	// constraint DDL only
	uint32_t          constraint_type;
	uint32_t          entity_type;
	uint32_t          status;

	// index DDL only
	//
	// 'field_type' is part of the key rather than a payload field: a statement
	// creates fields of ONE type, and two types in one record would leave apply
	// no way to tell which field wanted which
	uint32_t              field_type;
	EffectsV3IndexOptions options;
	bool                  has_options;

	// both DDL families - index fields and constraint properties are the same
	// shape on the wire, differing only in the width of their count
	EffectsV3AttrRef *attrs_ref;    // owned, and each name owned
	uint16_t          n_attrs_ref;
	uint16_t          cap_attrs_ref;
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

// THE DEFERRED PATH'S SCRATCH, and nothing else.
//
// Six of the eight batchable opcodes go straight into a group when they arrive,
// because the caller hands over the whole shape at once. Only UPDATE_NODE and
// UPDATE_EDGE arrive one attribute at a time, so only they need somewhere to
// wait - and this is it.
//
// Separate from the payload below because its LIFETIME is different: it fills
// during the query, drains into groups once, and is dead from then on. Keeping
// it in the same struct made "the flush is a phase change" a fact about a
// comment rather than about a type, and left scratch alive until free.
typedef struct {
	PendingUpdate *updates;  // arr

	// (opcode, entity id) -> index into 'updates', plus one
	//
	// NOT an arr, and the exception is the point: this is a hash table, so its
	// capacity is a power of two for mask probing and it grows at 70% load
	// rather than when full. An arr's len/cap mean the wrong things here.
	//
	// OPEN ADDRESSED, and chosen for its TEARDOWN rather than its lookups. This
	// table is built and destroyed once per payload, and on a single-attribute
	// update every lookup misses and nothing is ever merged - so the work that
	// dominates is creating and freeing it, not probing it. A flat table frees
	// in one call and resets with a memset; a node-per-entry structure pays for
	// the same lifetime one allocation at a time.
	//
	// It stores an INDEX rather than a pointer because 'updates' is reallocated,
	// and index+1 so zero means empty.
	UpdateSlot *index;
	uint32_t    index_cap;   // always a power of two
	uint32_t    index_n;     // occupied slots
} Staging;

struct EffectsV3Grouping {
	//--------------------------------------------------------------------------
	// the payload being built - this is the answer
	//--------------------------------------------------------------------------

	// emitted FIRST, because a batched record carries a bare schema or
	// attribute id and the replica has to have seen the name
	Announcement *announcements;  // arr

	// emitted after them, in key order
	Group *groups;                // arr

	// the group the last lookup found, as an INDEX
	//
	// _group_for is a linear scan with a memcmp comparator, run once per
	// flushed entity, and consecutive entities overwhelmingly share a shape -
	// a bulk create has exactly one. One slot in front of the scan turns the
	// common case into a single comparison.
	//
	// An index rather than a pointer, because the array is realloc'd when it
	// grows: a cached pointer would dangle into freed memory and usually still
	// work, which is the worst kind. UINT32_MAX means nothing memoed.
	uint32_t    last_group;

	//--------------------------------------------------------------------------
	// scratch for computing it - dead once the flush has run
	//--------------------------------------------------------------------------

	Staging staging;
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

	// the opcodes are equal from here, so the same arm is live on both and
	// only one of these is even a question
	switch(_shape_of(a->opcode)) {
		case SHAPE_NODE:
			c = _cmp_u32(a->shape.node.n_labels, b->shape.node.n_labels);
			if(c != 0) return c;
			for(uint16_t i = 0; i < a->shape.node.n_labels; i++) {
				c = _cmp_u32((uint32_t)a->shape.node.labels[i],
						(uint32_t)b->shape.node.labels[i]);
				if(c != 0) return c;
			}
			break;

		case SHAPE_EDGE:
			c = _cmp_u32((uint32_t)a->shape.edge.relation_id,
					(uint32_t)b->shape.edge.relation_id);
			if(c != 0) return c;
			break;

		case SHAPE_NONE:
			break;
	}

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

// whether a record would actually say something about the entities it names
//
// A record with NO effect must not be emitted: it is well formed, it parses,
// and applying it changes nothing. The rule covers three categories without a
// per-record table:
//
//   A record has no effect if removing an empty block leaves it saying nothing
//   about any entity it names. An empty block that DESCRIBES the entities the
//   record names is information and is legal. A schema announcement is a
//   binding rather than an instruction and is legal regardless of whether
//   anything references it.
//
// So the distinction is which block is the record's SUBJECT, not whether a
// block is empty:
//
//   DELETE_NODE with no labels    LEGAL - says these nodes carried no labels
//   CREATE_NODE with no attrs     LEGAL - says these nodes have no properties
//   SET_LABELS with no labels     NO EFFECT - a label record's whole payload IS
//                                 its label set, so with none it is an
//                                 instruction to do nothing
//   any record naming no entities NO EFFECT - it cannot say anything about any
//
// Readers deliberately TOLERATE these rather than refusing them. Rejecting
// count = 0 at the header removes parse surface, because every block would
// otherwise need a zero-length path both engines agree on; rejecting a
// zero-label record removes nothing, since DELETE_NODE and CREATE_NODE require
// the zero-length LabelSet path anyway. So rejection buys no safety and costs a
// resync loop against any peer still emitting one. Tolerate on read, refuse to
// emit.
static bool _has_effect(const Group *grp) {
	// names no entities, so it says nothing about any
	if(grp->count == 0) {
		return false;
	}

	// a label record's payload is its label set
	if((grp->opcode == EFFECT_SET_LABELS
				|| grp->opcode == EFFECT_REMOVE_LABELS)
			&& grp->shape.node.n_labels == 0) {
		return false;
	}

	return true;
}

//------------------------------------------------------------------------------
// the accumulator
//------------------------------------------------------------------------------

EffectsV3Grouping *EffectsV3Grouping_New(void) {
	EffectsV3Grouping *g = rm_malloc(sizeof(EffectsV3Grouping));

	g->groups        = arr_new(Group, 4);
	g->announcements = arr_new(Announcement, 4);
	g->last_group    = UINT32_MAX;

	g->staging.updates   = arr_new(PendingUpdate, 4);
	g->staging.index_cap = 256;   // power of two
	g->staging.index_n   = 0;
	g->staging.index     = rm_calloc(g->staging.index_cap, sizeof(UpdateSlot));

	return g;
}

// find the group matching this shape, or open one
//
// A linear scan rather than a hash table, deliberately: the number of distinct
// shapes in a query is small - a label set is parsed rather than computed, so
// it is bounded by the query text - and a scan has no iteration order to get
// wrong. The sort before emission is what makes the order normative either way.
//
// See _update_for for why its lookup is the other way round.
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
		.opcode   = opcode,
		.attr_ids = (AttributeID *)attr_ids,
		.n_attrs  = n_attrs,
	};

	if(_shape_of(opcode) == SHAPE_EDGE) {
		probe.shape.edge.relation_id = relation_id;
	} else {
		probe.shape.node.labels   = (LabelID *)labels;
		probe.shape.node.n_labels = n_labels;
	}

	// the shape the previous entity used, checked before the scan. Revalidated
	// with the same comparator rather than trusted, so it is a shortcut and
	// never a second source of truth
	const uint32_t n_groups = arr_len(g->groups);

	if(g->last_group < n_groups &&
	   _cmp_group(&probe, g->groups + g->last_group) == 0) {
		return g->groups + g->last_group;
	}

	for(uint32_t i = 0; i < n_groups; i++) {
		if(_cmp_group(&probe, g->groups + i) == 0) {
			g->last_group = i;
			return g->groups + i;
		}
	}

	const Group blank = { 0 };
	arr_append(g->groups, blank);

	g->last_group = n_groups;
	Group *grp = g->groups + n_groups;

	grp->opcode = opcode;

	if(_shape_of(opcode) == SHAPE_EDGE) {
		grp->shape.edge.relation_id = relation_id;
	} else {
		grp->shape.node.n_labels = n_labels;
	}
	grp->n_attrs     = n_attrs;
	grp->ids         = EffectsV3IdListBuilder_New();
	grp->values      = EffectsBytes_New(1024);

	if(_shape_of(opcode) == SHAPE_NODE && n_labels > 0) {
		grp->shape.node.labels = rm_malloc(sizeof(LabelID) * n_labels);
		memcpy(grp->shape.node.labels, labels, sizeof(LabelID) * n_labels);
	}

	if(n_attrs > 0) {
		grp->attr_ids = rm_malloc(sizeof(AttributeID) * n_attrs);
		memcpy(grp->attr_ids, attr_ids, sizeof(AttributeID) * n_attrs);
	}

	if(opcode == EFFECT_CREATE_EDGE || opcode == EFFECT_DELETE_EDGE) {
		grp->shape.edge.src = EffectsV3IdListBuilder_New();
		grp->shape.edge.dst = EffectsV3IdListBuilder_New();
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
	uint16_t    seq;  // arrival position; the tie-break qsort does not give
} AttrPair;

// ascending by id, and by ARRIVAL for equal ids
//
// qsort is not stable, so without the second key two entries sharing an id
// could come out either way round and the collapse below would keep a
// different value on two runs of the same query. The wire would still be
// well-formed and the two engines would hold different data.
static int _cmp_attr_pair(const void *a, const void *b) {
	const AttrPair *x = (const AttrPair *)a;
	const AttrPair *y = (const AttrPair *)b;
	if(x->id != y->id) {
		return (x->id < y->id) ? -1 : 1;
	}
	return (x->seq < y->seq) ? -1 : (x->seq > y->seq) ? 1 : 0;
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
//
// REPEATS ARE COLLAPSED, LAST WRITE WINNING, and the surviving count returned.
// A repeated id is an upstream bug - C's own update path already folds
// `SET n.v = 1, n.v = 2` into one effect, and ATTRIBUTE_ID_ALL is expanded to
// a diff rather than emitted - but the encoder does not get to rely on that,
// because the contract it is writing to is STRICTLY ascending: Rust refuses
// `w[0] >= w[1]` outright (graph/src/effects/v3/apply.rs), so a repeat is a
// resync there, and measured on a C replica it is worse - the reader takes
// both pairs, the second write to the id is dropped, and the attribute that
// lost its column is never set at all. No error, no log, no resync: the
// replica quietly holds different data. Collapsing here is the encoder
// refusing to put that on the wire.
//
// Deliberately NOT an assert. ASSERT aborts, and this is the replication
// path: on data the encoder can now handle correctly, taking the process down
// is the worse failure. It also makes the collapse untestable.
static uint16_t _sort_attrs
(
	AttrPair *out,               // n pairs
	const AttributeID *attr_ids, // ids, any order
	const SIValue *values,       // one value per id, positionally
	uint16_t n                   // how many
) {
	for(uint16_t i = 0; i < n; i++) {
		out[i].id  = attr_ids[i];
		out[i].v   = values[i];
		out[i].seq = i;
	}
	qsort(out, n, sizeof(AttrPair), _cmp_attr_pair);

	// values are not owned here - they are written straight into the group's
	// buffer by the caller - so a dropped pair needs no free
	uint16_t w = 0;
	for(uint16_t r = 0; r < n; r++) {
		if(w > 0 && out[w - 1].id == out[r].id) {
			out[w - 1] = out[r];  // later write wins
			continue;
		}
		out[w++] = out[r];
	}

	return w;
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
		// the count can SHRINK here: a repeated id is collapsed
		n_attrs = _sort_attrs(ap, attr_ids, values, n_attrs);
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
		// the count can SHRINK here: a repeated id is collapsed
		n_attrs = _sort_attrs(ap, attr_ids, values, n_attrs);
		for(uint16_t i = 0; i < n_attrs; i++) {
			sorted_ids[i] = ap[i].id;
		}
	}

	Group *grp = _group_for(g, opcode, NULL, 0, relation_id, sorted_ids,
			n_attrs);

	EffectsV3IdListBuilder_Push(grp->ids, id);

	// an update recovers its endpoints from the graph, so it carries none
	if(grp->shape.edge.src != NULL) {
		EffectsV3IdListBuilder_Push(grp->shape.edge.src, src);
		EffectsV3IdListBuilder_Push(grp->shape.edge.dst, dst);
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
	const Announcement blank = { 0 };
	arr_append(g->announcements, blank);
	return &arr_tail(g->announcements);
}

void EffectsV3Grouping_AddSchema
(
	EffectsV3Grouping *g,  // accumulator
	SchemaType t,          // node or edge
	int id,                // schema id
	const char *name       // schema name
) {
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
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
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
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

// ids are unique within one staged update - EffectsV3Grouping_StageUpdate
// replaces in place rather than appending a second entry for an attribute - so
// there are no equal keys for an unstable qsort to reorder
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
	const uint32_t cap = g->staging.index_cap * 2;
	UpdateSlot *tbl = rm_calloc(cap, sizeof(UpdateSlot));

	for(uint32_t i = 0; i < g->staging.index_cap; i++) {
		if(g->staging.index[i].idx1 == 0) {
			continue;
		}
		UpdateSlot *d = _index_probe(tbl, cap, g->staging.index[i].op,
			g->staging.index[i].id);
		*d = g->staging.index[i];
	}

	rm_free(g->staging.index);
	g->staging.index     = tbl;
	g->staging.index_cap = cap;
}

// find the staged update for this entity, or open one
//
// THE MIRROR OF _group_for, and the contrast is the clearest way to hold both:
// that one is keyed by VALUE, this one by IDENTITY.
//
//   _group_for    (opcode, shape)       one per distinct SHAPE    memo + scan
//   _update_for   (opcode, entity id)   one per ENTITY            hash probe
//
// Which is what decides the lookup. Distinct shapes are few and bounded by the
// query text, for the reason _group_for's own comment gives, so a scan wins on
// constant factors there. ENTITIES are unbounded and data-dependent - a
// `MATCH (n:P) SET ...` touches as many as match - so the same scan here would
// be quadratic in the size of the match, which is why this one hashes.
//
// The memo follows from the same split. Consecutive entities overwhelmingly
// share a shape, so one slot in front of _group_for's scan pays; there is no
// equivalent win in front of an O(1) probe.
//
// In one line each: _group_for answers "which record does this finished shape
// belong to", this answers "where is this entity's half-built update". Path 2
// asks this one many times and that one once, at the flush.
static PendingUpdate *_update_for
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // UPDATE_NODE or UPDATE_EDGE
	uint64_t id,            // entity id
	const LabelID *labels,  // normalised labels
	uint16_t n_labels,      // how many
	RelationID relation_id  // relationship type
) {
	UpdateSlot *slot = _index_probe(g->staging.index, g->staging.index_cap,
			(uint32_t)opcode, id);
	if(slot->idx1 != 0) {
		return g->staging.updates + (slot->idx1 - 1);
	}

	const uint32_t idx = arr_len(g->staging.updates);
	const PendingUpdate blank = { 0 };
	arr_append(g->staging.updates, blank);
	PendingUpdate *u = g->staging.updates + idx;

	// grown BEFORE the insert would push past 70%, and the slot re-probed
	// because growing rehashes everything
	if((g->staging.index_n + 1) * 10 >= g->staging.index_cap * 7) {
		_index_grow(g);
		slot = _index_probe(g->staging.index, g->staging.index_cap,
				(uint32_t)opcode, id);
	}
	slot->id   = id;
	slot->op   = (uint32_t)opcode;
	slot->idx1 = idx + 1;
	g->staging.index_n++;

	u->opcode    = opcode;
	u->id        = id;
	u->attrs     = arr_new(StagedAttr, 4);

	if(_shape_of(opcode) == SHAPE_EDGE) {
		u->shape.edge.relation_id = relation_id;
	} else {
		u->shape.node.n_labels = n_labels;

		if(n_labels > 0) {
			u->shape.node.labels = rm_malloc(sizeof(LabelID) * n_labels);
			memcpy(u->shape.node.labels, labels, sizeof(LabelID) * n_labels);
		}
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
	for(uint32_t i = 0; i < arr_len(u->attrs); i++) {
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

	const StagedAttr blank = { 0 };
	arr_append(u->attrs, blank);
	StagedAttr *a = &arr_tail(u->attrs);
	a->id = attr_id;

	// PERSISTED, not encoded. The caller's value does not outlive this call, so
	// it is cloned if volatile and taken as is if it already owns its memory
	a->v = value;
	SIValue_Persist(&a->v);
}

// release everything a staged update owns
//
// Both the flush and the free path go through this, and the flush is the one
// that matters: it clears the updates arr, and EffectsV3Grouping_Free releases
// per update by iterating that arr - so anything still owned at that point is
// unreachable. Every entity in a `SET` leaked its encoded value, its attribute
// array and its label array, once per query.
//
// Idempotent, so freeing an accumulator that was never flushed is still correct.
static void _update_release(PendingUpdate *u) {
	// each staged value is owned here now rather than living in a shared
	// arena, so each is freed
	for(uint32_t k = 0; k < arr_len(u->attrs); k++) {
		SIValue_Free(u->attrs[k].v);
	}
	arr_free(u->attrs);
	u->attrs = NULL;

	// only the node arm holds a pointer; freeing unconditionally would hand
	// rm_free a relationship id on every edge update
	if(_shape_of(u->opcode) == SHAPE_NODE) {
		rm_free(u->shape.node.labels);
		u->shape.node.labels = NULL;
	}
}

// fold every staged update into its group
//
// deferred to here because an entity's SHAPE is not known until the query stops
// producing attributes for it, and the shape is what selects the group
static void _flush_updates(EffectsV3Grouping *g) {
	for(uint32_t i = 0; i < arr_len(g->staging.updates); i++) {
		PendingUpdate *u = g->staging.updates + i;

		if(arr_len(u->attrs) == 0) {
			// still owns the array _update_for allocated for it
			_update_release(u);
			continue;
		}

		// attribute-id order is what makes two entities with the same set land
		// in one group however their attributes happened to arrive
		qsort(u->attrs, arr_len(u->attrs), sizeof(StagedAttr),
				_cmp_staged_attr);

		AttributeID ids[256];
		const uint32_t n_staged = arr_len(u->attrs);
		AttributeID *attr_ids = (n_staged <= 256)
			? ids
			: rm_malloc(sizeof(AttributeID) * n_staged);

		for(uint32_t k = 0; k < n_staged; k++) {
			attr_ids[k] = u->attrs[k].id;
		}

		const bool is_edge = (_shape_of(u->opcode) == SHAPE_EDGE);
		Group *grp = _group_for(g, u->opcode,
				is_edge ? NULL : u->shape.node.labels,
				is_edge ? 0    : u->shape.node.n_labels,
				is_edge ? u->shape.edge.relation_id : 0,
				attr_ids, (uint16_t)n_staged);

		EffectsV3IdListBuilder_Push(grp->ids, u->id);
		// SERIALISED HERE, once, straight into the group's byte sequence -
		// the only point at which these values become bytes
		EffectsBuffer *w = EffectsBuffer_Wrap(grp->values);
		for(uint32_t k = 0; k < n_staged; k++) {
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

	arr_clear(g->staging.updates);

	// the indices it holds now point past the end of a zero-length array.
	// Cleared with a memset and the allocation kept - this is the teardown
	// that cost 14.2% as a radix tree
	memset(g->staging.index, 0, g->staging.index_cap * sizeof(UpdateSlot));
	g->staging.index_n = 0;
}


void EffectsV3Grouping_AddConstraint
(
	EffectsV3Grouping *g,         // accumulator
	EffectType opcode,            // CREATE_CONSTRAINT or DROP_CONSTRAINT
	uint32_t constraint_type,     // unique or mandatory
	uint32_t entity_type,         // 1-based
	uint32_t status,              // ConstraintStatus; CREATE only
	int label_id,                 // schema id
	const char *label,            // schema name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attr_names,      // their names
	uint8_t n                     // how many
) {
	Announcement *a = _new_announcement(g);

	a->opcode          = opcode;
	a->constraint_type = constraint_type;
	a->entity_type     = entity_type;
	a->status          = status;
	a->schema_id       = label_id;
	a->name            = rm_strdup(label);
	a->n_attrs_ref     = n;

	if(n > 0) {
		a->attrs_ref = rm_calloc(n, sizeof(EffectsV3AttrRef));
		for(uint8_t i = 0; i < n; i++) {
			a->attrs_ref[i].id   = attr_ids[i];
			a->attrs_ref[i].name = rm_strdup(attr_names[i]);
		}
	}
}

//------------------------------------------------------------------------------
// index DDL
//------------------------------------------------------------------------------

// convert C's options map into the wire's typed block
//
// THE MAP KEY IS THE PRESENCE BIT. A key present in the map is exactly an
// option the statement stated, so every field here is a lookup and never a
// default - which is what keeps "the statement did not say" distinguishable
// from "the statement said the default value".
//
// Reaching past the map to the field's stored options would break that, and
// not subtly: C seeds every field's phonetic with the literal string "no"
// (index_field.h:15, applied at index_field.c:24 and :57), while Rust reads
// any non-empty phonetic as ENABLED. An unstated phonetic sent as its stored
// default therefore turns itself on when it crosses engines. Same reasoning
// for the other four, and language has already diverged a live replica this
// way once - "Can not override index configuration: Language is already set".
static void _options_from_map
(
	EffectsV3IndexOptions *o,  // block to fill
	SIValue options,           // C's options map
	uint32_t field_type        // IndexFieldType, a bit set
) {
	memset(o, 0, sizeof(*o));

	// a vector field's block carries the vector half; every field type carries
	// the text half, five clear bytes when nothing is stated
	o->is_vector = (field_type & INDEX_FLD_VECTOR) != 0;

	if(SI_TYPE(options) != T_MAP) {
		return;
	}

	SIValue v;

	//--------------------------------------------------------------------------
	// the text half
	//--------------------------------------------------------------------------

	if(MAP_GET(options, "language", v) && SI_TYPE(v) == T_STRING) {
		o->has_language = true;
		o->language     = rm_strdup(v.stringval);
	}

	if(MAP_GET(options, "stopwords", v) && SI_TYPE(v) == T_ARRAY) {
		uint32_t n = SIArray_Length(v);
		o->has_stopwords = true;
		o->n_stopwords   = n;
		o->stopwords     = (n > 0) ? rm_calloc(n, sizeof(char *)) : NULL;
		for(uint32_t i = 0; i < n; i++) {
			SIValue w = SIArray_Get(v, i);
			o->stopwords[i] =
				rm_strdup(SI_TYPE(w) == T_STRING ? w.stringval : "");
		}
	}

	// weight is the one numeric C accepts as either an int or a double, since
	// `weight: 1` and `weight: 1.0` are the same statement to a user
	if(MAP_GET(options, "weight", v) && (SI_TYPE(v) & SI_NUMERIC)) {
		o->has_weight = true;
		o->weight     = SI_GET_NUMERIC(v);
	}

	if(MAP_GET(options, "nostem", v) && SI_TYPE(v) == T_BOOL) {
		o->has_nostem = true;
		o->nostem     = (v.longval != 0);
	}

	if(MAP_GET(options, "phonetic", v) && SI_TYPE(v) == T_STRING) {
		o->has_phonetic = true;
		o->phonetic     = rm_strdup(v.stringval);
	}

	if(!o->is_vector) {
		return;
	}

	//--------------------------------------------------------------------------
	// the vector half
	//--------------------------------------------------------------------------

	// dimension has no presence byte - a vector field must have one, and
	// _parseOptions refuses a field without it, so a statement that reached
	// here has stated it
	if(MAP_GET(options, "dimension", v) && SI_TYPE(v) == T_INT64) {
		o->dimension = (uint64_t)v.longval;
	}

	// the wire carries a CODE where C's statement carries a string. The codes
	// are the VecSimMetric values, so this is C's own parser inverted; an
	// unrecognised name is left unstated rather than guessed, since the same
	// parser would have refused the statement outright
	if(MAP_GET(options, "similarityFunction", v) && SI_TYPE(v) == T_STRING) {
		if(strcasecmp(v.stringval, "euclidean") == 0) {
			o->has_sim_func = true;
			o->sim_func     = V3_SIMFUNC_L2;
		} else if(strcasecmp(v.stringval, "cosine") == 0) {
			o->has_sim_func = true;
			o->sim_func     = V3_SIMFUNC_COSINE;
		} else if(strcasecmp(v.stringval, "ip") == 0) {
			o->has_sim_func = true;
			o->sim_func     = V3_SIMFUNC_IP;
		}
	}

	if(MAP_GET(options, "M", v) && SI_TYPE(v) == T_INT64) {
		o->has_m = true;
		o->m     = (uint64_t)v.longval;
	}

	if(MAP_GET(options, "efConstruction", v) && SI_TYPE(v) == T_INT64) {
		o->has_ef_construction = true;
		o->ef_construction     = (uint64_t)v.longval;
	}

	if(MAP_GET(options, "efRuntime", v) && SI_TYPE(v) == T_INT64) {
		o->has_ef_runtime = true;
		o->ef_runtime     = (uint64_t)v.longval;
	}
}

static void _options_free(EffectsV3IndexOptions *o) {
	if(o->language != NULL) {
		rm_free(o->language);
	}
	if(o->phonetic != NULL) {
		rm_free(o->phonetic);
	}
	if(o->stopwords != NULL) {
		for(uint64_t i = 0; i < o->n_stopwords; i++) {
			rm_free(o->stopwords[i]);
		}
		rm_free(o->stopwords);
	}
	memset(o, 0, sizeof(*o));
}

static bool _streq(const char *a, const char *b) {
	if(a == NULL || b == NULL) {
		return a == b;
	}
	return strcmp(a, b) == 0;
}

// do two option blocks state THE SAME THING?
//
// Compared field by field rather than with memcmp: the struct holds pointers
// and padding, so two blocks that say the same thing rarely have the same
// bytes. Both the flag and the value must agree - a stated weight of 1.0 and
// an unstated weight are different statements even though the value slot of
// the second is also 1.0 by memset.
static bool _options_eq
(
	const EffectsV3IndexOptions *a,
	const EffectsV3IndexOptions *b
) {
	if(a->is_vector    != b->is_vector    ||
	   a->has_language != b->has_language ||
	   a->has_weight   != b->has_weight   ||
	   a->has_nostem   != b->has_nostem   ||
	   a->has_phonetic != b->has_phonetic ||
	   a->has_stopwords != b->has_stopwords) {
		return false;
	}

	if(a->has_language && !_streq(a->language, b->language))  return false;
	if(a->has_phonetic && !_streq(a->phonetic, b->phonetic))  return false;
	if(a->has_nostem   && a->nostem != b->nostem)             return false;

	// compared as BITS, not as doubles: this is deciding whether two records
	// carry the same bytes, and == would fuse a weight of -0.0 with 0.0 while
	// the wire keeps them apart
	if(a->has_weight) {
		uint64_t x, y;
		memcpy(&x, &a->weight, sizeof(x));
		memcpy(&y, &b->weight, sizeof(y));
		if(x != y) return false;
	}

	if(a->has_stopwords) {
		if(a->n_stopwords != b->n_stopwords) return false;
		for(uint64_t i = 0; i < a->n_stopwords; i++) {
			if(!_streq(a->stopwords[i], b->stopwords[i])) return false;
		}
	}

	if(!a->is_vector) {
		return true;
	}

	return a->dimension           == b->dimension           &&
	       a->has_m               == b->has_m               &&
	       a->has_ef_construction == b->has_ef_construction &&
	       a->has_ef_runtime      == b->has_ef_runtime      &&
	       a->has_sim_func        == b->has_sim_func        &&
	       (!a->has_m               || a->m               == b->m)               &&
	       (!a->has_ef_construction || a->ef_construction == b->ef_construction) &&
	       (!a->has_ef_runtime      || a->ef_runtime      == b->ef_runtime)      &&
	       (!a->has_sim_func        || a->sim_func        == b->sim_func);
}

// append one (id, name) to an announcement's field list
static void _append_attr_ref
(
	Announcement *a,       // announcement to extend
	AttributeID id,        // the field
	const char *name       // its name
) {
	// a field already named by this record is not added twice - the same
	// attribute cannot be indexed twice by one statement, and a duplicate
	// would inflate the count the record states ahead of its list
	for(uint16_t i = 0; i < a->n_attrs_ref; i++) {
		if(a->attrs_ref[i].id == id) {
			return;
		}
	}

	if(a->n_attrs_ref == a->cap_attrs_ref) {
		a->cap_attrs_ref = (a->cap_attrs_ref == 0) ? 4 : a->cap_attrs_ref * 2;
		a->attrs_ref = rm_realloc(a->attrs_ref,
				a->cap_attrs_ref * sizeof(EffectsV3AttrRef));
	}

	a->attrs_ref[a->n_attrs_ref].id   = id;
	a->attrs_ref[a->n_attrs_ref].name = rm_strdup(name);
	a->n_attrs_ref++;
}

void EffectsV3Grouping_AddIndexField
(
	EffectsV3Grouping *g,   // accumulator
	EffectType opcode,      // CREATE_INDEX or DROP_INDEX
	SchemaType schema_type, // node or edge
	int schema_id,          // schema id
	const char *label,      // schema name
	uint32_t field_type,    // IndexFieldType, a bit set
	AttributeID attr_id,    // the field
	const char *attr_name,  // its name
	SIValue options         // C's options map; ignored for a drop
) {
	EffectsV3IndexOptions o;
	if(opcode == EFFECT_CREATE_INDEX) {
		_options_from_map(&o, options, field_type);
	} else {
		// a drop carries no options at all - zero bytes, not an empty block
		memset(&o, 0, sizeof(o));
	}

	//--------------------------------------------------------------------------
	// fold into an existing record for this statement, if there is one
	//--------------------------------------------------------------------------

	// scanned rather than matched against the most recent announcement: a
	// field's own ADD_SCHEMA and ADD_ATTRIBUTE announcements are appended
	// between two index effects, so the record this belongs to is not the
	// last one
	bool index_seen = false;  // any earlier record naming the same index
	for(uint32_t i = 0; i < g->n_announcements; i++) {
		Announcement *a = g->announcements + i;

		if(a->opcode != opcode || a->schema_type != schema_type
				|| a->schema_id != schema_id) {
			continue;
		}

		index_seen = true;

		if(a->field_type == field_type && _options_eq(&a->options, &o)) {
			_append_attr_ref(a, attr_id, attr_name);
			_options_free(&o);
			return;
		}
	}

	//--------------------------------------------------------------------------
	// a new record
	//--------------------------------------------------------------------------

	// INDEX-LEVEL OPTIONS ARE STATED ONCE PER INDEX PER PAYLOAD. language and
	// stopwords configure the index, not the field, and their setters are
	// asymmetric: Index_SetLanguage (index.c:852) objects only when the
	// language DIFFERS, but Index_SetStopwords (index.c:871) objects when
	// stopwords are set AT ALL. Repeating them on a second record for the same
	// index would apply cleanly for language and fail for stopwords - and no
	// single-field statement can show it.
	if(index_seen) {
		if(o.has_language) {
			rm_free(o.language);
			o.language     = NULL;
			o.has_language = false;
		}
		if(o.has_stopwords) {
			for(uint64_t i = 0; i < o.n_stopwords; i++) {
				rm_free(o.stopwords[i]);
			}
			rm_free(o.stopwords);
			o.stopwords     = NULL;
			o.n_stopwords   = 0;
			o.has_stopwords = false;
		}
	}

	Announcement *a = _new_announcement(g);

	a->opcode      = opcode;
	a->schema_type = schema_type;
	a->schema_id   = schema_id;
	a->name        = rm_strdup(label);
	a->field_type  = field_type;
	a->options     = o;
	a->has_options = (opcode == EFFECT_CREATE_INDEX);

	_append_attr_ref(a, attr_id, attr_name);
}

uint32_t EffectsV3Grouping_RecordCount
(
	EffectsV3Grouping *g  // accumulator
) {
	// fold first: a staged update has no group until its shape is complete
	_flush_updates(g);

	uint32_t n = arr_len(g->announcements);

	// counts what would be EMITTED, so a group with no effect is not counted
	// either - a caller deciding whether a payload is worth sending must not
	// be told about records that will not be in it
	for(uint32_t i = 0; i < arr_len(g->groups); i++) {
		if(_has_effect(g->groups + i)) {
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
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		const Announcement *a = g->announcements + i;
		// each opcode fills its own arm; an announcement is exactly one of
		// these, and the record no longer has a shared slot that would let the
		// wrong one be read
		EffectsV3Record r = { .opcode = a->opcode };
		switch(a->opcode) {
			case EFFECT_ADD_SCHEMA:
				r.add_schema.schema_type = a->schema_type;
				r.add_schema.schema_id   = a->schema_id;
				r.add_schema.name        = a->name;
				break;

			case EFFECT_ADD_ATTRIBUTE:
				r.add_attribute.attr_id = a->attr_id;
				r.add_attribute.name    = a->name;
				break;

			case EFFECT_CREATE_CONSTRAINT:
				r.create_constraint.constraint_type = a->constraint_type;
				r.create_constraint.entity_type     = a->entity_type;
				r.create_constraint.status          = a->status;
				r.create_constraint.schema_id       = a->schema_id;
				r.create_constraint.name            = a->name;
				r.create_constraint.attrs           = a->attrs_ref;
				r.create_constraint.n_attrs         = a->n_attrs_ref;
				break;

			case EFFECT_DROP_CONSTRAINT:
				// no status: there is nothing to converge on, and the type has
				// no field for one
				r.drop_constraint.constraint_type = a->constraint_type;
				r.drop_constraint.entity_type     = a->entity_type;
				r.drop_constraint.schema_id       = a->schema_id;
				r.drop_constraint.name            = a->name;
				r.drop_constraint.attrs           = a->attrs_ref;
				r.drop_constraint.n_attrs         = a->n_attrs_ref;
				break;

			case EFFECT_CREATE_INDEX:
				r.create_index.schema_type = a->schema_type;
				r.create_index.schema_id   = a->schema_id;
				r.create_index.name        = a->name;
				r.create_index.field_type  = a->field_type;
				r.create_index.attrs       = a->attrs_ref;
				r.create_index.n_attrs     = a->n_attrs_ref;
				r.create_index.has_options = true;
				r.create_index.options     = a->options;
				break;

			case EFFECT_DROP_INDEX:
				// no options field at all on this arm
				r.drop_index.schema_type = a->schema_type;
				r.drop_index.schema_id   = a->schema_id;
				r.drop_index.name        = a->name;
				r.drop_index.field_type  = a->field_type;
				r.drop_index.attrs       = a->attrs_ref;
				r.drop_index.n_attrs     = a->n_attrs_ref;
				break;

			default:
				ASSERT(false && "not a singular record");
				break;
		}
		EffectsV3_EncodeRecord(&r, out);
	}

	// RULE 2: sorted by key. A scan finds groups in arrival order, which is a
	// property of the query rather than of the ids, so it is sorted here
	qsort(g->groups, arr_len(g->groups), sizeof(Group), _cmp_group);

	for(uint32_t i = 0; i < arr_len(g->groups); i++) {
		Group *grp = g->groups + i;

		if(!_has_effect(grp)) {
			continue;
		}

		EffectsV3IdList ids = EffectsV3IdListBuilder_ToIdList(grp->ids);
		EffectsV3IdList src = { 0 };
		EffectsV3IdList dst = { 0 };
		if(_shape_of(grp->opcode) == SHAPE_EDGE &&
		   grp->shape.edge.src != NULL) {
			src = EffectsV3IdListBuilder_ToIdList(grp->shape.edge.src);
			dst = EffectsV3IdListBuilder_ToIdList(grp->shape.edge.dst);
		}

		// FILLED PER OPCODE, because each now owns its arm. A group knows its
		// shape - which halves it carries - but the arm it fills is selected by
		// the opcode, and writing the wrong one is valid C that encodes zeroes.
		EffectsV3Record r = { .opcode = grp->opcode };
		switch(grp->opcode) {
			case EFFECT_UPDATE_NODE:
				r.update_node.count    = grp->count;
				r.update_node.labels   = grp->shape.node.labels;
				r.update_node.n_labels = grp->shape.node.n_labels;
				r.update_node.attr_ids = grp->attr_ids;
				r.update_node.n_attrs  = grp->n_attrs;
				r.update_node.ids      = ids;
				break;

			case EFFECT_UPDATE_EDGE:
				r.update_edge.count       = grp->count;
				r.update_edge.relation_id = grp->shape.edge.relation_id;
				r.update_edge.attr_ids    = grp->attr_ids;
				r.update_edge.n_attrs     = grp->n_attrs;
				r.update_edge.ids         = ids;
				break;

			case EFFECT_CREATE_NODE:
				r.create_node.count    = grp->count;
				r.create_node.labels   = grp->shape.node.labels;
				r.create_node.n_labels = grp->shape.node.n_labels;
				r.create_node.attr_ids = grp->attr_ids;
				r.create_node.n_attrs  = grp->n_attrs;
				r.create_node.ids      = ids;
				break;

			case EFFECT_CREATE_EDGE:
				r.create_edge.count       = grp->count;
				r.create_edge.relation_id = grp->shape.edge.relation_id;
				r.create_edge.attr_ids    = grp->attr_ids;
				r.create_edge.n_attrs     = grp->n_attrs;
				r.create_edge.ids         = ids;
				r.create_edge.src         = src;
				r.create_edge.dst         = dst;
				break;

			case EFFECT_DELETE_NODE:
				r.delete_node.count    = grp->count;
				r.delete_node.labels   = grp->shape.node.labels;
				r.delete_node.n_labels = grp->shape.node.n_labels;
				r.delete_node.ids      = ids;
				break;

			case EFFECT_DELETE_EDGE:
				r.delete_edge.count       = grp->count;
				r.delete_edge.relation_id = grp->shape.edge.relation_id;
				r.delete_edge.ids         = ids;
				r.delete_edge.src         = src;
				r.delete_edge.dst         = dst;
				break;

			case EFFECT_SET_LABELS:
				r.set_labels.count    = grp->count;
				r.set_labels.labels   = grp->shape.node.labels;
				r.set_labels.n_labels = grp->shape.node.n_labels;
				r.set_labels.ids      = ids;
				break;

			case EFFECT_REMOVE_LABELS:
				r.remove_labels.count    = grp->count;
				r.remove_labels.labels   = grp->shape.node.labels;
				r.remove_labels.n_labels = grp->shape.node.n_labels;
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
		if(_shape_of(grp->opcode) == SHAPE_EDGE &&
		   grp->shape.edge.src != NULL) {
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

	for(uint32_t i = 0; i < arr_len(g->groups); i++) {
		Group *grp = g->groups + i;
		EffectsV3IdListBuilder_Free(grp->ids);
		EffectsBytes_Free(grp->values);
		rm_free(grp->attr_ids);

		// THE ARM DECIDES WHAT IS A POINTER. Freeing unconditionally here
		// would hand rm_free a relationship id on every edge group
		switch(_shape_of(grp->opcode)) {
			case SHAPE_NODE:
				rm_free(grp->shape.node.labels);
				break;
			case SHAPE_EDGE:
				EffectsV3IdListBuilder_Free(grp->shape.edge.src);
				EffectsV3IdListBuilder_Free(grp->shape.edge.dst);
				break;
			case SHAPE_NONE:
				break;
		}
	}

	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		Announcement *a = g->announcements + i;
		rm_free(a->name);
		for(uint16_t k = 0; k < a->n_attrs_ref; k++) {
			rm_free(a->attrs_ref[k].name);
		}
		rm_free(a->attrs_ref);
		_options_free(&a->options);
	}

	// anything still staged - an accumulator freed without being encoded
	for(uint32_t i = 0; i < arr_len(g->staging.updates); i++) {
		_update_release(g->staging.updates + i);
	}

	// arr_free, not rm_free: an arr's pointer is the DATA, and its allocation
	// begins one header earlier. Handing rm_free the data pointer frees an
	// address the allocator never returned
	arr_free(g->staging.updates);
	rm_free(g->staging.index);

	arr_free(g->groups);
	arr_free(g->announcements);
	rm_free(g);
}
