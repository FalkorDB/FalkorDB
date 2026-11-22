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
//   3  AddSchema / AddAttribute -------------> EffectsV3Record --------> Encode
//      singular records, no grouping              (emitted FIRST)
//
// PATH 1 normalises on the way IN - labels sorted and deduplicated, attribute
// ids sorted with their values carried along.
//
// PATH 2 cannot: an update's shape is the entity's WHOLE attribute set and C's
// write API delivers one attribute at a time, so the entity waits in Staging,
// keyed by (opcode, entity id), until the query stops producing attributes for
// it - last value winning. It normalises at the FLUSH, which is the source of
// most of the asymmetry in this file.
//
// PATH 3 is emitted first because a batched record carries a bare schema or
// attribute id, and the replica must have seen the name before anything
// references it.
//
// THE JOIN is _flush_updates, idempotent, reached from both exits. THE EXITS are
// Encode and RecordCount - the same decision made as a dry run, which is why
// RecordCount cannot be const and consults _has_effect too.
//
// WHERE THE HEADER'S FOUR RULES ARE ENFORCED
//
//   1  one record per (opcode, shape)   _group_for, via _cmp_group
//   2  groups sorted by key             Encode, the qsort - NOT the scan
//   3  label sets normalised ascending  on the way in: AddNode, StageUpdate
//   4  an attribute announced once      AddAttribute, the scan before appending
//
// Rule 2 sits at the exit because _group_for finds groups in ARRIVAL order, a
// property of the query text rather than of the data; the canonical order is
// imposed once at emission. So _cmp_group serves rule 1 and rule 2 both, which
// is why it is a total order and not merely an equality test.

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

		// NOT V3 OPCODES. A CCH statement is staged as CREATE_INDEX or
		// DROP_INDEX carrying INDEX_FLD_CCH - the v3 wire has fourteen
		// opcodes and none of them is CCH - so these two never reach here.
		case EFFECT_CREATE_CCH:
		case EFFECT_DROP_CCH:
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

// one attribute of a staged update, holding the VALUE and not its encoding
typedef struct {
	AttributeID id;

	// OWNED HERE, a clone of whatever the caller passed. Encoding at stage
	// time would need an arena, a scratch sink and a second copy per
	// attribute, and would leave dead bytes behind whenever a later write
	// superseded one - so these are serialised once, at the flush.
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

// the deferred path's scratch, and nothing else
//
// Only UPDATE_NODE and UPDATE_EDGE arrive one attribute at a time; the other six
// batchable opcodes carry their whole shape and go straight into a group.
// Separate from the payload below because its LIFETIME is: it fills during the
// query, drains into groups once, and is dead from then on.
typedef struct {
	PendingUpdate *updates;  // arr

	// (opcode, entity id) -> index into 'updates', plus one
	//
	// Open addressed, so not an arr: power-of-two capacity for mask probing,
	// grown at 70% load. Chosen for its TEARDOWN rather than its lookups - it is
	// built and destroyed once per payload, and on a single-attribute update
	// every lookup misses, so allocating and freeing dominates probing.
	//
	// NOT util/dict.c, and measured rather than assumed: with keys packed into
	// the pointer so dict needs no keyDup or keyDestructor - the friendliest
	// shape it can be given here - dict costs one heap block PER ENTITY against
	// this table's one per payload (a dictEntry each, dict.c:448), and 5x to
	// 20x the CYCLES over the same key stream and the same hash in the
	// 10-to-1,000 entity band a statement touches, narrowing to 2.1x at 100k
	// where this table turns memory-bound. A one-entity payload is too small to
	// call either way.
	//
	// What dict does NOT cost more of is BYTES: below ~100 entities it holds
	// fewer than this table does, because the 256-slot floor is 4 KB whatever
	// the payload turns out to hold. The win here is cycles and block count.
	//
	// An INDEX rather than a pointer because 'updates' is reallocated, and
	// index+1 so zero means empty.
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
	// AS RECORDS, not as a struct of their own: an announcement is already a
	// singular record - ADD_SCHEMA or ADD_ATTRIBUTE, no count and no ids - so a
	// parallel struct bought nothing but a transcription loop at the exit that
	// could fill the wrong arm. The 'name' in whichever arm is live is owned
	EffectsV3Record *announcements;  // arr

	// emitted after them, in key order
	Group *groups;                // arr

	// the group the last lookup found, as an INDEX
	//
	// _group_for is a linear scan and consecutive entities overwhelmingly share
	// a shape - a bulk create has exactly one - so one slot in front of it turns
	// the common case into a single comparison. An index rather than a pointer,
	// because the array is realloc'd when it grows: a cached pointer would
	// dangle and usually still work. UINT32_MAX means nothing memoed.
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
// Such a record is well formed and parses, and applying it changes nothing. The
// rule, without a per-record table: it has no effect if removing an empty block
// leaves it saying nothing about any entity it names. What decides is which
// block is the record's SUBJECT, not whether a block is empty:
//
//   DELETE_NODE with no labels    LEGAL - says these nodes carried no labels
//   CREATE_NODE with no attrs     LEGAL - says these nodes have no properties
//   SET_LABELS with no labels     NO EFFECT - the label set IS the whole payload
//   any record naming no entities NO EFFECT - it names nothing to speak about
//
// Readers TOLERATE these rather than refusing them: DELETE_NODE and CREATE_NODE
// need the zero-length LabelSet path anyway, so refusing buys no parse surface
// back and costs a resync loop against any peer still emitting one.
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
	// CAPACITY 1, NOT 4: an EffectsV3Record is 192 bytes against the 24 of the
	// struct this used to hold, so a capacity of four would cost 768 bytes on
	// EVERY payload - 576 more than this does - to spare a realloc on the rare
	// one that announces anything.
	// Most payloads announce nothing at all - a schema or attribute is new only
	// the first time a query mentions it
	g->announcements = arr_new(EffectsV3Record, 1);
	g->last_group    = UINT32_MAX;

	g->staging.updates   = arr_new(PendingUpdate, 4);
	g->staging.index_cap = 256;   // power of two
	g->staging.index_n   = 0;
	g->staging.index     = rm_calloc(g->staging.index_cap, sizeof(UpdateSlot));

	return g;
}

// find the group matching this shape, or open one
//
// A linear scan rather than a hash, deliberately: the number of distinct shapes
// in a query is small - a label set is parsed rather than computed, so it is
// bounded by the query text - and a scan has no iteration order to get wrong.
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

// one (attribute id, value) pair
//
// Exists so that sorting the ids CARRIES THE VALUES WITH THEM. A record's rows
// are row-major - entity k, attribute j is values[k * n + j] - so sorting the id
// array alone lands every value on the wrong attribute, in a payload that is
// well formed, passes every length check, and passes a receiver's ascending
// check too. Pairing makes that unspellable rather than a thing to remember.
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
// The emitter sorts rather than the receiver because attr_ids is half the
// partition key of a batched record: without a canonical order {a,b} and {b,a}
// are different shapes, so the same logical write produces a different record
// count and different bytes on two engines. A receiver cannot fix that, the
// grouping has already happened. It is a same-engine defect too - _group_for
// compares the id array as given.
//
// REPEATS ARE COLLAPSED, LAST WRITE WINNING, and the surviving count returned. A
// repeated id is an upstream bug - C's own update path already folds
// `SET n.v = 1, n.v = 2` into one effect - but the contract being written to is
// STRICTLY ascending: Rust refuses `w[0] >= w[1]` outright
// (graph/src/effects/v3/apply.rs), so a repeat is a resync there, and on a C
// replica it is worse - the reader takes both pairs, the second write to the id
// is dropped, and the attribute that lost its column is never set at all. No
// error, no log, no resync.
//
// Deliberately NOT an assert: ASSERT aborts, and this is the replication path,
// so on data the encoder can now handle correctly, taking the process down is
// the worse failure. It also makes the collapse untestable.
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

// does this record INTRODUCE an id, rather than reference one?
//
// ADD_SCHEMA and ADD_ATTRIBUTE are what every other record's bare ids refer
// back to, so they are the half of the announcement list that has to be
// written first. Everything else in that list - the index and constraint DDL -
// names ids rather than introducing them.
static bool _introduces_an_id
(
	EffectType opcode  // the record's opcode
) {
	return opcode == EFFECT_ADD_SCHEMA || opcode == EFFECT_ADD_ATTRIBUTE;
}

static EffectsV3Record *_new_announcement(EffectsV3Grouping *g) {
	const EffectsV3Record blank = { 0 };
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
		const EffectsV3Record *a = g->announcements + i;
		if(a->opcode == EFFECT_ADD_SCHEMA && a->add_schema.schema_type == t
				&& a->add_schema.schema_id == id) {
			return;
		}
	}

	EffectsV3Record *a = _new_announcement(g);
	a->opcode                 = EFFECT_ADD_SCHEMA;
	a->add_schema.schema_type = t;
	a->add_schema.schema_id   = id;
	a->add_schema.name        = rm_strdup(name);
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
		const EffectsV3Record *a = g->announcements + i;
		if(a->opcode == EFFECT_ADD_ATTRIBUTE
				&& a->add_attribute.attr_id == id) {
			return;
		}
	}

	EffectsV3Record *a = _new_announcement(g);
	a->opcode                 = EFFECT_ADD_ATTRIBUTE;
	a->add_attribute.attr_id  = id;
	a->add_attribute.name     = rm_strdup(name);
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

// empty the table, keeping its allocation
//
// A memset rather than a free: this runs once per payload, the table is built
// and destroyed whole, and on a single-attribute update every lookup misses -
// so allocating and freeing dominates probing. That teardown cost 14.2% when
// this was a radix tree; see the Staging comment for how it compares to dict
static void _staging_index_clear
(
	Staging *s  // staging area
) {
	memset(s->index, 0, s->index_cap * sizeof(UpdateSlot));
	s->index_n = 0;
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
// The mirror of _group_for: that one is keyed by VALUE, this one by IDENTITY.
//
//   _group_for    (opcode, shape)       one per distinct SHAPE    memo + scan
//   _update_for   (opcode, entity id)   one per ENTITY            hash probe
//
// Which is what decides the lookup. Shapes are few and bounded by the query
// text, so a scan wins on constant factors; ENTITIES are unbounded and
// data-dependent - `MATCH (n:P) SET ...` touches as many as match - so the same
// scan would be quadratic in the size of the match. The memo follows from the
// same split: there is no win in front of an O(1) probe.
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

// take a value the accumulator will own and later free
//
// _update_release frees exactly M_SELF, so the question this answers is which
// values have to be copied for that to be safe:
//
//   M_SELF      COPIED. The caller owns this allocation, and storing it bare
//               meant freeing the caller's memory at the flush. This is the
//               defect; SIValue_Persist does not cover it, because it clones
//               M_VOLATILE and leaves every other kind alone.
//
//               NOT THEORETICAL: a runtime-computed string is M_SELF, and
//               `SET n.s = n.s + '!'` killed the server on a build without
//               this. `SET n.s = 'b'` did not - constant folding makes that
//               M_CONST, and nothing frees an M_CONST, which is the same
//               reason this leaves them alone below.
//   M_VOLATILE  copied, as before: it will not outlive the call.
//   M_CONST     TAKEN AS IT IS, and deliberately. Cloning one would be safe,
//               but an interned string's clone rents from the global string
//               pool and its free returns it - two trips through that mutex
//               per staged attribute, on the write path, for a value that was
//               already safe to borrow. _update_release will not free it.
//   M_NONE      taken as it is; a number, a boolean or a null owns nothing.
static SIValue _take_value(SIValue value) {
	if(value.allocation == M_SELF) {
		return SI_CloneValue(value);
	}

	SIValue v = value;
	SIValue_Persist(&v);   // clones M_VOLATILE, leaves M_CONST and M_NONE
	return v;
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

	// SORTED AND DEDUPLICATED, the same normalisation AddNode applies: sorting
	// alone does not make [7,7] and [7] the same shape, so a staged update
	// would open its own group and write the repeated label into the record's
	// LabelSet - while the identical update filed through AddNode landed in a
	// different group. Rule 3 in effects_v3_group.h is over the label SET.
	if(n_labels > 0) {
		memcpy(norm, labels, sizeof(LabelID) * n_labels);
		qsort(norm, n_labels, sizeof(LabelID), _cmp_label);
		n_labels = _dedup_labels(norm, n_labels);
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
			u->attrs[i].v = _take_value(value);
			return;
		}
	}

	const StagedAttr blank = { 0 };
	arr_append(u->attrs, blank);
	StagedAttr *a = &arr_tail(u->attrs);
	a->id = attr_id;

	a->v = _take_value(value);
}

// release everything a staged update owns
//
// Both the flush and the free path go through this, and the flush is the one
// that matters: it clears the updates arr, which EffectsV3Grouping_Free iterates
// to release per update - so anything still owned at that point is unreachable.
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

		// THE WIRE STATES n_attrs IN TWO BYTES, and a cast that overflows it
		// turns 65,536 into 0: the record would announce no attributes while
		// the loop below still wrote all of their values, and a decoder would
		// read those bytes as the record after this one.
		//
		// Nothing can reach that today - GraphContext stops issuing attribute
		// ids at ATTRIBUTE_ID_ALL (graphcontext.c:1270), and 65,534 and 65,535
		// are reserved, so an entity tops out at 65,534 distinct attributes,
		// two below the field's limit. The guard is here because that ceiling
		// lives in another file and is not what this cast depends on.
		//
		// RELEASE_ASSERT because there is no third option here. _flush_updates
		// returns void, so nothing can be refused upward; skipping the record
		// would drop a write the primary made and diverge the replica with
		// nothing logged; and splitting it in two would emit two records each
		// claiming to be the entity's whole attribute set, which is what the
		// shape means. Stopping is the only outcome that is not silent.
		RELEASE_ASSERT(n_staged <= UINT16_MAX &&
				"an entity with more attributes than the v3 wire can state");

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

	// every slot held an index into the array just emptied
	_staging_index_clear(&g->staging);
}


// build the id-and-name pairs a DDL record carries
//
// BOTH HALVES, because apply checks one against the other: a diverged
// dictionary is then a refusal rather than an index over whatever that id
// happens to mean on the replica. The names are copied - the caller's belong to
// the schema, which outlives neither the payload nor this accumulator reliably
static EffectsV3AttrRef *_attr_refs_new
(
	const AttributeID *ids,  // attribute ids
	const char **names,      // their names
	uint16_t n               // how many
) {
	if(n == 0) {
		return NULL;
	}

	EffectsV3AttrRef *refs = rm_calloc(n, sizeof(EffectsV3AttrRef));
	for(uint16_t i = 0; i < n; i++) {
		refs[i].id   = ids[i];
		refs[i].name = rm_strdup(names[i]);
	}

	return refs;
}

// release them, names included
static void _attr_refs_free
(
	EffectsV3AttrRef *refs,  // pairs, NULL when there were none
	uint16_t n               // how many
) {
	for(uint16_t i = 0; i < n; i++) {
		rm_free(refs[i].name);
	}
	rm_free(refs);
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
	EffectsV3AttrRef *attrs = _attr_refs_new(attr_ids, attr_names, n);

	// THE OPCODE PICKS THE ARM. The two constraint arms carry the same fields in
	// the same order today and differ only in 'status', so filling one through
	// the other would work - by the coincidence the free path refuses to rely on,
	// and for exactly as long.
	//
	// FILLED BEFORE IT IS APPENDED, so the list never holds a record whose arm
	// was not written: the free path reads the arm the opcode names, and a slot
	// that reached it half-filled would be freeing an uninitialised pointer
	EffectsV3Record r = { .opcode = opcode };

	switch(opcode) {
		case EFFECT_CREATE_CONSTRAINT:
			r.create_constraint.constraint_type = constraint_type;
			r.create_constraint.entity_type     = entity_type;
			r.create_constraint.status          = status;
			r.create_constraint.schema_id       = label_id;
			r.create_constraint.name            = rm_strdup(label);
			r.create_constraint.attrs           = attrs;
			r.create_constraint.n_attrs         = n;
			break;

		case EFFECT_DROP_CONSTRAINT:
			// no status: there is nothing to converge on, and the record has
			// no field for one
			r.drop_constraint.constraint_type = constraint_type;
			r.drop_constraint.entity_type     = entity_type;
			r.drop_constraint.schema_id       = label_id;
			r.drop_constraint.name            = rm_strdup(label);
			r.drop_constraint.attrs           = attrs;
			r.drop_constraint.n_attrs         = n;
			break;

		default:
			// nothing owns 'attrs' once this returns
			ASSERT(false && "not a constraint opcode");
			_attr_refs_free(attrs, n);
			return;
	}

	arr_append(g->announcements, r);
}

//------------------------------------------------------------------------------
// index DDL
//------------------------------------------------------------------------------

// convert C's options map into the wire's typed block
//
// THE MAP KEY IS THE PRESENCE BIT. A key present in the map is exactly an option
// the statement stated, so every field here is a lookup and never a default -
// which keeps "the statement did not say" distinguishable from "the statement
// said the default value".
//
// Reaching past the map to the field's stored options breaks that, and not
// subtly: C seeds every field's phonetic with the literal string "no"
// (index_field.h:15), while Rust reads any non-empty phonetic as ENABLED, so an
// unstated phonetic sent as its stored default turns itself on when it crosses
// engines. Language has already diverged a live replica this way once.
static void _options_from_map
(
	EffectsV3IndexOptions *o,  // block to fill
	SIValue options,           // C's options map
	uint32_t field_type        // IndexFieldType, a bit set
) {
	memset(o, 0, sizeof(*o));

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

	// A VECTOR FIELD'S BLOCK CARRIES THE VECTOR HALF; every field type carries
	// the text half, five clear bytes when nothing is stated. Read off the
	// field type rather than stored in the block: the block has no flag of its
	// own that could disagree with the record's, and the reader has only the
	// record's
	if((field_type & INDEX_FLD_VECTOR) == 0) {
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
			o->sim_func     = VecSimMetric_L2;
		} else if(strcasecmp(v.stringval, "cosine") == 0) {
			o->has_sim_func = true;
			o->sim_func     = VecSimMetric_Cosine;
		} else if(strcasecmp(v.stringval, "ip") == 0) {
			o->has_sim_func = true;
			o->sim_func     = VecSimMetric_IP;
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
//
// 'vector' is the two records' shared field type, already compared by the
// caller: whether the vector half exists at all is a property of the RECORD -
// field_type & INDEX_FLD_VECTOR - and the block carries no second copy of it
// that could disagree
static bool _options_eq
(
	const EffectsV3IndexOptions *a,
	const EffectsV3IndexOptions *b,
	bool vector
) {
	if(a->has_language != b->has_language ||
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

	if(!vector) {
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

// the schemas an index statement names
//
// A LIST, because one CCH statement indexes several relationship types at once
// while every other index statement names exactly one
static EffectsV3SchemaRef *_schema_refs_new
(
	const uint32_t *ids,   // schema ids
	const char **names,    // their names, the cross-check
	uint16_t n             // how many
) {
	EffectsV3SchemaRef *refs = rm_calloc(n, sizeof(EffectsV3SchemaRef));

	for(uint16_t i = 0; i < n; i++) {
		refs[i].id   = ids[i];
		refs[i].name = rm_strdup(names[i]);
	}

	return refs;
}

// release them, names included
static void _schema_refs_free
(
	EffectsV3SchemaRef *refs,  // refs, NULL when there were none
	uint16_t n                 // how many
) {
	for(uint16_t i = 0; i < n; i++) {
		rm_free(refs[i].name);
	}
	rm_free(refs);
}

// append one (id, name) to a record's field list
//
// GROWN BY ONE, not doubled: the record type carries a count and no capacity
// beside it, so a capacity would have to live somewhere the record cannot see
// and be kept in step by hand. A statement names a handful of fields, and this
// runs once per field at statement time
static void _append_attr_ref
(
	EffectsV3AttrRef **attrs,  // the list to extend, NULL when empty
	uint16_t *n,               // its length
	AttributeID id,            // the field
	const char *name           // its name
) {
	// a field already named by this record is not added twice - the same
	// attribute cannot be indexed twice by one statement, and a duplicate
	// would inflate the count the record states ahead of its list
	for(uint16_t i = 0; i < *n; i++) {
		if((*attrs)[i].id == id) {
			return;
		}
	}

	*attrs = rm_realloc(*attrs, (*n + 1) * sizeof(EffectsV3AttrRef));

	(*attrs)[*n].id   = id;
	(*attrs)[*n].name = rm_strdup(name);
	(*n)++;
}

// the fields of an index record the fold compares, read off its own arm
//
// PER ARM, because the opcode owns the arm. Create and drop carry these in the
// same order today, so reading one through the other would work - until a field
// is added ahead of one of them, and then it would read the next field along
// and still compile. Returns false for a record that is not an index record
static bool _index_record_key
(
	const EffectsV3Record *r,            // record
	SchemaType *schema_type,             // out: node or edge
	const EffectsV3SchemaRef **schemas,  // out: the schemas it names
	uint16_t *n_schemas,                 // out: how many
	uint32_t *field_type                 // out: IndexFieldType, a bit set
) {
	switch(r->opcode) {
		case EFFECT_CREATE_INDEX:
			*schema_type = r->create_index.schema_type;
			*schemas     = r->create_index.schemas;
			*n_schemas   = r->create_index.n_schemas;
			*field_type  = r->create_index.field_type;
			return true;

		case EFFECT_DROP_INDEX:
			*schema_type = r->drop_index.schema_type;
			*schemas     = r->drop_index.schemas;
			*n_schemas   = r->drop_index.n_schemas;
			*field_type  = r->drop_index.field_type;
			return true;

		default:
			return false;
	}
}

// append a field to whichever index arm this record is
static void _record_append_attr_ref
(
	EffectsV3Record *r,    // record to extend
	AttributeID id,        // the field
	const char *name       // its name
) {
	switch(r->opcode) {
		case EFFECT_CREATE_INDEX:
			_append_attr_ref(&r->create_index.attrs,
					&r->create_index.n_attrs, id, name);
			break;

		case EFFECT_DROP_INDEX:
			_append_attr_ref(&r->drop_index.attrs,
					&r->drop_index.n_attrs, id, name);
			break;

		default:
			ASSERT(false && "not an index record");
			break;
	}
}

// does this record name exactly the one schema 'id'?
//
// The key a field folds into. An ordinary index statement names one schema, so
// this is the single-id comparison it always was; a CCH statement names several
// and nothing ever folds into it, because its writer builds its record whole.
static bool _names_one_schema
(
	const EffectsV3SchemaRef *schemas,  // the record's schemas
	uint16_t n,                         // how many
	int id                              // schema id
) {
	return n == 1 && schemas[0].id == (uint32_t)id;
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
		// a drop carries no options at all - zero bytes, not an empty block,
		// and its arm has no field to put one in
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
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		EffectsV3Record *a = g->announcements + i;

		SchemaType                a_schema_type;
		const EffectsV3SchemaRef *a_schemas;
		uint16_t                  a_n_schemas;
		uint32_t                  a_field_type;
		if(a->opcode != opcode ||
		   !_index_record_key(a, &a_schema_type, &a_schemas, &a_n_schemas,
			   &a_field_type)) {
			continue;
		}

		// A CCH RECORD IS NEVER THE SAME INDEX, even when it names exactly
		// one relationship type and so matches on opcode, schema type and
		// schema id. A path index is its own object: counting it here would
		// set 'index_seen' and strip the language and stopwords off the first
		// ordinary record that follows it in the same payload.
		if(a_schema_type != schema_type
				|| (a_field_type & INDEX_FLD_CCH) != 0
				|| !_names_one_schema(a_schemas, a_n_schemas, schema_id)) {
			continue;
		}

		index_seen = true;

		if(a_field_type != field_type) {
			continue;
		}

		// A DROP IS FOLDED ON THE KEY ALONE. It carries no options, so there
		// is no second half of the key to agree on
		if(opcode == EFFECT_DROP_INDEX) {
			_record_append_attr_ref(a, attr_id, attr_name);
			return;
		}

		if(_options_eq(&a->create_index.options, &o,
				(field_type & INDEX_FLD_VECTOR) != 0)) {
			_record_append_attr_ref(a, attr_id, attr_name);
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

	// FILLED BEFORE IT IS APPENDED, so the list never holds a record whose arm
	// was not written - the free path reads the arm the opcode names
	EffectsV3Record r = { .opcode = opcode };
	const uint32_t id = (uint32_t)schema_id;
	EffectsV3SchemaRef *schemas = _schema_refs_new(&id, &label, 1);

	switch(opcode) {
		case EFFECT_CREATE_INDEX:
			r.create_index.schema_type = schema_type;
			r.create_index.schemas     = schemas;
			r.create_index.n_schemas   = 1;
			r.create_index.field_type  = field_type;
			r.create_index.has_options = true;
			r.create_index.options     = o;
			break;

		case EFFECT_DROP_INDEX:
			r.drop_index.schema_type = schema_type;
			r.drop_index.schemas     = schemas;
			r.drop_index.n_schemas   = 1;
			r.drop_index.field_type  = field_type;
			break;

		default:
			// nothing owns either of these once this returns
			ASSERT(false && "not an index opcode");
			_schema_refs_free(schemas, 1);
			_options_free(&o);
			return;
	}

	_record_append_attr_ref(&r, attr_id, attr_name);

	arr_append(g->announcements, r);
}

void EffectsV3Grouping_AddCCHIndex
(
	EffectsV3Grouping *g,
	EffectType opcode,
	const uint32_t *rel_ids,
	const char **rel_names,
	uint16_t n_rels,
	AttributeID weight_attr,
	const char *weight_name
) {
	ASSERT(g != NULL);
	ASSERT(opcode == EFFECT_CREATE_INDEX || opcode == EFFECT_DROP_INDEX);
	ASSERT(n_rels > 0);

	EffectsV3SchemaRef *schemas = _schema_refs_new(rel_ids, rel_names, n_rels);

	// FILLED BEFORE IT IS APPENDED, so the list never holds a record whose arm
	// was not written - the free path reads the arm the opcode names
	EffectsV3Record r = { .opcode = opcode };

	switch(opcode) {
		case EFFECT_CREATE_INDEX:
			r.create_index.schema_type = SCHEMA_EDGE;  // a path index spans
			r.create_index.schemas     = schemas;      // relationship types
			r.create_index.n_schemas   = n_rels;
			r.create_index.field_type  = INDEX_FLD_CCH;

			// an options block that states nothing: a path index has none to
			// state, and what a create writes is five clear presence bytes.
			// The record was zeroed above, so the block is already that
			r.create_index.has_options = true;
			break;

		case EFFECT_DROP_INDEX:
			// no options arm at all on a drop
			r.drop_index.schema_type = SCHEMA_EDGE;
			r.drop_index.schemas     = schemas;
			r.drop_index.n_schemas   = n_rels;
			r.drop_index.field_type  = INDEX_FLD_CCH;
			break;

		default:
			// nothing owns the list once this returns
			ASSERT(false && "not an index opcode");
			_schema_refs_free(schemas, n_rels);
			return;
	}

	// the weight attribute is the statement's one field
	_record_append_attr_ref(&r, weight_attr, weight_name);

	arr_append(g->announcements, r);
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

// whether this group carries endpoint id lists beside its entity ids
//
// SHAPE_EDGE IS BROADER THAN "HAS ENDPOINTS": an update-edge record names a
// relationship type and the edges it touches, and nothing else - the endpoints
// are on the wire for create and delete only, so _group_for allocates the two
// builders for those opcodes alone. Asking the shape would read two null
// builders on every edge update
static bool _group_has_endpoints
(
	const Group *grp  // group
) {
	return _shape_of(grp->opcode) == SHAPE_EDGE &&
	       grp->shape.edge.src != NULL;
}

// project a group onto the record it will be encoded as
//
// FILLED PER OPCODE, because each owns its arm. A group knows its shape - which
// halves it carries - but the arm it fills is selected by the opcode, and
// writing the wrong one is valid C that encodes zeroes.
static EffectsV3Record _group_to_record
(
	const Group *grp,       // group
	EffectsV3IdList ids,    // its entity ids
	EffectsV3IdList src,    // its source endpoints, empty unless it has them
	EffectsV3IdList dst     // its destination endpoints, likewise
) {
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

	return r;
}

void EffectsV3Grouping_Encode
(
	EffectsV3Grouping *g,  // accumulator
	EffectsBytes *out      // sink
) {
	// staged updates become groups before anything is counted or written
	_flush_updates(g);

	// ANNOUNCEMENTS FIRST, IN TWO PASSES OVER ONE LIST. A bulk or DDL record
	// carries a bare schema or attribute id, so the replica has to have seen
	// the name before anything references it. EffectsV3_Apply states the rule
	// it relies on and does not re-check: records 9 and 10 are normatively
	// ahead of anything referencing the ids they introduce, and nothing on the
	// reading side reorders.
	//
	// TWO PASSES BECAUSE THE DDL RECORDS SHARE THIS LIST AND FOLD BACKWARDS.
	// AddIndexField appends a later field into an EARLIER record, so a
	// statement whose second field introduces a new attribute leaves its
	// CREATE_INDEX sitting ahead of that attribute's own announcement - and
	// _VerifyDDLRefs refuses a DDL record naming an id that does not resolve
	// locally, which costs the whole payload and marks the replica diverged.
	//
	// A STABLE PARTITION, not a sort: every announcement goes before every
	// record that can name one, and arrival order is kept within each half,
	// because two announcements of the same kind have no order between them
	// that means anything and the DDL records do.
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		if(_introduces_an_id(g->announcements[i].opcode)) {
			EffectsV3_EncodeRecord(g->announcements + i, out);
		}
	}

	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		if(!_introduces_an_id(g->announcements[i].opcode)) {
			EffectsV3_EncodeRecord(g->announcements + i, out);
		}
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
		if(_group_has_endpoints(grp)) {
			src = EffectsV3IdListBuilder_ToIdList(grp->shape.edge.src);
			dst = EffectsV3IdListBuilder_ToIdList(grp->shape.edge.dst);
		}

		EffectsV3Record r = _group_to_record(grp, ids, src, dst);

		// the values were encoded as they arrived, so they are appended as
		// bytes rather than re-encoded from SIValues the group no longer holds
		EffectsV3_EncodeRecordWithRawValues(&r, grp->values, out);

		EffectsV3IdListBuilder_FreeIdList(&ids);
		if(_group_has_endpoints(grp)) {
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

	// THE ARM DECIDES WHICH 'name' IS LIVE. Both arms hold the pointer at the
	// same offset today, so freeing through either would work by coincidence -
	// and stop working the moment a field is added ahead of one of them.
	//
	// Spelled as a switch and not an if/else: an else would hand a third
	// announcement opcode to add_attribute's arm, which is the coincidence
	// above relied on rather than guarded against. The default leaks the name
	// instead, because leaking a pointer is recoverable and freeing the wrong
	// one is not
	for(uint32_t i = 0; i < arr_len(g->announcements); i++) {
		EffectsV3Record *a = g->announcements + i;
		switch(a->opcode) {
			case EFFECT_ADD_SCHEMA:
				rm_free(a->add_schema.name);
				break;
			case EFFECT_ADD_ATTRIBUTE:
				rm_free(a->add_attribute.name);
				break;
			case EFFECT_CREATE_CONSTRAINT:
				rm_free(a->create_constraint.name);
				_attr_refs_free(a->create_constraint.attrs,
						a->create_constraint.n_attrs);
				break;
			case EFFECT_DROP_CONSTRAINT:
				rm_free(a->drop_constraint.name);
				_attr_refs_free(a->drop_constraint.attrs,
						a->drop_constraint.n_attrs);
				break;
			case EFFECT_CREATE_INDEX:
				_schema_refs_free(a->create_index.schemas,
						a->create_index.n_schemas);
				_attr_refs_free(a->create_index.attrs,
						a->create_index.n_attrs);
				_options_free(&a->create_index.options);
				break;
			case EFFECT_DROP_INDEX:
				// no options arm at all on a drop
				_schema_refs_free(a->drop_index.schemas,
						a->drop_index.n_schemas);
				_attr_refs_free(a->drop_index.attrs,
						a->drop_index.n_attrs);
				break;
			default:
				ASSERT(false && "announcement opcode has no free arm");
				break;
		}
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
