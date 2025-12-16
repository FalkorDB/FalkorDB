/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "../query_ctx.h"
#include "../datatypes/vector.h"

// determine block available space 
#define BLOCK_AVAILABLE_SPACE(b) (b->cap - BLOCK_USED_SPACE(b))

// determine how many bytes been written to buffer
#define BLOCK_USED_SPACE(b) (b->offset - b->buffer)

// linked list of EffectsBufferblocks
struct EffectsBufferBlock {
	size_t cap;                       // block capacity
	unsigned char *offset;            // buffer offset
	struct EffectsBufferBlock *next;  // pointer to next buffer
	unsigned char buffer[];           // buffer
};

// effects buffer is a linked-list of buffers
struct _EffectsBuffer {
	size_t block_size;                   // block size
	struct EffectsBufferBlock *head;     // first block
	struct EffectsBufferBlock *current;  // current block
	uint64_t n;                          // number of effects in buffer
};

// forward declarations

// write array to effects buffer
static void EffectsBuffer_WriteSIArray
(
	const SIValue *arr,  // array
	EffectsBuffer *buff  // effect buffer
);

// write vector to effects buffer
static void EffectsBuffer_WriteSIVector
(
	const SIValue *v,    // vector
	EffectsBuffer *buff  // effect buffer
);

// create a new effects-buffer block
static struct EffectsBufferBlock *EffectsBufferBlock_New
(
	size_t n  // size of block
) {
	size_t _n = sizeof(struct EffectsBufferBlock) + n;
	struct EffectsBufferBlock *b = rm_malloc(_n);

	b->cap    = n;
	b->next   = NULL;
	b->offset = b->buffer;

	return b;
}

// add a new block to effects-buffer
static void EffectsBuffer_AddBlock
(
	EffectsBuffer *eb  // effects-buffer
) {
	// create a new block and link
	struct EffectsBufferBlock *b = EffectsBufferBlock_New(eb->block_size);
	eb->current->next = b;
	eb->current       = b;
}

// write n bytes from ptr into block
// returns actual number of bytes written
// if buffer isn't large enough only a portion of the bytes will be written
static size_t EffectsBufferBlock_WriteBytes
(
	const unsigned char *ptr,     // data to write
	size_t n,                     // number of bytes to write
	struct EffectsBufferBlock *b  // block to write to
) {
	// validations
	ASSERT(n   > 0);
	ASSERT(b   != NULL);
	ASSERT(ptr != NULL);

	// determine number of bytes we can write
	n = MIN(n, BLOCK_AVAILABLE_SPACE(b));

	// write n bytes to buffer
	memcpy(b->offset, ptr, n);

	// update offset
	b->offset += n;

	return n;
}

// write n bytes from ptr into effects-buffer
static void EffectsBuffer_WriteBytes
(
	const void *ptr,   // data to write
	size_t n,          // number of bytes to write
	EffectsBuffer *eb  // effects-buffer
) {
	ASSERT (n   > 0) ;
	ASSERT (eb  != NULL) ;
	ASSERT (ptr != NULL) ;

	while (n > 0) {
		struct EffectsBufferBlock *b = eb->current ;
		size_t written = EffectsBufferBlock_WriteBytes (ptr, n, b) ;

		// advance ptr
		ptr += written ;

		if (written == 0) {
			// no bytes written block is full, create a new block
			EffectsBuffer_AddBlock (eb) ;
		}

		// update remaining bytes to write
		n -= written ;
	}
}

static void EffectsBuffer_WriteString
(
	const char *str,
	EffectsBuffer *eb
) {
	ASSERT(eb  != NULL);
	ASSERT(str != NULL);

	size_t l = strlen(str) + 1;
	EffectsBuffer_WriteBytes(&l, sizeof(size_t), eb);
	EffectsBuffer_WriteBytes(str, l, eb);
}

// writes a binary representation of v into Effect-Buffer
static void EffectsBuffer_WriteSIValue
(
	const SIValue *v,
	EffectsBuffer *buff
) {
	ASSERT(v != NULL);
	ASSERT(buff != NULL);

	// format:
	//    type
	//    value
	bool b;
	size_t len = 0;

	// set type to intern string incase the allocation type is intern
	// otherwise use v's original type
	SIType t = v->type;

	// write type
	EffectsBuffer_WriteBytes(&t, sizeof(SIType), buff);

	// write value
	switch(t) {
		case T_POINT:
			// write value to stream
			EffectsBuffer_WriteBytes (&v->point, sizeof (Point), buff) ;
			break ;

		case T_ARRAY:
			// write array to stream
			EffectsBuffer_WriteSIArray (v, buff) ;
			break ;

		case T_STRING:
		case T_INTERN_STRING:
			EffectsBuffer_WriteString (v->stringval, buff) ;
			break ;

		case T_BOOL:
			// write bool to stream
			b = SIValue_IsTrue (*v) ;
			EffectsBuffer_WriteBytes (&b, sizeof (bool), buff) ;
			break ;

		case T_INT64:
			// write int to stream
			EffectsBuffer_WriteBytes (&v->longval, sizeof (v->longval), buff) ;
			break ;

		case T_DOUBLE:
			// write double to stream
			EffectsBuffer_WriteBytes (&v->doubleval, sizeof (v->doubleval), buff) ;
			break ;

		case T_TIME:
		case T_DATE:
		case T_DATETIME:
		case T_DURATION:
			// write temporal time_t
			EffectsBuffer_WriteBytes (&v->datetimeval, sizeof (v->datetimeval), buff) ;
			break ;

		case T_NULL:
			// no additional data is required to represent NULL
			break ;

		case T_VECTOR_F32:
			EffectsBuffer_WriteSIVector (v, buff) ;
			break ;

		default:
			assert (false && "unknown SIValue type") ;
	}
}

// writes a binary representation of arr into Effect-Buffer
static void EffectsBuffer_WriteSIArray
(
	const SIValue *arr,  // array
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	SIValue *elements = arr->array;
	uint32_t len = array_len(elements);

	// write number of elements
	EffectsBuffer_WriteBytes(&len, sizeof(uint32_t), buff);

	// write each element
	for (uint32_t i = 0; i < len; i++) {
		EffectsBuffer_WriteSIValue(elements + i, buff);
	}
}

// write vector to effects buffer
static void EffectsBuffer_WriteSIVector
(
	const SIValue *v,    // vector
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	// write vector dimension
	uint32_t dim = SIVector_Dim(*v);
	EffectsBuffer_WriteBytes(&dim, sizeof(uint32_t), buff);

	// write vector elements
	void *elements   = SIVector_Elements(*v);
	size_t elem_size = sizeof(float);
	size_t n = dim * elem_size;

	if(n > 0) {
		EffectsBuffer_WriteBytes(elements, n, buff);
	}
}

// dump attributes to stream
static void EffectsBuffer_WriteAttributeSet
(
	const AttributeSet attrs,  // attribute set to write to stream
	EffectsBuffer *buff
) {
	//--------------------------------------------------------------------------
	// write attribute count
	//--------------------------------------------------------------------------

	ushort attr_count = AttributeSet_Count(attrs);
	EffectsBuffer_WriteBytes(&attr_count, sizeof(attr_count), buff);

	//--------------------------------------------------------------------------
	// write attributes
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < attr_count; i++) {
		// get current attribute name and value
		SIValue attr ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (attrs, i, &attr_id, &attr) ;

		// write attribute ID
		EffectsBuffer_WriteBytes (&attr_id, sizeof (AttributeID), buff) ;

		// write attribute value
		EffectsBuffer_WriteSIValue (&attr, buff) ;

		// free attribute
		SIValue_Free (attr) ;
	}
}

static inline void EffectsBuffer_IncEffectCount
(
	EffectsBuffer *buff
) {
	ASSERT(buff != NULL);
	
	buff->n++;
}

static inline void EffectsBufferBlock_Free
(
	struct EffectsBufferBlock *b
) {
	ASSERT(b != NULL);
	rm_free(b);
}

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New
(
	void
) {
	size_t n = 62500;  // initial size of buffer
	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	struct EffectsBufferBlock *b = EffectsBufferBlock_New(n);

	eb->n          = 0;
	eb->head       = b;
	eb->current    = b;
	eb->block_size = n;

	// write effects version to newly created buffer
	uint8_t v = EFFECTS_VERSION;
	EffectsBuffer_WriteBytes(&v, sizeof(v), eb);

	return eb;
}

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);

	// free all blocks except the first one
	struct EffectsBufferBlock *b = buff->head->next;
	while(b != NULL) {
		struct EffectsBufferBlock *next = b->next;
		EffectsBufferBlock_Free(b);
		b = next;
	}

	// clear first block
	buff->n = 0;
	buff->current = buff->head;

	// write effects version
	uint8_t v = EFFECTS_VERSION;
	EffectsBuffer_WriteBytes(&v, sizeof(v), buff);
}

// returns number of effects in buffer
uint64_t EffectsBuffer_Length
(
	const EffectsBuffer *buff  // effects-buffer
) {
	ASSERT(buff != NULL);
	
	return buff->n;
}

// get a copy of effects-buffer internal buffer
unsigned char *EffectsBuffer_Buffer
(
	const EffectsBuffer *eb,  // effects-buffer
	size_t *n                 // size of returned buffer
) {
	ASSERT(eb != NULL);

	//--------------------------------------------------------------------------
	// determine required buffer size
	//--------------------------------------------------------------------------

	size_t l = 0;  // required buffer size
	struct EffectsBufferBlock *b = eb->head;
	while(b != NULL) {
		l += BLOCK_USED_SPACE(b);
		b = b->next;
	}

	//--------------------------------------------------------------------------
	// allocate buffer and populate
	//--------------------------------------------------------------------------

	unsigned char *buffer = rm_malloc(sizeof(unsigned char) * l);
	unsigned char *offset = buffer;

	b = eb->head;
	while(b != NULL) {
		// write block's data to buffer
		size_t _n = BLOCK_USED_SPACE(b);
		memcpy(offset, b->buffer, _n);
		offset += _n;

		// advance to next block
		b = b->next;
	}

	*n = l;
	return buffer;
}

//------------------------------------------------------------------------------
// effects creation API
//------------------------------------------------------------------------------

// add a node creation effect to buffer
void EffectsBuffer_AddCreateNodeEffect
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------
	
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->nodes_created++;
	stats->properties_set += AttributeSet_Count(*n->attributes);

	EffectType t = EFFECT_CREATE_NODE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write label count
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&label_count, sizeof(label_count), buff);

	//--------------------------------------------------------------------------
	// write labels
	//--------------------------------------------------------------------------

	if(label_count > 0) {
		EffectsBuffer_WriteBytes(labels, sizeof(LabelID) * label_count, buff);
	}

	//--------------------------------------------------------------------------
	// write attribute set
	//--------------------------------------------------------------------------

	const AttributeSet attrs = GraphEntity_GetAttributes((const GraphEntity*)n);
	EffectsBuffer_WriteAttributeSet(attrs, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add a edge creation effect to buffer
void EffectsBuffer_AddCreateEdgeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
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
	
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics () ;
	stats->relationships_created++ ;
	stats->properties_set += AttributeSet_Count (*edge->attributes) ;

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		uint16_t rel_count ;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _create_edge_desc;
	#pragma pack(pop)

	//--------------------------------------------------------------------------
	// populate & write edge
	//--------------------------------------------------------------------------

	_create_edge_desc.t         = EFFECT_CREATE_EDGE ;
	_create_edge_desc.rel_count = 1 ;
	_create_edge_desc.r         = Edge_GetRelationID (edge) ;
	_create_edge_desc.src_id    = Edge_GetSrcNodeID  (edge) ;
	_create_edge_desc.dest_id   = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_create_edge_desc, sizeof (_create_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute set 
	//--------------------------------------------------------------------------

	const AttributeSet attrs =
		GraphEntity_GetAttributes ((const GraphEntity*)edge) ;

	EffectsBuffer_WriteAttributeSet (attrs, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add a node deletion effect to buffer
void EffectsBuffer_AddDeleteNodeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//--------------------------------------------------------------------------

	QueryCtx_GetResultSetStatistics()->nodes_deleted++;

	#pragma pack(push, 1)
	struct {
		EffectType t;
		EntityID id;
	} _delete_node_desc ;
	#pragma pack(pop)

	_delete_node_desc.t  = EFFECT_DELETE_NODE ;
	_delete_node_desc.id = ENTITY_GET_ID (node) ;

	EffectsBuffer_WriteBytes (&_delete_node_desc, sizeof(_delete_node_desc),
			buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add a edge deletion effect to buffer
void EffectsBuffer_AddDeleteEdgeEffect
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	QueryCtx_GetResultSetStatistics()->relationships_deleted++;

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _delete_edge_desc;
	#pragma pack(pop)

	_delete_edge_desc.t       = EFFECT_DELETE_EDGE ;
	_delete_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_delete_edge_desc.r       = Edge_GetRelationID (edge) ;
	_delete_edge_desc.src_id  = Edge_GetSrcNodeID  (edge) ;
	_delete_edge_desc.dest_id = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_delete_edge_desc, sizeof(_delete_edge_desc),
			eb) ;

	EffectsBuffer_IncEffectCount(eb);
}

// add an entity update effect to buffer
static void EffectsBuffer_AddNodeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Node *node,           // updated node
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    entity ID
	//    attribute id
	//    attribute value
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		AttributeID attr_id ;
	} _update_node_desc;
	#pragma pack(pop)

	_update_node_desc.t       = EFFECT_UPDATE_NODE ;
	_update_node_desc.id      = ENTITY_GET_ID (node) ;
	_update_node_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_node_desc, sizeof(_update_node_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue (&value, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add an entity update effect to buffer
static void EffectsBuffer_AddEdgeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Edge *edge,           // updated edge
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//    attribute count (=n)
	//    attributes (id,value) pair
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r;
		NodeID s;
		NodeID d;
		AttributeID attr_id ;
	} _update_edge_desc;
	#pragma pack(pop)

	_update_edge_desc.t       = EFFECT_UPDATE_EDGE ;
	_update_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_update_edge_desc.r       = Edge_GetRelationID (edge) ;
	_update_edge_desc.s       = Edge_GetSrcNodeID  (edge) ;
	_update_edge_desc.d       = Edge_GetDestNodeID (edge) ;
	_update_edge_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_edge_desc, sizeof (_update_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue(&value, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add an entity attribute removal effect to buffer
void EffectsBuffer_AddEntityRemoveAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	GraphEntityType entity_type  // entity type
) {
	// attribute was deleted
	int n = (attr_id == ATTRIBUTE_ID_ALL)
		? AttributeSet_Count(*entity->attributes)
		: 1;

	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->properties_removed += n;

	SIValue v = SI_NullVal();
	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, v);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, v);
	}
}

// add an entity add new attribute effect to buffer
void EffectsBuffer_AddEntityAddAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	// attribute was added
	QueryCtx_GetResultSetStatistics()->properties_set++;

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, value);
	}
}

// add an entity update attribute effect to buffer
void EffectsBuffer_AddEntityUpdateAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->properties_set++; // attribute was set
	stats->properties_removed++; // old attribute was deleted

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, value);
	}
}

// add a node add label effect to buffer
void EffectsBuffer_AddSetRemoveLabelsEffect
(
	EffectsBuffer *buff,     // effect buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // labels
	uint8_t lbl_count,       // number of labels
	EffectType t             // effect type
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	// write node ID
	EffectsBuffer_WriteBytes(&ENTITY_GET_ID(node), sizeof(EntityID), buff); 
	
	// write labels count
	EffectsBuffer_WriteBytes(&lbl_count, sizeof(lbl_count), buff); 
	
	// write label IDs
	EffectsBuffer_WriteBytes(lbl_ids, sizeof(LabelID) * lbl_count, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add a node add labels effect to buffer
void EffectsBuffer_AddLabelsEffect
(
	EffectsBuffer *buff,     // effect buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // added labels
	size_t lbl_count         // number of removed labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	QueryCtx_GetResultSetStatistics()->labels_added += lbl_count;

	EffectType t = EFFECT_SET_LABELS;
	EffectsBuffer_AddSetRemoveLabelsEffect(buff, node, lbl_ids, lbl_count, t);
}

// add a node remove labels effect to buffer
void EffectsBuffer_AddRemoveLabelsEffect
(
	EffectsBuffer *buff,     // effect buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // removed labels
	size_t lbl_count         // number of removed labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	QueryCtx_GetResultSetStatistics()->labels_removed += lbl_count;

	EffectType t = EFFECT_REMOVE_LABELS;
	EffectsBuffer_AddSetRemoveLabelsEffect(buff, node, lbl_ids, lbl_count, t);
}

// add a schema addition effect to buffer
void EffectsBuffer_AddNewSchemaEffect
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // id of the schema
	SchemaType st             // type of the schema
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_SCHEMA;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write schema type
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&st, sizeof(st), buff);

	//--------------------------------------------------------------------------
	// write schema name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(schema_name, buff);

	EffectsBuffer_IncEffectCount(buff);
}

// add an attribute addition effect to buffer
void EffectsBuffer_AddNewAttributeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_ATTRIBUTE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write attribute name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(attr, buff);

	EffectsBuffer_IncEffectCount(buff);
}

void EffectsBuffer_Free
(
	EffectsBuffer *eb
) {
	if(eb == NULL) return;

	// free blocks
	struct EffectsBufferBlock *b = eb->head;
	while(b != NULL) {
		struct EffectsBufferBlock *next = b->next;
		EffectsBufferBlock_Free(b);
		b = next;
	}

	rm_free(eb);
}

