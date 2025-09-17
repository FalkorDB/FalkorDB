/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../util/datablock/datablock.h"

// defrage stage
typedef enum {
	DEFRAG_NOT_STARTED,
	DEFRAG_NODES,
	DEFRAG_EDGES
} defrag_stage;

void defrag_attributeset
(
	RedisModuleDefragCtx *ctx,
	AttributeSet set
) {
	ASSERT (set != NULL) ;

	uint16_t n = AttributeSet_Count (set) ;

	for (uint16_t i = 0; i < n ; i++) {
		SIValue v = AttributeSet_GetIdx (set, i, NULL) ;
		if (!SI_HEAP_ALLOCATED (v)) {
			continue ;
		}
		void *p = NULL ;
		void **ref_p = NULL ;
		SIType t = SI_TYPE (v) ;

		switch (t) {
			case T_MAP:
				p     = v.map  ;
				ref_p = (void**)&v.map ;
				break ;

			case T_ARRAY:
				p     = v.array  ;
				ref_p = (void**)&v.array ;
				break ;

			case T_STRING:
			case T_INTERN_STRING:
				p     = v.stringval  ;
				ref_p = (void**)&v.stringval ;
				break ;

			default :
				p     = v.ptrval  ;
				ref_p = (void**)&v.ptrval ;
				break ;
		}

		ASSERT (p     != NULL) ;
		ASSERT (ref_p != NULL) ;

        void *new = RedisModule_DefragAlloc (ctx, p) ;
        if (new) {
			*ref_p = new ;
        }
	}
}

static int defrag_entities
(
	RedisModuleDefragCtx *ctx,
	const Graph *g,
	GraphContext *gc,
	DataBlockIterator *it
) {
	unsigned long i   = 0 ;
	uint64_t      id  = 0 ;
	AttributeSet *set = NULL ;

	while ((set = (AttributeSet*)(DataBlockIterator_Next (it, &id))) != NULL) {
        if ((i % 64 == 0) && RedisModule_DefragShouldStop(ctx)) {
            RedisModule_DefragCursorSet(ctx, i);
            return 1;
        }
		i++ ;
	}

	defrag_attributeset (ctx, *set) ;

	return 0 ;
}

static int defrag_edges
(
	RedisModuleDefragCtx *ctx,
	GraphContext *gc
) {
	const Graph *g = GraphContext_GetGraph (gc) ;
	DataBlockIterator *it = Graph_ScanEdges (g) ;

	return defrag_entities (ctx, g, gc, it) ;
}

static int defrag_nodes
(
	RedisModuleDefragCtx *ctx,
	GraphContext *gc
) {
	const Graph *g = GraphContext_GetGraph (gc) ;
	DataBlockIterator *it = Graph_ScanNodes (g) ;

	return defrag_entities (ctx, g, gc, it) ;
}

int _GraphContextType_Defrag
(
	RedisModuleDefragCtx *ctx,
	RedisModuleString *key,
	void **value
) {
	ASSERT (ctx   != NULL) ;	
	ASSERT (key   != NULL) ;	
	ASSERT (value != NULL) ;	

    int steps = 0;
    unsigned long i = 0;
	defrag_stage stage = DEFRAG_NOT_STARTED ;

    RedisModule_Log (NULL, "notice", "Defrag key: %s",
			RedisModule_StringPtrLen(key, NULL)) ;

	GraphContext *gc = *((GraphContext**)(value)) ;

    // attempt to get cursor
    if (RedisModule_DefragCursorGet (ctx, &i) != REDISMODULE_OK) {
		stage = DEFRAG_NODES ;
	} else {
		stage = (defrag_stage) i ;  // resume
	}

	int res = 0 ;

	switch (stage) {
		case DEFRAG_NODES:
			res = defrag_nodes (ctx, gc) ;
			break ;
		case DEFRAG_EDGES:
			res = defrag_edges (ctx, gc) ;
			break ;
		default:
			ASSERT (false && "unknown defrag stage") ;
	}

    return 0;
}

