/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the storage component is responsible for managing data loading and offloading
// from and to disk via tidesdb
//
// FalkorDB instance manages a single tidesdb database where each graph uses
// one of more column families
//
// at the moment FalkorDB offloads entire attribute-sets to disk
// node attribute-sets are mapped to disk via the key: N<node_id> e.g.
// 'N280982' while edges uses the E<edge_id> prefix
//
// storage exposes three main functions:
// 1. offload an attribute-set to disk
// 2. load an attribute-set from disk
// 3. delete an attribute-set from disk
#pragma once

#include "db.h"
#include "../graph/entities/graph_entity.h"
#include "../graph/entities/attribute_set.h"

// initialize storage
// returns 0 on success
int Storage_init(void) ;

// offloads attribute sets to tidesdb
// returns 0 on success
int Storage_putAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const AttributeSet *sets,     // array of attribute sets
	size_t n_sets,                // number of sets to offload
	const EntityID *ids           // array of entity IDs
);

// loads attribute sets from tidesdb
// returns 0 on success
int Storage_loadAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	void **sets,                  // array of attribute sets
	size_t n_sets,                // number of sets to load
	const EntityID *ids           // array of entity IDs
);

// delete attribute sets from tidesdb
// returns 0 on success
int Storage_deleteAttributes
(
	tidesdb_column_family_t *cf,  // tidesdb column family
	const EntityID *ids,          // array of entity IDs
	size_t n_ids                  // number of IDs
);

// finalize tidesdb
// returns 0 on success
int Storage_finalize (void) ;

