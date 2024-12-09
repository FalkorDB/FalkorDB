/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/entities/attribute_set.h"
#include "../arithmetic/arithmetic_expression.h"

// enum describing how a SET directive should treat pre-existing properties
typedef enum {
	UPDATE_UNSET   = 0,  // default, should not be encountered
	UPDATE_MERGE   = 1,  // merge new properties into existing property map
	UPDATE_REPLACE = 2,  // replace existing property map with new properties
} UPDATE_MODE;

// key-value pair of an attribute ID and the value to be associated with it
// TODO: consider replacing contents of PropertyMap
// (for ops like Create) with this
typedef struct {
	const char *attribute;   // attribute name
	AttributeID attr_id;     // attribute id
	struct AR_ExpNode *exp;  // LHS expression
	UPDATE_MODE mode;        // update mode
	const char **sub_path;   // nested attributes
} PropertySetCtx;

// context describing an update expression
typedef struct {
	int record_idx;              // record offset this entity is stored at
	const char *alias;           // access-safe alias of the entity being updated
	const char **add_labels;     // labels to add to the node
	const char **remove_labels;  // labels to remove from node
	PropertySetCtx *properties;  // properties to set
} EntityUpdateEvalCtx;

// create a new update context
EntityUpdateEvalCtx *UpdateCtx_New
(
	const char *alias  // entity alias
);

// clone update context
EntityUpdateEvalCtx *UpdateCtx_Clone
(
	const EntityUpdateEvalCtx *orig  // context to clone
);

// clear update context
void UpdateCtx_Clear
(
	EntityUpdateEvalCtx *ctx  // context to clear
);

// free update contex
void UpdateCtx_Free
(
	EntityUpdateEvalCtx *ctx  // context to free
);

