/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graph/entities/attribute_set.h"

// property path defines an access path to either a node or edge attribute-set
// as well as a dictionary
// e.g.
//
// N.flight[1].duration.minutes
//
// Where 'N' is the container we're accessing (Node/Edge/Map)
// 'flight[1].duration.minutes' is the access path
//
// a path is constructed out of attribute and array sub-scripts
// in the above example 'flight' referes to an attribute access of 'N'
// while '[1]' is accessing the second element in the 'N.flight' array


// PropertyPathElementType defines the different path element types
typedef enum {
	PATH_ELEMENT_PROPERTY_ID = 1,  // access a property by its ID
	PATH_ELEMENT_PROPERTY    = 2,  // access a property by its name
	PATH_ELEMENT_INDEX       = 4,  // access an array by index
} PropertyPathElementType;

// PropertyPathElement a element on the path
typedef union {
	const char *property;  // property name
	unsigned int index;    // either a property ID or an array index
} PropertyPathElement;

// PropertyPath defines an access path e.g. `flight[1].duration.minutes`
// by breaking the path into its individual elements:
// 1. flight
// 2. [1]
// 3. duration
// 4. minutes
typedef struct {
	PropertyPathElementType *types;  // type of each element
	PropertyPathElement *elements;   // array of path elements
	unsigned int size;               // number of elements in the path
} PropertyPath;

// create a new empty property path
PropertyPath *PropertyPath_new(void);

// returns the length of the path
unsigned int PropertyPath_len
(
	const PropertyPath *path
);

//------------------------------------------------------------------------------
// PropertyPath getters
//------------------------------------------------------------------------------

// get path element at specified position
// returns false if element doesn't exists, true otherwise
bool PropertyPath_getElement
(
	PropertyPathElementType *t,  // [output] [optional] element type
	PropertyPathElement *v,      // [output] [optional] element value
	const PropertyPath *path,    // path object
	unsigned int i               // element position
);

// get a property element from position i
// return false if element doesn't exists, true otherwise
void PropertyPath_getProperty
(
	const char **v,            // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
);

// get a property id element from position i
// return false if element doesn't exists, true otherwise
void PropertyPath_getPropertyID
(
	AttributeID *v,            // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
);

// get a array subscript element from position i
// return false if element doesn't exists, true otherwise
void PropertyPath_getArrayIdx
(
	unsigned int *v,           // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
);

//------------------------------------------------------------------------------
// PropertyPath construction
//------------------------------------------------------------------------------

// appends property access to the end of the path
void PropertyPath_addProperty
(
	PropertyPath *path,   // path to append to
	const char *property  // property to add
);

// appends property ID access to the end of the path
void PropertyPath_addPropertyID
(
	PropertyPath *path,  // path to append to
	AttributeID id       // property ID to add
);

// appends array subscript access to the end of the path
void PropertyPath_addArrayIdx
(
	PropertyPath *path,   // path to append to
	unsigned int idx      // property ID to add
);

// update an existing element
void PropertyPath_updateElement
(
	PropertyPath *path,         // path to update
	unsigned int i,             // element index to update
	PropertyPathElementType t,  // new element type
	PropertyPathElement v       // new element value
);

//------------------------------------------------------------------------------
// PropertyPath debug
//------------------------------------------------------------------------------

// prints path to stdout
void PropertyPath_print
(
	const PropertyPath *path  // path to print
);

// free a property path
void PropertyPath_free
(
	PropertyPath *path
);

