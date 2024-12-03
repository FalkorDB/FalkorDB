/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "query_ctx.h"
#include "util/rmalloc.h"
#include "property_path.h"

// create a new empty property path
PropertyPath *PropertyPath_new(void) {
	PropertyPath *path;

	path = rm_malloc(sizeof(PropertyPath));

	path->types    = NULL;
	path->elements = NULL;
	path->size     = 0;

	return path;
}

// create a new property path for a property access
PropertyPath *PropertyPath_propertyAccess
(
	const char *property  // property to access
) {
	ASSERT(property != NULL);

	PropertyPath *path = rm_malloc(sizeof(PropertyPath));

	path->size     = 1;
	path->types    = rm_malloc(sizeof(PropertyPathElementType));
	path->elements = rm_malloc(sizeof(PropertyPathElement));

	path->types[0]             = PATH_ELEMENT_PROPERTY;
	path->elements[0].property = property;

	// set propery ID
	GraphContext *gc        = QueryCtx_GetGraphCtx();
	path->elements[0].index = GraphContext_GetAttributeID(gc, property);

	return path;
}

// reverse path
// a.b.c -> c.b.a
void PropertyPath_reverse
(
	PropertyPath *path  // path to reverse
) {
	ASSERT(path != NULL);

	for(int i = 0; i < path->size / 2; i++) {
		int j = path->size -i -1;  // replace i <-> j

		PropertyPathElementType t_type = path->types[i];
		path->types[i] = path->types[j];
		path->types[j] = t_type;

		PropertyPathElement t_elem = path->elements[i];
		path->elements[i] = path->elements[j];
		path->elements[j] = t_elem;
	}
}

// returns the length of the path
unsigned int PropertyPath_len
(
	const PropertyPath *path
) {
	ASSERT(path != NULL);

	return path->size;
}

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
) {
	ASSERT(path != NULL);

	// make sure i is within bounds
	if(i > path->size) {
		return false;
	}

	if(t != NULL) {
		*t = path->types[i];
	}

	if(v != NULL) {
		*v = path->elements[i];
	}

	return true;
}

// get a property element from position i
void PropertyPath_getProperty
(
	const char **v,            // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
) {
	// validations
	ASSERT(v              != NULL);
	ASSERT(i              <= path->size);             // i must be within bounds
	ASSERT(path           != NULL);
	ASSERT(path->types[i] == PATH_ELEMENT_PROPERTY);  // assert element type

	*v = path->elements[i].property;
}

// get a property id element from position i
void PropertyPath_getPropertyID
(
	AttributeID *v,            // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
) {
	// validations
	ASSERT(v              != NULL);
	ASSERT(i              <= path->size);             // i must be within bounds
	ASSERT(path           != NULL);
	ASSERT(path->types[i] == PATH_ELEMENT_PROPERTY);  // assert element type

	*v = path->elements[i].index;
}

// get a array subscript element from position i
void PropertyPath_getArrayIdx
(
	unsigned int *v,           // [output] element value
	const PropertyPath *path,  // path object
	unsigned int i             // element position
) {
	// validations
	ASSERT(v              != NULL);
	ASSERT(i              <= path->size);          // i must be within bounds
	ASSERT(path           != NULL);
	ASSERT(path->types[i] == PATH_ELEMENT_INDEX);  // assert element type

	*v = path->elements[i].index;
}

//------------------------------------------------------------------------------
// PropertyPath construction
//------------------------------------------------------------------------------

// grow property path by `n` elements
static void _PropertyPath_grow
(
	PropertyPath *path
) {
	ASSERT(path != NULL);

	// make room for new element
	path->elements = rm_realloc(path->elements,
			sizeof(PropertyPathElement) * (path->size + 1));

	path->types = rm_realloc(path->types,
			sizeof(PropertyPathElementType) * (path->size + 1));

	// increase element count
	path->size++;
}

// appends property access to the end of the path
void PropertyPath_addProperty
(
	PropertyPath *path,   // path to append to
	const char *property  // property to add
) {
	// validations
	ASSERT(path     != NULL);
	ASSERT(property != NULL);

	_PropertyPath_grow(path);

	// set property name
	path->types[path->size-1]             = PATH_ELEMENT_PROPERTY;
	path->elements[path->size-1].property = property;

	// set property ID
	GraphContext *gc = QueryCtx_GetGraphCtx();
	AttributeID attr_id = GraphContext_GetAttributeID(gc, property);
	path->elements[path->size-1].index = attr_id;
}

// appends array subscript access to the end of the path
void PropertyPath_addArrayIdx
(
	PropertyPath *path,  // path to append to
	unsigned int idx     // property ID to add
) {
	// validations
	ASSERT(path != NULL);

	_PropertyPath_grow(path);

	// set array index
	path->types[path->size-1]             = PATH_ELEMENT_INDEX;
	path->elements[path->size-1].index    = idx;
	path->elements[path->size-1].property = NULL;
}

//------------------------------------------------------------------------------
// PropertyPath debug
//------------------------------------------------------------------------------

// prints path to stdout
void PropertyPath_print
(
	const PropertyPath *path  // path to print
) {
	ASSERT(path != NULL);

	for(unsigned int i = 0; i < path->size; i++) {
		switch(path->types[i]) {
			case PATH_ELEMENT_INDEX:
				printf("[%d]", path->elements[i].index);
				break;
			case PATH_ELEMENT_PROPERTY:
				printf(".%s", path->elements[i].property);
				break;
			default:
				ASSERT(false && "unknown PropertyPathElementType");
				break;
		}
	}
	printf("\r\n");
}

// free a property path
void PropertyPath_free
(
	PropertyPath *path
) {
	ASSERT(path != NULL);
	
	if(path->types != NULL)    rm_free(path->types);
	if(path->elements != NULL) rm_free(path->elements);

	rm_free(path);
}

