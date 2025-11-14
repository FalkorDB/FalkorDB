// free vector entries of a tensor
static void _free_vectors
(
	uint64_t *z,       // [ignored] new value
	const uint64_t *x  // current entry
) {
	// see if entry is a vector
	if(!SCALAR_ENTRY(*x)) {
		// free vector
		GrB_Vector V = AS_VECTOR(*x);
		GrB_free(&V);
	}
	*z = MSB_MASK;
}

static void _push_element
(
	uint64_t *z,
	const uint64_t *x,
	const uint64_t *y
) {
	ASSERT(SCALAR_ENTRY(*y));
	if(SCALAR_ENTRY(*x)){
		GrB_Vector v = NULL;
		GrB_Vector_new(&v, GrB_BOOL, GrB_INDEX_MAX);
		GrB_Vector_setElement_BOOL(v, true, *x);
		GrB_Vector_setElement_BOOL(v, true, *y);
		*z = SET_MSB((uint64_t) v);
	} else {
		GrB_Vector v = AS_VECTOR(*x);
		GrB_Vector_setElement_BOOL(v, true, *y);
		*z = SET_MSB((uint64_t) v);
	}
}
