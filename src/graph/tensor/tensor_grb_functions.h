// sets z to true if both entries are not explicit zombies.
// if either x or y is an explicit zombie z is set to false (bool zombie).
static void _entry_present (bool *z, const bool *x, const uint64_t *y)
{
	*z = (*x != BOOL_ZOMBIE) && ((*y) != U64_ZOMBIE);
}

// define the jit string for _entry_present
#define _ENTRY_PRESENT_JIT_STR                                                 \
"void _entry_present (bool *z, const bool *x, const uint64_t *y)\n"            \
"{\n"                                                                          \
"	*z = (*x) && ((*y) !=  (1UL << (sizeof(uint64_t) * 8 - 1))) ;\n"             \
"}"

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
	*z = U64_ZOMBIE;
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
        ASSERT(*z == *x);
        GrB_Vector v = AS_VECTOR(*z);
        GrB_Vector_setElement_BOOL(v, true, *y);
    }
}