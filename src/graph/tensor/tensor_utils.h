#pragma once
#define FDB_IDXBIOP_SIGNATURE(F, CTX_TYPE)                                     \
static void F                                                                  \
(                                                                              \
	double *z,                                                                 \
	const uint64_t *x,                                                         \
    GrB_Index ix,                                                              \
    GrB_Index jx,                                                              \
    const bool *y,                                                             \
    GrB_Index iy,                                                              \
    GrB_Index jy,                                                              \
	const CTX_TYPE *ctx                                                        \
)
#define FDB_IDXUOP_SIGNATURE(F, CTX_TYPE)                                      \
static void F                                                                  \
(                                                                              \
	double *z,                                                                 \
	const uint64_t *x,                                                         \
    GrB_Index ix,                                                              \
    GrB_Index jx,                                                              \
	const CTX_TYPE *ctx                                                        \
)

// requires:
// #define FUNCTION_IDENTITY to the monoid identity of the semiring.
// #define GET_VALUE(x, y) x= double value given EdgeID y.
// #define ACCUM(x, y) x += y
// Function definition to reduce x to a scalar value.
#define TENSORPICK(Z_TYPE)                                                     \
{                                                                              \
	if(SCALAR_ENTRY(*x)) {                                                     \
        Edge currE;                                                            \
        Graph_GetEdge(ctx->g, (EdgeID) *x, &currE);                            \
		GET_VALUE(*z, *x);                                                     \
	} else { /* Find the minimum weighted edge in the vector. */               \
		GrB_Vector _v = AS_VECTOR(*x);                                         \
		GxB_Iterator i = NULL;                                                 \
		*z = FUNCTION_IDENTITY;                                                \
		if (_v == NULL)                                                        \
            return;                                                            \
                                                                               \
		GrB_Info info = GxB_Iterator_new(&i);                                  \
		ASSERT(info == GrB_SUCCESS)                                            \
		info = GxB_Vector_Iterator_attach(i, _v, NULL);                        \
		ASSERT(info == GrB_SUCCESS)                                            \
		info = GxB_Vector_Iterator_seek(i, 0);                                 \
		ASSERT(info == GrB_SUCCESS)                                            \
        Z_TYPE currV;                                                          \
		EdgeID currID;                                                         \
                                                                               \
		while(info != GxB_EXHAUSTED)                                           \
		{                                                                      \
		    currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);                 \
		    GET_VALUE(currV, currID);                                          \
			ACCUM(*z, currV);                                                  \
		    info = GxB_Vector_Iterator_next(i);                                \
		}                                                                      \
                                                                               \
		GxB_Iterator_free(&i);                                                 \
	}                                                                          \
}
