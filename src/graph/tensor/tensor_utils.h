#pragma once

// requires:
// #define FUNCTION_IDENTITY to the monoid identity of the semiring.
// #define GET_VALUE(x, y) x= double value given EdgeID y.
// #define ACCUM(x, y) x += y
// Function definition to reduce x to a scalar value.
#define TENSORPICK(Z_TYPE)                                                     \
{                                                                              \
	if(SCALAR_ENTRY(*x)) {                                                     \
		GET_VALUE(*z, (EdgeID) *x);                                            \
	} else { /* Find the minimum weighted edge in the vector. */               \
		GrB_Vector _v = AS_VECTOR(*x);                                         \
		/* stack allocate the iterator */                                      \
		struct GB_Iterator_opaque _i;                                          \
		GxB_Iterator i = &_i;                                                  \
		*z = FUNCTION_IDENTITY;                                                \
		if (_v == NULL) return;                                                \
                                                                               \
		GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));                       \
		GrB_Info info  = GxB_Vector_Iterator_seek(i, 0);                       \
                                                                               \
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
	}                                                                          \
}

#define TENSOR_REDUCE_TWO \
{                                                                              \
	*z = FUNCTION_IDENTITY;\                                                   \
	for(int k = 0; k < 2; k++) { \                                             \
		uint64_t _x = k == 1? *y: *x;                                          \
		Z_TYPE currV;                                                          \
		if(SCALAR_ENTRY(_x)) {                                                 \
			GET_VALUE(currV, _x);                                              \
			ACCUM(*z, currV);                                                  \
		} else { /* Find the minimum weighted edge in the vector. */           \
			GrB_Vector _v = AS_VECTOR(_x);                                     \
			ASSERT(_v != NULL);                                                \
			struct GB_Iterator_opaque _i;                                      \
			GxB_Iterator i = &_i;                                              \
			GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));                   \
			GrB_Info info = GxB_Vector_Iterator_seek(i, 0);                    \
			EdgeID currID;                                                     \
			while(info != GxB_EXHAUSTED) {                                     \
				currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);             \
				GET_VALUE(currV, currID);                                      \
				ACCUM(*z, currV);                                              \
				info = GxB_Vector_Iterator_next(i);                            \
			}                                                                  \
		}                                                                      \
	}                                                                          \
}
