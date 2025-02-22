//------------------------------------------------------------------------------
// GxB_Matrix_Pending: checks to see if matrix has pending operations
//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_hyper.h"
#include "GB_Pending.h"

GrB_Info GxB_Matrix_Pending     // does matrix has pending operations
(
	GrB_Matrix A,           // matrix to query
	bool *pending           // are there any pending operations
) {
	GB_WHERE1 (A, "GxB_Matrix_Pending (A)") ;

	//--------------------------------------------------------------------------
	// check inputs
	//--------------------------------------------------------------------------
	GB_RETURN_IF_NULL_OR_FAULTY(A) ;
	GB_RETURN_IF_NULL(pending) ;

	int will_wait ;
	GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT) ;

	(*pending) = (will_wait == true) ;

	return (GrB_SUCCESS) ;
}

