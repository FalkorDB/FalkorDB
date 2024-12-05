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
	GB_WHERE1 ("GxB_Matrix_Pending (A)") ;
	//--------------------------------------------------------------------------
	// check inputs
	//--------------------------------------------------------------------------
	GB_RETURN_IF_NULL_OR_FAULTY(A) ;
	GB_RETURN_IF_NULL(pending) ;

	(*pending) = (GB_ANY_PENDING_WORK (A) || GB_hyper_hash_need (A));

	return (GrB_SUCCESS) ;
}

