/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"
#include "../../graph/delta_matrix/delta_utils.h"

GrB_Info _get_rows_delta
(
	Delta_Matrix *C, 	   // output matrix
	const Delta_Matrix A,  // input matrix
	const GrB_Vector _N    // filtered rows
) {
	ASSERT (C != NULL);
	ASSERT (*C == NULL);
	GrB_Descriptor  desc   =  NULL;
	GrB_Matrix      m      =  DELTA_MATRIX_M(A);
	GrB_Matrix      dp     =  DELTA_MATRIX_DELTA_PLUS(A);
	GrB_Matrix      dm     =  DELTA_MATRIX_DELTA_MINUS(A);

	GrB_Index       nrows;
	GrB_Index       ncols;
	GrB_Index       nvals;
	GrB_Info        info;
	Delta_Matrix_nrows(&nrows, A);
	Delta_Matrix_ncols(&ncols, A);
	GrB_Vector_nvals  (&nvals, _N);

	Delta_Matrix_new(C, GrB_UINT64, nvals, ncols, false);

	GrB_Matrix      cm     =  DELTA_MATRIX_M(*C);
	GrB_Matrix      cdp    =  DELTA_MATRIX_DELTA_PLUS(*C);
	GrB_Matrix      cdm    =  DELTA_MATRIX_DELTA_MINUS(*C);

	GrB_Descriptor_new(&desc);

	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	info = GxB_Matrix_extract_Vector(cm, NULL, NULL, m, _N, NULL, desc);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_Matrix_extract_Vector(cdp, NULL, NULL, dp, _N, NULL, desc);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_Matrix_extract_Vector(cdm, NULL, NULL, dm, _N, NULL, desc);
	ASSERT(info == GrB_SUCCESS);

	GrB_Descriptor_free(&desc);
}

GrB_Info _compile_matricies
(
	GrB_Matrix *A,             // [output] matrix
	const Delta_Matrix *mats,  // [optional] matricies to consider
	unsigned short n_mats,     // number of matricies
	GrB_Scalar azomb,          // zombie value of output matrix
	GrB_Scalar relzomb         // Input matrix zombie value
) {
	ASSERT(n_mats > 0);

	GrB_Info      info;
	Delta_Matrix  C = NULL;
	GrB_Index     nrows;
	GrB_Index     ncols;
	Delta_Matrix_nrows(&nrows, mats[0]);
	Delta_Matrix_ncols(&ncols, mats[0]);
	if (n_mats == 1){
		// export relation matrix to A
		info = Delta_Matrix_export_structure(A, mats[0]);
		ASSERT(info == GrB_SUCCESS);
	} else {
		// given the semiring I am using, C will be an invalid delta matrix.
		// but it is quickly exported so it should not be an issue.
		info = Delta_Matrix_new(&C, GrB_BOOL, nrows, ncols, false);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_eWiseUnion(
			C, GrB_ONEB_BOOL, mats[0], relzomb, mats[1], relzomb);
		ASSERT(info == GrB_SUCCESS);

		// in case there are multiple relation types, include them in A
		for(unsigned short i = 2; i < n_mats; i++) {
			info = Delta_eWiseUnion(
				C, GrB_ONEB_BOOL, C, azomb, mats[i], relzomb);
			ASSERT(info == GrB_SUCCESS);
		}
		
		Delta_Matrix_wait(C, true);
		*A = DELTA_MATRIX_M(C);
		DELTA_MATRIX_M(C) = NULL;
		// info = Delta_Matrix_export(A, C);
		ASSERT(info == GrB_SUCCESS);
	}
	Delta_Matrix_free(&C);
	return info;
}

GrB_Info _get_rows_with_labels
(
	GrB_Vector rows,         // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls    // number of labels
) {
	ASSERT(rows != NULL);
	
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);
		GrB_Matrix L = NULL;	
		GrB_OK(Delta_Matrix_export(&L, DL));

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			GrB_Matrix M;
			GrB_OK(Delta_Matrix_export(&M, DL));

			GrB_OK(GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL));

			GrB_Matrix_free(&M);
		}

		GrB_OK(GxB_Vector_diag(rows, L, 0, NULL));
		GrB_free(&L);
	} else if(rows != NULL) {
		// N = [1,....1]
		GrB_OK(GrB_Vector_assign_BOOL(
			rows, NULL, NULL, true, GrB_ALL, 0, NULL));
	}
}

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 U R1 U ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info get_sub_adjecency_matrix
(
	GrB_Matrix *A,           // [output] matrix
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	bool symmetric           // build a symmetric matrix
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);

	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	GrB_Info info;
	bool extractRows = false;

	GrB_Matrix      _A       =  NULL;    // output matrix
	GrB_Matrix      _A_T     =  NULL;    // output matrix
	GrB_Vector      _N       =  NULL;    // output filtered rows
	GrB_Scalar      u64zomb  =  NULL;
	GrB_Scalar      bzomb    =  NULL;
	GrB_Matrix      L        =  NULL;
	GrB_Descriptor  desc     =  NULL;

	GrB_Index nrows;  // number of rows in matrix
	GrB_Index rows_nvals; // number of entries in rows vector  

	// GrB_Global_set_INT32(GrB_GLOBAL, true, GxB_BURBLE);
	GrB_Scalar_new(&bzomb, GrB_BOOL);
	GrB_Scalar_setElement_BOOL(bzomb, BOOL_ZOMBIE);

	GrB_Scalar_new(&u64zomb, GrB_UINT64);
	GrB_Scalar_setElement_UINT64(u64zomb, U64_ZOMBIE);

	GrB_Descriptor_new(&desc);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST);

	nrows = Graph_RequiredMatrixDim(g);

	// create vector N denoting all nodes passing the labels filter
	info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	_get_rows_with_labels(_N, g, lbls, n_lbls);

	info = GrB_Vector_nvals(&rows_nvals, _N);

	//TODO: find best value for this hueristic
	extractRows = rows_nvals < Graph_NodeCount(g) / 2 && n_rels > 0;

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		Delta_Matrix D = Graph_GetAdjacencyMatrix(g, false);
		info = Delta_Matrix_export_structure(&_A, D);
		ASSERT(info == GrB_SUCCESS);
		if(symmetric) {
			D = Graph_GetAdjacencyMatrix(g, true);
			info = Delta_Matrix_export_structure(&_A_T, D);
			ASSERT(info == GrB_SUCCESS);
		}
	} else {
		Delta_Matrix  *rel_ms  =  rm_calloc(n_rels, sizeof(Delta_Matrix)) ;
		for(int i = 0; i < n_rels; ++i) {
			RelationID id = rels[i];
			Delta_Matrix temp_m = Graph_GetRelationMatrix(g, id, false);
			if(extractRows){
				_get_rows_delta(&rel_ms[i], temp_m, _N);
			} else {
				rel_ms[i] = temp_m;
			}
		}
		info = _compile_matricies(
			&_A, rel_ms, n_rels, bzomb, u64zomb);
		ASSERT(info == GrB_SUCCESS);

		if(extractRows){
			for(int i = 0; i < n_rels; ++i) 
				Delta_Matrix_free(&rel_ms[i]);
		}
		
		rm_free(rel_ms);
	}

	if(n_lbls > 0) {
		// A = L * A * L
		GrB_Matrix temp = NULL;
		GrB_Matrix_new(&temp, GrB_BOOL, rows_nvals, rows_nvals);
		info = GxB_Matrix_extract_Vector(
			temp, NULL, NULL, _A, (extractRows? NULL: _N), _N, desc);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_free(&_A);
		_A = temp;
		temp = NULL;
		if(_A_T){
			GrB_Matrix_new(&temp, GrB_BOOL, rows_nvals, rows_nvals);
			info = GxB_Matrix_extract_Vector(
			temp, NULL, NULL, _A_T, (extractRows? NULL: _N), _N, desc);
			info = GrB_Matrix_free(&_A_T);
			_A_T = temp;
		}
	}
	
	if(symmetric) {
		if(_A_T) {
			// make A symmetric A = A + At
			info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
					_A, _A_T, NULL);
			ASSERT(info == GrB_SUCCESS);
		} else {
			// make A symmetric A = A + At
			info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
					_A, _A, GrB_DESC_T1);
			ASSERT(info == GrB_SUCCESS);
		}
		
	}

	// determine the number of nodes in the graph
	// this includes deleted nodes
	size_t n = Graph_UncompactedNodeCount(g);

	if(n_lbls == 0){
		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);
	}

	if(rows != NULL) {
		info = GrB_Vector_resize(_N, n);
		ASSERT(info == GrB_SUCCESS);
	}

	// set outputs
	*A = _A;
	if(rows) {
		*rows = _N;
		_N = NULL;
	}
	GrB_free(&L);
	GrB_free(&_N);
	GrB_free(&_A_T);
	GrB_free(&bzomb);
	GrB_free(&u64zomb);
	GrB_free(&desc);
	// GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	return info;
}

#if 0
// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 U R1 U ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Matrix
(
	GrB_Matrix *A,           // [output] matrix
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	bool symmetric,          // build a symmetric matrix
	bool compact             // remove unused row & columns
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);

	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	GrB_Info info;
	bool extractRows = false;

	GrB_Matrix      _A       =  NULL;    // output matrix
	GrB_Matrix      _A_T     =  NULL;    // output matrix
	GrB_Vector      _N       =  NULL;    // output filtered rows
	GrB_Scalar      u64zomb  =  NULL;
	GrB_Scalar      bzomb    =  NULL;
	GrB_Matrix      L        =  NULL;
	GrB_Descriptor  desc     =  NULL;

	GrB_Index nrows;  // number of rows in matrix
	GrB_Index rows_nvals; // number of entries in rows vector  

	GrB_Global_set_INT32(GrB_GLOBAL, true, GxB_BURBLE);
	GrB_Scalar_new(&bzomb, GrB_BOOL);
	GrB_Scalar_setElement_BOOL(bzomb, BOOL_ZOMBIE);

	GrB_Scalar_new(&u64zomb, GrB_UINT64);
	GrB_Scalar_setElement_UINT64(u64zomb, U64_ZOMBIE);

	GrB_Descriptor_new(&desc);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST);

	nrows = Graph_RequiredMatrixDim(g);

	// create vector N denoting all nodes passing the labels filter
	info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);


	// enforce labels
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);
		
		info = Delta_Matrix_export(&L, DL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		info = GxB_Vector_diag(_N, L, 0, NULL);
		ASSERT(info == GrB_SUCCESS);
	} else if(rows != NULL) {
		// N = [1,....1]
		info = GrB_Vector_assign_BOOL(
			_N, NULL, NULL, true, GrB_ALL, nrows, NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	info = GrB_Vector_nvals(&rows_nvals, _N);

	//TODO: find best value for this hueristic
	extractRows = rows_nvals < Graph_NodeCount(g) / 2 && n_rels > 0;

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		Delta_Matrix D = Graph_GetAdjacencyMatrix(g, false);
		info = Delta_Matrix_export_structure(&_A, D);
		ASSERT(info == GrB_SUCCESS);
		if(symmetric) {
			D = Graph_GetAdjacencyMatrix(g, true);
			info = Delta_Matrix_export_structure(&_A_T, D);
			ASSERT(info == GrB_SUCCESS);
		}
	} else {
		Delta_Matrix  *rel_ms  =  rm_calloc(n_rels, sizeof(Delta_Matrix)) ;
		for(int i = 0; i < n_rels; ++i) {
			RelationID id = rels[i];
			Delta_Matrix temp_m = Graph_GetRelationMatrix(g, id, false);
			if(extractRows){
				_get_rows_delta(&rel_ms[i], temp_m, _N);
			} else {
				rel_ms[i] = temp_m;
			}
		}
		info = _compile_matricies(
			&_A, rel_ms, n_rels, bzomb, u64zomb);
		ASSERT(info == GrB_SUCCESS);

		if(extractRows){
			for(int i = 0; i < n_rels; ++i) 
				Delta_Matrix_free(&rel_ms[i]);
		}
		
		rm_free(rel_ms);
	}

	if(L != NULL) {
		// A = L * A * L
		GrB_Matrix temp = NULL;
		GrB_Matrix_new(&temp, GrB_BOOL, rows_nvals, rows_nvals);
		info = GxB_Matrix_extract_Vector(
			temp, NULL, NULL, _A, (extractRows? NULL: _N), _N, desc);
		ASSERT(info == GrB_SUCCESS);

		// expand matrix back to original size.
		info = GxB_Matrix_assign_Vector(
			_A, NULL, NULL, temp, _N, _N, desc);
		ASSERT(info == GrB_SUCCESS);
		GrB_free(&temp);

		if(_A_T){
			// A = L * A * L
			info = GrB_mxm(_A_T, NULL, NULL, GxB_ANY_PAIR_BOOL, L, _A_T, NULL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_mxm(_A_T, NULL, NULL, GxB_ANY_PAIR_BOOL, _A_T, L, NULL);
			ASSERT(info == GrB_SUCCESS);
		}
		
		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	}
	
	if(symmetric) {
		if(_A_T) {
			// make A symmetric A = A + At
			info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
					_A, _A_T, NULL);
			ASSERT(info == GrB_SUCCESS);
		} else {
			// make A symmetric A = A + At
			info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
					_A, _A, GrB_DESC_T1);
			ASSERT(info == GrB_SUCCESS);
		}
		
	}

	if(compact) {
		// determine the number of nodes in the graph
		// this includes deleted nodes
		size_t n = Graph_UncompactedNodeCount(g);

		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);

		if(rows != NULL) {
			info = GrB_Vector_resize(_N, n);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	// set outputs
	*A = _A;
	if(rows) {
		*rows = _N;
		_N = NULL;
	}
	GrB_free(&_N);
	GrB_free(&_A_T);
	GrB_free(&bzomb);
	GrB_free(&u64zomb);
	GrB_free(&desc);
	GrB_Global_set_INT32(GrB_GLOBAL, false, GxB_BURBLE);
	return info;
}
#else // preserved for benchmarking
// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 U R1 U ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Matrix
(
	GrB_Matrix *A,           // [output] matrix
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	bool symmetric,          // build a symmetric matrix
	bool compact             // remove unused row & columns
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);

	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	GrB_Info info;
	Delta_Matrix D;   // graph delta matrix
	GrB_Index nrows;  // number of rows in matrix
	GrB_Index ncols;  // number of columns in matrix
	GrB_Matrix _A;    // output matrix
	GrB_Vector _N;    // output filtered rows

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		D = Graph_GetAdjacencyMatrix(g, false);
	} else {
		RelationID id = rels[0];
		D = Graph_GetRelationMatrix(g, id, false);
	}
	ASSERT(D != NULL);
	
	// export relation matrix to A
	// TODO: extend Delta_Matrix_export to include a exported matrix type
	// cast if needed
	info = Delta_Matrix_export(&_A, D);
	ASSERT(info == GrB_SUCCESS);

	// in case there are multiple relation types, include them in A
	for(unsigned short i = 1; i < n_rels; i++) {
		D = Graph_GetRelationMatrix(g, rels[i], false);

		GrB_Matrix M;
		info = Delta_Matrix_export(&M, D);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_eWiseAdd_Monoid(_A, NULL, NULL, GxB_ANY_BOOL_MONOID,
				_A, M, NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix_free(&M);
	}

	info = GrB_Matrix_nrows(&nrows, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, _A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a square matrix
	ASSERT(nrows == ncols);

	// create vector N denoting all nodes passing the labels filter
	if(rows != NULL) {
		info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	// enforce labels
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// A = L * A * L
		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_PAIR_BOOL, L, _A, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_PAIR_BOOL, _A, L, NULL);
		ASSERT(info == GrB_SUCCESS);

		// set N to L's main diagonal denoting all participating nodes 
		if(rows != NULL) {
			info = GxB_Vector_diag(_N, L, 0, NULL);
			ASSERT(info == GrB_SUCCESS);
		}

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	} else if(rows != NULL) {
		// N = [1,....1]
		GrB_Scalar scalar;
		info = GrB_Scalar_new(&scalar, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Scalar_setElement_BOOL(scalar, true);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(_N, NULL, NULL, scalar, GrB_ALL, nrows,
				NULL);
		ASSERT(info == GrB_SUCCESS);
    
		info = GrB_free(&scalar);
		ASSERT(info == GrB_SUCCESS);
	}

	if(symmetric) {
		// make A symmetric A = A + At
		info = GrB_Matrix_eWiseAdd_Semiring(_A, NULL, NULL, GxB_ANY_PAIR_BOOL,
				_A, _A, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
	}

	if(compact) {
		// determine the number of nodes in the graph
		// this includes deleted nodes
		size_t n = Graph_UncompactedNodeCount(g);

		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);

		if(rows != NULL) {
			info = GrB_Vector_resize(_N, n);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	// set outputs
	*A = _A;
	if(rows) *rows = _N;

	return info;
}
#endif
