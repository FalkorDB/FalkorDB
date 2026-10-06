/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../serializer_io.h"
#include "../../../../graph/graphcontext.h"
#include "../../../../graph/tensor/tensor.h"
#include "../../../../graph/graph_statistics.h"
#include "../../../../graph/delta_matrix/delta_matrix.h"

// decode tensors
static void _DecodeTensors
(
	SerializerIO rdb,     // RDB
	Delta_Matrix D,       // matrix to populate with tensors
	uint64_t *n_tensors,  // [output] number of tensors loaded
	uint64_t *n_elem      // [output] number of edges loaded
) {
	// format:
	//  total number of tensors
	//
	//  M - number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor
	//
	//  DP - number of tensors
	//  tensors:
	//   tensor i index
	//   tensor j index
	//   tensor

	ASSERT(D         != NULL);
	ASSERT(n_elem    != NULL);
	ASSERT(n_tensors != NULL);

	*n_elem    = 0;
	*n_tensors = 0;

	// read number of tensors
	uint64_t n = SerializerIO_ReadUnsigned(rdb);

	// no tensors, simply return
	if(n == 0) {
		return;
	}

	GrB_Matrix M  = Delta_Matrix_M  (D) ;
	GrB_Matrix DP = Delta_Matrix_DP (D) ;

	GrB_Matrix matrices[2] = {M, DP} ;

	for (int l = 0; l < 2; l++) {
		GrB_Matrix A = matrices[l] ;

		uint64_t _n_tensors = SerializerIO_ReadUnsigned (rdb) ;

		// decode and set tensors
		for (uint64_t k = 0; k < _n_tensors; k++) {
			// read tensor i,j indicies
			GrB_Index i = SerializerIO_ReadUnsigned (rdb) ;
			GrB_Index j = SerializerIO_ReadUnsigned (rdb) ;

			// read tensor blob
			GrB_Index blob_size;
			void *blob = SerializerIO_ReadBuffer (rdb, (size_t*)&blob_size) ;

			// abort on a short read before deserializing a partial tensor blob
			if (SerializerIO_Error (rdb)) {
				rm_free (blob) ;
				return ;
			}
			ASSERT (blob != NULL) ;

			GrB_Vector u;
			GrB_Info info =
				GxB_Vector_deserialize (&u, NULL, blob, blob_size, NULL) ;
			rm_free (blob) ;
			if (info != GrB_SUCCESS) {
				SerializerIO_SetError (rdb,
						"GraphBLAS error %d deserializing tensor (%llu, %llu)",
						(int) info, (unsigned long long) i,
						(unsigned long long) j) ;
				return ;
			}

			// update number of elements loaded
			GrB_Index nvals;
			info = GrB_Vector_nvals (&nvals, u) ;
			ASSERT (info == GrB_SUCCESS) ;
			// a tensor holds at least two edges, a single edge is stored
			// as a scalar entry
			if (nvals < 2) {
				GrB_Vector_free (&u) ;
				SerializerIO_SetError (rdb,
						"tensor (%llu, %llu) holds %llu edges, expected at least 2",
						(unsigned long long) i, (unsigned long long) j,
						(unsigned long long) nvals) ;
				return ;
			}
			*n_elem += nvals;

			// set tensor
			uint64_t v = SET_MSB ((uint64_t)(uintptr_t) u) ;
			info = GrB_Matrix_setElement_UINT64 (A, v, i, j) ;
			if (info != GrB_SUCCESS) {
				GrB_Vector_free (&u) ;
				SerializerIO_SetError (rdb,
						"GraphBLAS error %d setting tensor (%llu, %llu)",
						(int) info, (unsigned long long) i,
						(unsigned long long) j) ;
				return ;
			}
		}

		// set number of loaded tensors
		*n_tensors += _n_tensors;
	}
}

static void _decode_and_load_vector
(
	SerializerIO rdb,
	GrB_Vector *v
) {
	// format:
	//   array
	//   type name
	//   number of entries
	//   number of bytes
	//   handeling

	ASSERT (v   != NULL) ;
	ASSERT (rdb != NULL) ;

	void *arr;              // vector's data
	size_t n;               // number of bytes read
	uint64_t n_entries;     // number of entries
	uint64_t n_bytes;       // data size in bytes
	int handling;           // memory owner GraphBLAS / App
	char   *t_name;         // type name
	size_t t_name_len = 0;  // type name length
	GrB_Info info;

	// load vector from stream
	arr       = SerializerIO_ReadBuffer   (rdb, &n) ;
	t_name    = SerializerIO_ReadBuffer   (rdb, &t_name_len) ;
	n_entries = SerializerIO_ReadUnsigned (rdb) ;
	n_bytes   = SerializerIO_ReadUnsigned (rdb) ;
	handling  = SerializerIO_ReadSigned   (rdb) ;

	// abort on a short read before deriving a GrB_Type from an empty name and
	// handing partial data to GraphBLAS
	if (SerializerIO_Error (rdb)) {
		rm_free (arr) ;
		rm_free (t_name) ;
		*v = NULL ;
		return ;
	}

	// the declared size must match the bytes actually read; GraphBLAS trusts it
	if (n != n_bytes) {
		rm_free (arr) ;
		rm_free (t_name) ;
		*v = NULL ;
		SerializerIO_SetError (rdb,
				"matrix vector holds %zu bytes but declares %llu", n,
				(unsigned long long) n_bytes) ;
		return ;
	}

	// get GrB_Type
	GrB_Type t;  // data type
	// an unrecognized name is not an error to GraphBLAS: it returns NULL
	info = GxB_Type_from_name (&t, t_name) ;
	if (info != GrB_SUCCESS || t == NULL) {
		int name_len = (int) (t_name_len < 64 ? t_name_len : 64) ;
		if (info != GrB_SUCCESS) {
			SerializerIO_SetError (rdb,
					"GraphBLAS error %d resolving matrix value type '%.*s'",
					(int) info, name_len, t_name) ;
		} else {
			SerializerIO_SetError (rdb, "unknown matrix value type '%.*s'",
					name_len, t_name) ;
		}
		rm_free (t_name) ;
		rm_free (arr) ;
		*v = NULL ;
		return ;
	}
	rm_free (t_name) ;

	// load vector
	info = GrB_Vector_new (v, t, 0) ;
	if (info != GrB_SUCCESS) {
		rm_free (arr) ;
		*v = NULL ;
		SerializerIO_SetError (rdb,
				"GraphBLAS error %d creating a matrix vector", (int) info) ;
		return ;
	}

	info = GxB_Vector_load (*v, &arr, t, n_entries, n_bytes, handling, NULL) ;
	if (info != GrB_SUCCESS) {
		// on failure GraphBLAS leaves 'arr' with us
		rm_free (arr) ;
		GrB_Vector_free (v) ;
		*v = NULL ;
		SerializerIO_SetError (rdb,
				"GraphBLAS error %d loading a matrix vector", (int) info) ;
	}
}

// decode a GraphBLAS matrix
static GrB_Matrix _Decode_GrB_Matrix
(
	SerializerIO rdb  // stream
) {
	// format:
	//  GraphBLAS container
	//  unloaded matrix components

	// decode container
	size_t n;
	GxB_Container container;

	container = SerializerIO_ReadBuffer (rdb, &n) ;

	// a short read yields an empty / wrong-sized buffer; interpreting it as a
	// container and writing its fields would overflow the allocation
	if (SerializerIO_Error (rdb) || n != sizeof(struct GxB_Container_struct)) {
		rm_free (container) ;
		SerializerIO_SetError (rdb,
				"matrix container holds %zu bytes, expected %zu", n,
				sizeof (struct GxB_Container_struct)) ;
		return NULL ;
	}

	// nullify container's vectors
    container->p = NULL ;
    container->h = NULL ;
    container->b = NULL ;
    container->i = NULL ;
    container->x = NULL ;
    container->Y = NULL ;

	_decode_and_load_vector (rdb, &container->x) ;
	_decode_and_load_vector (rdb, &container->h) ;
	_decode_and_load_vector (rdb, &container->p) ;
	_decode_and_load_vector (rdb, &container->i) ;
	_decode_and_load_vector (rdb, &container->b) ;

	// abort on a short read before loading a matrix from a partial container;
	// GxB_Container_free reclaims the container and any vectors already loaded
	if (SerializerIO_Error (rdb)) {
		GxB_Container_free (&container) ;
		return NULL ;
	}

	// load A from the container
	GrB_Matrix A;
	GrB_Info info;

	info = GrB_Matrix_new (&A, GrB_BOOL, 0, 0) ;  // matrix type doesn't matter
	ASSERT (info == GrB_SUCCESS) ;

	info = GxB_load_Matrix_from_Container (A, container, NULL) ;
	if (info != GrB_SUCCESS) {
		GrB_Matrix_free (&A) ;
		GxB_Container_free (&container) ;
		SerializerIO_SetError (rdb,
				"GraphBLAS error %d loading a matrix", (int) info) ;
		return NULL ;
	}

	// A is now back to its original state. The container and its p,h,b,i,x
	// GrB_Vectors exist but its vectors all have length 0.

	info = GxB_Container_free (&container) ; // does several O(1)-sized free’s
	ASSERT (info == GrB_SUCCESS) ;

	return A;
}

// decode matrix
static void _Decode_Delta_Matrix
(
	SerializerIO rdb,  // RDB
	Delta_Matrix D     // delta matrix to populate
) {
	// format:
	//  M
	//  DP
	//  DM

	ASSERT (D   != NULL) ;
	ASSERT (rdb != NULL) ;

	GrB_Matrix M  = _Decode_GrB_Matrix (rdb) ;
	GrB_Matrix DP = _Decode_GrB_Matrix (rdb) ;
	GrB_Matrix DM = _Decode_GrB_Matrix (rdb) ;

	// abort on a short read; free any matrices that were decoded and leave the
	// delta matrix empty for the partial-graph teardown to reclaim
	if (SerializerIO_Error (rdb) || M == NULL || DP == NULL || DM == NULL) {
		if (M  != NULL) GrB_Matrix_free (&M) ;
		if (DP != NULL) GrB_Matrix_free (&DP) ;
		if (DM != NULL) GrB_Matrix_free (&DM) ;
		return ;
	}

	GrB_Info info = Delta_Matrix_setMatrices (D, &M, &DP, &DM) ;
	if (info != GrB_SUCCESS) {
		if (M  != NULL) GrB_Matrix_free (&M) ;
		if (DP != NULL) GrB_Matrix_free (&DP) ;
		if (DM != NULL) GrB_Matrix_free (&DM) ;
		SerializerIO_SetError (rdb,
				"GraphBLAS error %d setting delta matrix", (int) info) ;
	}
}

// decode label matrices from rdb
void RdbLoadLabelMatrices_v20
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//  number of label matrices
	//   label id
	//   matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;

	// read number of label matricies
	int n = SerializerIO_ReadUnsigned(rdb);
	
	// decode each label matrix
	for(int i = 0; i < n; i++) {
		// abort on a short read
		if(SerializerIO_Error(rdb)) {
			return;
		}
		// read label ID
		LabelID l = SerializerIO_ReadUnsigned(rdb);
		if(SerializerIO_Error(rdb)) {
			return;
		}
		if(l >= Graph_LabelTypeCount(g)) {
			SerializerIO_SetError(rdb, "label matrix for an unknown label");
			return;
		}

		Delta_Matrix lbl = Graph_GetLabelMatrix(g, l);

		GrB_Index nvals;
		Delta_Matrix_nvals(&nvals, lbl);
		if(nvals != 0) {
			SerializerIO_SetError(rdb, "label matrix decoded twice");
			return;
		}

		_Decode_Delta_Matrix(rdb, lbl);
	}
}

// decode relationship matrices from rdb
void RdbLoadRelationMatrices_v20
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   relation id   X N
	//   matrix        X N
	//   tensors count X N
	//   tensors

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	GrB_Info info;

	// number of relation matricies
	int n = Graph_RelationTypeCount (g) ;

	// decode relationship matrices
	for (int i = 0; i < n; i++) {
		// abort on a short read
		if (SerializerIO_Error (rdb)) {
			return;
		}
		// read relation ID
		RelationID r = SerializerIO_ReadUnsigned (rdb) ;
		if (SerializerIO_Error (rdb)) {
			return;
		}

		// relation matrices are encoded in id order
		if (r != i) {
			SerializerIO_SetError (rdb, "relation matrix out of order") ;
			return ;
		}

		// plant M matrix
		Delta_Matrix DR = Graph_GetRelationMatrix (g, r, false) ;

		GrB_Index nvals;
		Delta_Matrix_nvals (&nvals, DR) ;
		if (nvals != 0) {
			SerializerIO_SetError (rdb, "relation matrix decoded twice") ;
			return ;
		}

		_Decode_Delta_Matrix(rdb, DR);

		// decode tensors
		uint64_t n_elem    = 0;  // number of tensor edges
		uint64_t n_tensors = 0;  // number of tensors in matrix
		_DecodeTensors (rdb, DR, &n_tensors, &n_elem) ;
		if (SerializerIO_Error (rdb)) {
			return ;
		}

		// update graph edge statistics
		// number of edges of type 'r' equals to:
		// |R| - n_tensors + n_elem
		info = Delta_Matrix_nvals (&nvals, DR) ;
		ASSERT (info == GrB_SUCCESS) ;

		GraphStatistics_IncEdgeCount (&g->stats, r, nvals - n_tensors + n_elem) ;
	}
}

// decode adjacency matrix
void RdbLoadAdjMatrix_v20
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   adjacency matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);
	_Decode_Delta_Matrix(rdb, adj);
}

// decode labels matrix
void RdbLoadLblsMatrix_v20
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//   lbls matrix

	ASSERT(g   != NULL);
	ASSERT(rdb != NULL);

	Delta_Matrix lbl = Graph_GetNodeLabelMatrix(g);
	_Decode_Delta_Matrix(rdb, lbl);
}
