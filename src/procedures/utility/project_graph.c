/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"
#include "GraphBLAS.h"
#include "../../util/rmalloc.h"
#include <stdatomic.h>
#include <pthread.h>

typedef struct {
	const Graph *g;
	AttributeID attr_id;
	double default_value;
	bool has_default;
	project_strategy strategy;
	atomic_bool *invalid;
} edge_project_ctx;

typedef struct {
	const Graph *g;
	AttributeID attr_id;
	double default_value;
	bool has_default;
	atomic_bool *invalid;
} node_project_ctx;

static void _project_edge_entry
(
	double *z,                   // [output] projected edge value
	const void *x,               // relation matrix entry (edge id/tensor)
	GrB_Index i,                 // row index [unused]
	GrB_Index j,                 // col index [unused]
	const edge_project_ctx *ctx  // projection context
);

static void _project_row_entry
(
	double *z,                   // [output] projected node value
	const void *x,               // row entry [unused]
	GrB_Index i,                 // row index (node id)
	GrB_Index j,                 // col index [unused]
	const node_project_ctx *ctx  // projection context
);

static GrB_Type pgtm_edge_ctx_type = NULL;
static GrB_Type pgtm_node_ctx_type = NULL;
static GrB_IndexUnaryOp pgtm_edge_project_op = NULL;
static GrB_IndexUnaryOp pgtm_row_project_op = NULL;

// guards pgtm_* globals below - can't use pthread_once here since GraphBLAS
// may be finalized/reinitialized (e.g. between unit tests in the same
// process), which requires re-running _init_pgtm_ops more than once
static pthread_mutex_t pgtm_ops_mutex = PTHREAD_MUTEX_INITIALIZER;

static void _init_pgtm_ops
(
	void
) {
	GrB_OK (GrB_Type_new(&pgtm_edge_ctx_type, sizeof(edge_project_ctx)));
	GrB_OK (GrB_Type_new(&pgtm_node_ctx_type, sizeof(node_project_ctx)));

	GrB_OK (GrB_IndexUnaryOp_new(&pgtm_edge_project_op,
			(GxB_index_unary_function) _project_edge_entry,
			GrB_FP64, GrB_UINT64, pgtm_edge_ctx_type));

	GrB_OK (GrB_IndexUnaryOp_new(&pgtm_row_project_op,
			(GxB_index_unary_function) _project_row_entry,
			GrB_FP64, GrB_BOOL, pgtm_node_ctx_type));
}

static inline void _ensure_pgtm_ops
(
	void
) {
	pthread_mutex_lock(&pgtm_ops_mutex);

	if(pgtm_edge_ctx_type != NULL &&
	   pgtm_node_ctx_type != NULL &&
	   pgtm_edge_project_op != NULL &&
	   pgtm_row_project_op != NULL) {
		size_t t_size = 0;
		if(GxB_Type_size(&t_size, pgtm_edge_ctx_type) == GrB_SUCCESS) {
			pthread_mutex_unlock(&pgtm_ops_mutex);
			return;
		}

		// GraphBLAS can be finalized/reinitialized in unit tests; stale handles
		// are invalid after finalize and must be recreated.
		pgtm_edge_ctx_type = NULL;
		pgtm_node_ctx_type = NULL;
		pgtm_edge_project_op = NULL;
		pgtm_row_project_op = NULL;
	}

	_init_pgtm_ops();
	ASSERT(pgtm_edge_ctx_type != NULL);
	ASSERT(pgtm_node_ctx_type != NULL);
	ASSERT(pgtm_edge_project_op != NULL);
	ASSERT(pgtm_row_project_op != NULL);

	pthread_mutex_unlock(&pgtm_ops_mutex);
}

static inline void _mark_invalid
(
	atomic_bool *invalid
) {
	if(invalid == NULL) return;
	atomic_store_explicit(invalid, true, memory_order_relaxed);
}

static bool _resolve_default
(
	SIValue v,              // input default SIValue
	bool *has_default,      // [output] whether default was provided
	double *default_value   // [output] parsed default value
) {
	ASSERT(has_default != NULL);
	ASSERT(default_value != NULL);

	if(SIValue_IsNull(v)) {
		*has_default = false;
		*default_value = 0;
		return true;
	}

	if((SI_TYPE(v) & SI_NUMERIC) == 0) {
		return false;
	}

	return (SIValue_ToDouble(&v, default_value) == 1) &&
		   ((*has_default = true), true);
}

static bool _resolve_edge_weight
(
	double *w,                  // [output] resolved edge weight
	EdgeID id,                  // edge id
	const edge_project_ctx *ctx // projection context
) {
	ASSERT(w != NULL);
	ASSERT(ctx != NULL);

	if(ctx->attr_id == ATTRIBUTE_ID_NONE) {
		*w = ctx->default_value;
		return true;
	}

	Edge e;
	if (!Graph_GetEdge (ctx->g, id, &e)) return false;

	SIValue v;
	if(GraphEntity_GetProperty ((GraphEntity *) &e, ctx->attr_id, &v) &&
	   (SI_TYPE(v) & SI_NUMERIC) &&
	   SIValue_ToDouble(&v, w) == 1) {
		return true;
	}

	if(ctx->has_default) {
		*w = ctx->default_value;
		return true;
	}

	return false;
}

static bool _resolve_node_weight
(
	double *w,                  // [output] resolved node weight
	NodeID id,                  // node id
	const node_project_ctx *ctx // projection context
) {
	ASSERT(w != NULL);
	ASSERT(ctx != NULL);

	if(ctx->attr_id == ATTRIBUTE_ID_NONE) {
		*w = ctx->default_value;
		return true;
	}

	Node n;
	if(!Graph_GetNode(ctx->g, id, &n)) return false;

	SIValue v;
	if(GraphEntity_GetProperty((GraphEntity *)&n, ctx->attr_id, &v) &&
	   (SI_TYPE(v) & SI_NUMERIC) &&
	   SIValue_ToDouble(&v, w) == 1) {
		return true;
	}

	if(ctx->has_default) {
		*w = ctx->default_value;
		return true;
	}

	return false;
}

// convert relation-matrix entries (edge ids or tensor vectors of edge ids)
// into projected edge weights according to projection strategy
static void _project_edge_entry
(
	double *z,                   // [output] projected edge value
	const void *x,               // relation matrix entry (edge id/tensor)
	GrB_Index i,                 // row index [unused]
	GrB_Index j,                 // col index [unused]
	const edge_project_ctx *ctx  // projection context
) {
	UNUSED (i);
	UNUSED (j);

	uint64_t entry = *(const uint64_t *)x;
	double w = 0;

	if(SCALAR_ENTRY(entry)) {
		if(_resolve_edge_weight(&w, (EdgeID)entry, ctx)) {
			*z = w;
		} else {
			_mark_invalid(ctx->invalid);
			*z = 0;
		}
		return;
	}

	GrB_Vector ids = AS_VECTOR(entry);
	struct GB_Iterator_opaque _it;
	GxB_Iterator it = &_it;

	GrB_OK (GxB_Vector_Iterator_attach(it, ids, NULL));

	GrB_Info info;
	info = GxB_Vector_Iterator_seek(it, 0);
	if(info == GxB_EXHAUSTED) {
		_mark_invalid(ctx->invalid);
		*z = 0;
		return;
	}
	ASSERT(info == GrB_SUCCESS);

	EdgeID id = (EdgeID) GxB_Vector_Iterator_getIndex(it);
	bool ok = _resolve_edge_weight(&w, id, ctx);

	// PROJECT_TO_ANY uses only the first encountered edge.
	if(ctx->strategy == PROJECT_TO_ANY || !ok) {
		if(!ok) {
			_mark_invalid(ctx->invalid);
			*z = 0;
		} else {
			*z = w;
		}
		return;
	}

	double best = w;
	info = GxB_Vector_Iterator_next(it);
	while(info != GxB_EXHAUSTED) {
		id = (EdgeID)GxB_Vector_Iterator_getIndex(it);
		ok = _resolve_edge_weight(&w, id, ctx);
		if(!ok) {
			_mark_invalid(ctx->invalid);
			*z = 0;
			return;
		}

		if(ctx->strategy == PROJECT_TO_MIN) {
			if(w < best) best = w;
		} else {
			if(w > best) best = w;
		}

		info = GxB_Vector_Iterator_next(it);
	}

	*z = best;
}

// convert selected rows into weighted row values when a node attribute/default
// is configured
static void _project_row_entry
(
	double *z,                   // [output] projected node value
	const void *x,               // row entry [unused]
	GrB_Index i,                 // row index (node id)
	GrB_Index j,                 // col index [unused]
	const node_project_ctx *ctx  // projection context
) {
	UNUSED (x);
	UNUSED (j);

	double w = 0;
	bool ok = _resolve_node_weight(&w, (NodeID) i, ctx);
	if(!ok) {
		_mark_invalid(ctx->invalid);
		*z = 0;
		return;
	}

	*z = w;
}

// get a boolean vector with each selected row
static GrB_Info _get_rows_with_labels
(
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls    // number of labels
) {
	ASSERT(rows != NULL);
	ASSERT(g != NULL);

	GrB_Info info;
	GrB_Index n = Graph_RequiredMatrixDim(g);
	GrB_Index n_short = Graph_UncompactedNodeCount(g);
	GrB_Vector _rows = NULL;

	if(n_lbls > 0) {
		GrB_OK (GrB_Vector_new(&_rows, GrB_BOOL, n));

		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);
		ASSERT(DL != NULL);

		GrB_Matrix L = NULL;
		GrB_OK (Delta_Matrix_export(&L, DL, GrB_BOOL, NULL));

		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);
			ASSERT(DL != NULL);

			GrB_Matrix M = NULL;
			GrB_OK (Delta_Matrix_export(&M, DL, GrB_BOOL, NULL));
			// L = L U M
			GrB_OK (GrB_Matrix_assign (
				L, NULL, GrB_ONEB_BOOL, M, GrB_ALL, 0, GrB_ALL, 0, NULL));
			GrB_OK (GrB_Matrix_free(&M));
		}

		GrB_OK (GxB_Vector_diag(_rows, L, 0, NULL));
		GrB_OK (GrB_Matrix_free(&L));
	} else {
		GrB_OK (GrB_Vector_new(&_rows, GrB_BOOL, n_short));
		GrB_OK (GrB_Vector_assign_BOOL(_rows, NULL, NULL, true, GrB_ALL, n_short,
				NULL));

		if (Graph_DeletedNodeCount(g) > 0) {
			NodeID *deleted_nodes = NULL;
			uint64_t deleted_count = 0;
			Graph_DeletedNodes(g, &deleted_nodes, &deleted_count);

			for(uint64_t i = 0; i < deleted_count; i++) {
				GrB_OK (GrB_Vector_removeElement(_rows, deleted_nodes[i]));
			}
			rm_free(deleted_nodes);
		}
	}

	GrB_OK (GrB_Vector_resize(_rows, n_short));
	*rows = _rows;
	return GrB_SUCCESS;
}

static GrB_Info _collect_relation_matrices
(
	const Graph *g,         // graph
	const RelationID *rels, // [optional] relation ids
	unsigned short n_rels,  // relation count
	Delta_Matrix **mats,    // [output] collected relation matrices
	unsigned short *count   // [output] number of collected matrices
) {
	ASSERT(g != NULL);
	ASSERT(mats != NULL);
	ASSERT(count != NULL);

	unsigned short rel_count = n_rels;
	if(rels == NULL) rel_count = Graph_RelationTypeCount(g);

	if(rel_count == 0) {
		*mats = NULL;
		*count = 0;
		return GrB_SUCCESS;
	}

	Delta_Matrix *R = rm_malloc(sizeof(Delta_Matrix) * rel_count);
	for(unsigned short i = 0; i < rel_count; i++) {
		RelationID rel_id = (rels == NULL) ? i : rels[i];
		R[i] = Graph_GetRelationMatrix(g, rel_id, false);
		ASSERT(R[i] != NULL);
	}

	*mats = R;
	*count = rel_count;
	return GrB_SUCCESS;
}

static GrB_BinaryOp _select_reduce_op
(
	project_strategy strategy
) {
	switch(strategy) {
		case PROJECT_TO_MIN: return GrB_MIN_FP64;
		case PROJECT_TO_MAX: return GrB_MAX_FP64;
		case PROJECT_TO_ANY: return GxB_ANY_FP64;
		default: ASSERT(false); return GxB_ANY_FP64;
	}
}

GrB_Info _combine_matricies_and_extract
(
	GrB_Matrix *A,                    // [output] matrix
	const Delta_Matrix *mats,         // matricies to consider
	unsigned short n_mats,            // number of matricies
	const GrB_Vector rows,            // filtered rows
	GrB_Type out_type,                // output matrix type
	const GrB_BinaryOp op,            // addition op
	const GrB_IndexUnaryOp value_op,  // gets value from entry
	                                  // leave NULL to assign theta
	const GrB_Scalar thunk            // thunk for the value op
) {
	ASSERT(A != NULL);
	ASSERT(mats != NULL);
	ASSERT(n_mats > 0);

	GrB_Info info;
	GrB_Index nvals;
	GrB_Matrix _A = NULL;
	GrB_Matrix am = NULL;
	GrB_Matrix adp = NULL;
	GrB_Matrix adm = NULL;
	GrB_Matrix projected = NULL;
	GrB_Descriptor desc = NULL;

	GrB_OK (GrB_Vector_nvals(&nvals, rows));

	GrB_OK (GrB_Descriptor_new(&desc));
	GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
	GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));

	// DP/M carry edge IDs (possibly tensor-encoded), so keep extraction matrices
	// as UINT64 regardless of output mode.
	GrB_OK (GrB_Matrix_new(&_A, out_type, nvals, nvals));
	GrB_OK (GrB_Matrix_new(&adp, GrB_UINT64, nvals, nvals));
	GrB_OK (GrB_Matrix_new(&am, GrB_UINT64, nvals, nvals));
	GrB_OK (GrB_Matrix_new(&adm, GrB_BOOL, nvals, nvals));

	if(value_op != NULL) {
		GrB_OK (GrB_Matrix_new(&projected, out_type, nvals, nvals));
	}

	for(unsigned short i = 0; i < n_mats; i++) {
		GrB_Matrix m  = Delta_Matrix_M(mats[i]);
		GrB_Matrix dm = Delta_Matrix_DM(mats[i]);
		GrB_Matrix dp = Delta_Matrix_DP(mats[i]);

		// accumulate pending additions (DP) in the selected sub-domain
		GrB_OK (GxB_Matrix_extract_Vector(adp, NULL, NULL, dp, rows, rows, desc));
		if(value_op != NULL) {
			GrB_OK (GrB_Matrix_apply_IndexOp_Scalar (
				projected, NULL, NULL, value_op, adp, thunk, NULL)) ;
			GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (
				_A, NULL, NULL, op, _A, projected, NULL)) ;
		} else {
			GrB_OK (GrB_assign (
				_A, adp, NULL, thunk, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S)) ;
		}

		GrB_OK (GrB_Matrix_clear(adp));

		if(value_op != NULL) {
			GrB_OK (GrB_Matrix_clear(projected));
		}

		// accumulate committed M entries that are not deleted by DM
		GrB_OK (GxB_Matrix_extract_Vector(adm, NULL, NULL, dm, rows, rows, desc));

		GrB_OK (GrB_Descriptor_set_INT32(desc, GrB_COMP_STRUCTURE, GrB_MASK_FIELD));

		GrB_OK (GxB_Matrix_extract_Vector(am, adm, NULL, m, rows, rows, desc));

		if(value_op != NULL) {
			GrB_OK (GrB_Matrix_apply_IndexOp_Scalar(
				projected, NULL, NULL, value_op, am, thunk, NULL));
			GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (
				_A, NULL, NULL, op, _A, projected, NULL));
		} else {
			GrB_OK (GrB_assign (
				_A, am, NULL, thunk, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
		}

		GrB_OK (GrB_Descriptor_set_INT32(desc, GrB_DEFAULT, GrB_MASK_FIELD));

		GrB_OK (GrB_Matrix_clear(am));
		GrB_OK (GrB_Matrix_clear(adm));
		if(value_op != NULL) {
			GrB_OK (GrB_Matrix_clear(projected));
		}
	}

	*A = _A;
	_A = NULL;
	info = GrB_SUCCESS;

cleanup:
	GrB_free(&adp);
	GrB_free(&am);
	GrB_free(&adm);
	GrB_free(&projected);
	GrB_free(&desc);
	GrB_free(&_A);
	return info;
}

static GrB_Info _symmetrize_matrix
(
	GrB_Matrix A,
	bool bool_matrix,
	project_strategy strategy
) {
	ASSERT(A != NULL);

	if(bool_matrix) {
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (
			A, NULL, NULL, GxB_ANY_BOOL, A, A, GrB_DESC_T1));
	} else {
		GrB_BinaryOp op = _select_reduce_op(strategy);
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (
			A, NULL, NULL, op, A, A, GrB_DESC_T1));
	}

	return GrB_SUCCESS;
}

static GrB_Info _expand_to_full_domain
(
	GrB_Matrix *A,           // [input/output] compact matrix to expand
	const Graph *g,          // graph
	const GrB_Vector rows,   // row-id map used during compaction
	bool bool_matrix         // matrix type switch
) {
	ASSERT(A != NULL && *A != NULL);
	ASSERT(g != NULL);
	ASSERT(rows != NULL);

	GrB_Index n = Graph_UncompactedNodeCount(g);
	GrB_Matrix expanded = NULL;
	GrB_OK (GrB_Matrix_new(&expanded, bool_matrix ? GrB_BOOL : GrB_FP64, n, n));

	GrB_Index k = 0;
	GrB_OK (GrB_Vector_nvals(&k, rows));
	GrB_Descriptor desc = NULL;
	GrB_OK (GrB_Descriptor_new (&desc));
	GrB_OK (GrB_set (desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
	GrB_OK (GrB_set (desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));

	GrB_OK (GxB_Matrix_assign_Vector(expanded, NULL, NULL, *A, rows, rows, desc));

	GrB_free(A);
	*A = expanded;
	return GrB_SUCCESS;
}

// Make a matrix out of a graph, given an input configuration object
GrB_Info project_graph_to_matrix
(
	GrB_Matrix *A,     // [output] matrix weights
	GrB_Vector *rows,  // [optional output] filtered rows
	PGTM_config conf   // input configuration
) {
	ASSERT(conf.g != NULL);
	ASSERT(A != NULL);
	ASSERT((conf.lbls != NULL && conf.n_lbls > 0) || (conf.lbls == NULL && conf.n_lbls == 0));
	ASSERT((conf.rels != NULL && conf.n_rels > 0) || (conf.rels == NULL && conf.n_rels == 0));
	_ensure_pgtm_ops();

	// Compact projection remaps matrix indices to the selected rows domain.
	// Callers that don't consume `rows` cannot map compact indices back.
	if(conf.compact && rows == NULL) {
		conf.compact = false;
	}

	GrB_Matrix _A = NULL;
	GrB_Vector _rows = NULL;
	GrB_Vector _rows_out = NULL;
	Delta_Matrix *R = NULL;
	unsigned short n_rel_mats = 0;
	GrB_Info info = GrB_SUCCESS;

	bool edge_has_default = false;
	bool node_has_default = false;
	double edge_default = 0;
	double node_default = 0;

	if(!_resolve_default(conf.default_ew, &edge_has_default, &edge_default)) {
		return GrB_INVALID_VALUE;
	}

	if(!_resolve_default(conf.default_nw, &node_has_default, &node_default)) {
		return GrB_INVALID_VALUE;
	}

	bool bool_matrix = (conf.edge_weight == ATTRIBUTE_ID_NONE && !edge_has_default);
	bool bool_rows = (conf.node_weight == ATTRIBUTE_ID_NONE && !node_has_default);

	GrB_OK (_get_rows_with_labels(&_rows, conf.g, conf.lbls, conf.n_lbls));
	GrB_OK (_collect_relation_matrices(conf.g, conf.rels, conf.n_rels, &R,
			&n_rel_mats));

	if(n_rel_mats == 0) {
		GrB_Index n = 0;
		GrB_OK (GrB_Vector_nvals(&n, _rows));
		GrB_Index dim = conf.compact ? n : Graph_UncompactedNodeCount(conf.g);
		GrB_OK (GrB_Matrix_new(&_A, bool_matrix ? GrB_BOOL : GrB_FP64, dim, dim));
	} else if(bool_matrix) {
		GrB_Scalar t = NULL;
		GrB_OK (GrB_Scalar_new(&t, GrB_BOOL));
		GrB_OK (GrB_Scalar_setElement_BOOL(t, true));
		GrB_OK (_combine_matricies_and_extract(&_A, R, n_rel_mats, _rows,
				GrB_BOOL, GxB_ANY_BOOL, NULL, t));
		GrB_OK (GrB_free(&t));
	} else {
		atomic_bool invalid_edges = false;
		edge_project_ctx ectx = {
			.g = conf.g,
			.attr_id = conf.edge_weight,
			.default_value = edge_default,
			.has_default = edge_has_default,
			.strategy = conf.strategy,
			.invalid = &invalid_edges
		};

		GrB_Scalar ectx_s = NULL;
		GrB_OK (GrB_Scalar_new(&ectx_s, pgtm_edge_ctx_type));
		GrB_OK (GrB_Scalar_setElement_UDT(ectx_s, (void *)&ectx));

		GrB_BinaryOp op = _select_reduce_op(conf.strategy);
		GrB_OK (_combine_matricies_and_extract(&_A, R, n_rel_mats, _rows,
				GrB_FP64, op,
				pgtm_edge_project_op, ectx_s));

		GrB_OK (GrB_free(&ectx_s));

		if(atomic_load_explicit(&invalid_edges, memory_order_relaxed)) {
			info = GrB_INVALID_VALUE;
			goto cleanup;
		}
	}

	if(conf.direction == GRAPH_EDGE_DIR_INCOMING) {
		GrB_OK (GrB_transpose(_A, NULL, NULL, _A, NULL));
	} else if(conf.direction == GRAPH_EDGE_DIR_BOTH) {
		GrB_OK (_symmetrize_matrix(_A, bool_matrix, conf.strategy));
	}

	if(!conf.compact) {
		GrB_OK (_expand_to_full_domain(&_A, conf.g, _rows, bool_matrix));
	}

	if(rows != NULL) {
		if(bool_rows) {
			*rows = _rows;
			_rows = NULL;
		} else {
			atomic_bool invalid_nodes = false;
			node_project_ctx nctx = {
				.g = conf.g,
				.attr_id = conf.node_weight,
				.default_value = node_default,
				.has_default = node_has_default,
				.invalid = &invalid_nodes
			};

			GrB_Scalar nctx_s = NULL;
			GrB_OK (GrB_Scalar_new(&nctx_s, pgtm_node_ctx_type));
			GrB_OK (GrB_Scalar_setElement_UDT(nctx_s, (void *)&nctx));

			GrB_Index n = Graph_UncompactedNodeCount(conf.g);
			GrB_OK (GrB_Vector_new(&_rows_out, GrB_FP64, n));
			GrB_OK (GrB_Vector_apply_IndexOp_Scalar(_rows_out, NULL, NULL,
					pgtm_row_project_op,
					_rows, nctx_s, NULL));

			GrB_OK (GrB_free(&nctx_s));

			if (atomic_load_explicit (&invalid_nodes, memory_order_relaxed)) {
				info = GrB_INVALID_VALUE;
				goto cleanup;
			}

			*rows = _rows_out;
			_rows_out = NULL;
		}
	}

	*A = _A;
	_A = NULL;

cleanup:
	GrB_free(&_A);
	GrB_free(&_rows);
	GrB_free(&_rows_out);
	rm_free(R);
	return info;
}
