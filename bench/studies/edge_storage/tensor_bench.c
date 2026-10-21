// Standalone micro-benchmark harness for the C FalkorDB Tensor.
//
// Measures per-operation instruction counts using proc_pid_rusage(
// RUSAGE_INFO_V4).ri_instructions (macOS), amortised over many repetitions.
//
// Build: see build.sh next to this file.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc/malloc.h>
#include <libproc.h>
#include <unistd.h>
#include <time.h>

#include "RG.h"
#include "util/rmalloc.h"
#include "configuration/config.h"
#include "graph/tensor/tensor.h"
#include "graph/delta_matrix/delta_matrix.h"
#include "graph/delta_matrix/delta_matrix_iter.h"

// ---------------------------------------------------------------------------
// instruction counter
// ---------------------------------------------------------------------------

// rusage_info_v4: 16-byte ri_uuid followed by u64 fields.
// ri_instructions is u64 index 29 counting from byte 16 (see
// bench/src/falkorbench/counters.py read_rusage).
#define RI_INSTRUCTIONS_OFF (16 + 29 * 8)
#define RI_CYCLES_OFF       (16 + 30 * 8)

static uint64_t read_instructions(void) {
	static uint8_t buf[1024];
	if(proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)buf) != 0) {
		fprintf(stderr, "proc_pid_rusage failed\n");
		exit(1);
	}
	uint64_t v;
	memcpy(&v, buf + RI_INSTRUCTIONS_OFF, sizeof(v));
	return v;
}

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// keep the optimiser from deleting the measured work
static volatile uint64_t g_sink = 0;

// ---------------------------------------------------------------------------
// tensor construction helpers
// ---------------------------------------------------------------------------

// number of (src,dst) pairs
static uint64_t N = 200000;

// build a tensor with `edges_per_pair` edges on the diagonal-ish pairs
// pair k is (k, k) ; edge ids are dense and unique
// uses the batch entry point: one-at-a-time Tensor_SetElement is O(pending)
// per call (see the note on the 1-at-a-time benchmark) and would make setup
// quadratic.
static Tensor build(uint64_t n, int edges_per_pair) {
	Tensor T = Tensor_new(n + 1, n + 1);
	uint64_t m = n * edges_per_pair;
	GrB_Index *rows = malloc(m * sizeof(GrB_Index));
	GrB_Index *cols = malloc(m * sizeof(GrB_Index));
	uint64_t  *vals = malloc(m * sizeof(uint64_t));
	uint64_t k = 0;
	for(uint64_t i = 0; i < n; i++) {
		for(int e = 0; e < edges_per_pair; e++) {
			rows[k] = i; cols[k] = i; vals[k] = k; k++;
		}
	}
	Tensor_SetElements(T, rows, cols, vals, m);
	free(rows); free(cols); free(vals);
	// flush pending changes so reads hit M, not delta-plus
	GrB_Info info = Delta_Matrix_wait(T, true);
	if(info != GrB_SUCCESS) { fprintf(stderr, "wait failed %d\n", info); exit(1); }
	return T;
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

typedef struct {
	const char *name;
	double instr_per_op[3];
	double sec_per_op[3];
	uint64_t ops;
} Result;

static void report(const Result *r) {
	double mn = r->instr_per_op[0], mx = r->instr_per_op[0], sum = 0;
	for(int i = 0; i < 3; i++) {
		if(r->instr_per_op[i] < mn) mn = r->instr_per_op[i];
		if(r->instr_per_op[i] > mx) mx = r->instr_per_op[i];
		sum += r->instr_per_op[i];
	}
	double avg = sum / 3.0;
	printf("%-46s ops=%-9llu instr/op: %8.1f %8.1f %8.1f  (mean %8.1f, spread %+.2f%%)  ns/op %.1f\n",
	       r->name, (unsigned long long)r->ops,
	       r->instr_per_op[0], r->instr_per_op[1], r->instr_per_op[2],
	       avg, 100.0 * (mx - mn) / avg,
	       1e9 * (r->sec_per_op[0] + r->sec_per_op[1] + r->sec_per_op[2]) / 3.0);
	fflush(stdout);
}

// ---- point read -----------------------------------------------------------

// point read: extract the cell value at T[i,i].  This is what
// Tensor_SetElement / Tensor_RemoveElements / traversal all do first.
static void bench_point_read(Tensor T, const char *name, uint64_t reps) {
	Result r = { .name = name, .ops = reps };
	for(int rep = 0; rep < 3; rep++) {
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		uint64_t acc = 0;
		for(uint64_t k = 0; k < reps; k++) {
			uint64_t x;
			GrB_Index p = k % N;
			GrB_Info info = Delta_Matrix_extractElement_UINT64(&x, T, p, p);
			if(info != GrB_SUCCESS) { fprintf(stderr, "%s: miss at %llu\n", name, (unsigned long long)p); exit(1); }
			acc += x;
		}
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		g_sink += acc;
		r.instr_per_op[rep] = (double)(i1 - i0) / reps;
		r.sec_per_op[rep]   = (t1 - t0) / reps;
	}
	report(&r);
}

// point read that also materialises the edge id list, i.e. what a traversal
// actually pays: read the cell, then walk the cell's edges.
static void bench_point_read_edges(Tensor T, const char *name, uint64_t reps) {
	Result r = { .name = name, .ops = reps };
	for(int rep = 0; rep < 3; rep++) {
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		uint64_t acc = 0;
		for(uint64_t k = 0; k < reps; k++) {
			GrB_Index p = k % N;
			TensorIterator it;
			TensorIterator_ScanEntry(&it, T, p, p);
			uint64_t x;
			while(TensorIterator_next(&it, NULL, NULL, &x, NULL)) acc += x;
		}
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		g_sink += acc;
		r.instr_per_op[rep] = (double)(i1 - i0) / reps;
		r.sec_per_op[rep]   = (t1 - t0) / reps;
	}
	report(&r);
}

// baseline: the same point read straight against the underlying GrB_Matrix M,
// i.e. without the delta layer's DP / DM probes.
static void bench_raw_read(Tensor T, const char *name, uint64_t reps) {
	GrB_Matrix M = Delta_Matrix_M(T);
	Result r = { .name = name, .ops = reps };
	for(int rep = 0; rep < 3; rep++) {
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		uint64_t acc = 0;
		for(uint64_t k = 0; k < reps; k++) {
			uint64_t x;
			GrB_Index p = k % N;
			GrB_Info info = GrB_Matrix_extractElement_UINT64(&x, M, p, p);
			if(info != GrB_SUCCESS) { fprintf(stderr, "%s: miss\n", name); exit(1); }
			acc += x;
		}
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		g_sink += acc;
		r.instr_per_op[rep] = (double)(i1 - i0) / reps;
		r.sec_per_op[rep]   = (t1 - t0) / reps;
	}
	report(&r);
}

// count how many cells are inline scalars vs tagged container pointers
static void verify_cells(Tensor T, const char *name) {
	Delta_MatrixTupleIter it;
	Delta_MatrixTupleIter_attach(&it, T);
	uint64_t x, scalars = 0, vectors = 0, edges = 0;
	while(Delta_MatrixTupleIter_next_UINT64(&it, NULL, NULL, &x) == GrB_SUCCESS) {
		if(SCALAR_ENTRY(x)) { scalars++; edges++; }
		else {
			vectors++;
			GrB_Index nv;
			GrB_Vector_nvals(&nv, AS_VECTOR(x));
			edges += nv;
		}
	}
	Delta_MatrixTupleIter_detach(&it);
	printf("%-46s cells: %llu inline scalar, %llu container, %llu edges total\n",
	       name, (unsigned long long)scalars, (unsigned long long)vectors,
	       (unsigned long long)edges);
	fflush(stdout);
}

// same census, printed once (first rep only), used to prove which build path
// ends up with containers
static void verify_containers_once(Tensor T, int rep, const char *tag) {
	if(rep != 0) return;
	char name[80];
	snprintf(name, sizeof(name), "  [%s result]", tag);
	verify_cells(T, name);
}

// ---- full iteration -------------------------------------------------------

static void bench_iterate(Tensor T, const char *name, uint64_t expect_edges,
                          uint64_t passes) {
	Result r = { .name = name, .ops = expect_edges * passes };
	for(int rep = 0; rep < 3; rep++) {
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		uint64_t seen = 0, acc = 0;
		for(uint64_t p = 0; p < passes; p++) {
			TensorIterator it;
			TensorIterator_ScanRange(&it, T, 0, N, false);
			uint64_t x;
			while(TensorIterator_next(&it, NULL, NULL, &x, NULL)) { acc += x; seen++; }
		}
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		g_sink += acc;
		if(seen != expect_edges * passes) {
			fprintf(stderr, "%s: saw %llu edges, expected %llu\n", name,
			        (unsigned long long)seen, (unsigned long long)(expect_edges * passes));
			exit(1);
		}
		r.instr_per_op[rep] = (double)(i1 - i0) / (expect_edges * passes);
		r.sec_per_op[rep]   = (t1 - t0) / (expect_edges * passes);
	}
	report(&r);
}

// transposed iteration: TensorIterator_ScanRange with transposed = true walks
// the backward matrix, so its range selects by destination. The Rust side pays
// a forward eff_get per incoming pair here (it stores no ids in `mt`); the C
// side stores the same tagged cells both ways, so this is the row that says
// what that trade costs relative to a design that duplicates.
static void bench_iterate_t(Tensor T, const char *name, uint64_t expect_edges,
                            uint64_t passes) {
	Result r = { .name = name, .ops = expect_edges * passes };
	for(int rep = 0; rep < 3; rep++) {
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		uint64_t seen = 0, acc = 0;
		for(uint64_t p = 0; p < passes; p++) {
			TensorIterator it;
			TensorIterator_ScanRange(&it, T, 0, N, true);
			uint64_t x;
			while(TensorIterator_next(&it, NULL, NULL, &x, NULL)) { acc += x; seen++; }
		}
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		g_sink += acc;
		if(seen != expect_edges * passes) {
			fprintf(stderr, "%s: saw %llu edges, expected %llu\n", name,
			        (unsigned long long)seen, (unsigned long long)(expect_edges * passes));
			exit(1);
		}
		r.instr_per_op[rep] = (double)(i1 - i0) / (expect_edges * passes);
		r.sec_per_op[rep]   = (t1 - t0) / (expect_edges * passes);
	}
	report(&r);
}

// ---- promotion / demotion -------------------------------------------------

// promotion: each measured op inserts a 2nd edge into a single-edge pair,
// which allocates a GrB_Vector container.  Demotion: removing that edge again
// frees the container and writes the surviving edge id back inline.
//
// promotion and demotion must alternate to keep the tensor in a steady state,
// so we measure a promote+demote pair and also measure them separately by
// building fresh single-edge tensors per rep.
static void bench_promote_demote(uint64_t n) {
	// --- container cost in isolation: what promotion pays on top of the
	//     delta-matrix work -- allocate a GrB_Vector, put 2 edge ids in it,
	//     materialise it, free it
	{
		Result rc = { .name = "container only: new+2 set+wait+free", .ops = n };
		for(int rep = 0; rep < 3; rep++) {
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < n; i++) {
				GrB_Vector V;
				GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
				GrB_Vector_setElement_BOOL(V, true, 2 * i);
				GrB_Vector_setElement_BOOL(V, true, 2 * i + 1);
				GrB_wait(V, GrB_MATERIALIZE);
				GrB_free(&V);
			}
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rc.instr_per_op[rep] = (double)(i1 - i0) / n;
			rc.sec_per_op[rep]   = (t1 - t0) / n;
		}
		report(&rc);
	}

	// --- appending one more id to an ALREADY-MATERIALISED container, in
	//     isolation from any delta-matrix work. This is the GraphBLAS-level
	//     explanation for why the "add an edge to an already-multi-edge pair"
	//     control comes out MORE expensive than the promotion it controls for:
	//     GrB_wait on a vector that already holds entries rebuilds the whole
	//     vector, whereas a fresh vector's first wait only builds the pendings.
	for(int have = 2; have <= 4; have += 2) {
		char name[80];
		snprintf(name, sizeof(name), "container only: append 1 id to %d-id cntnr", have);
		Result ra = { .name = name, .ops = n };
		for(int rep = 0; rep < 3; rep++) {
			// pre-build the containers so the measured region is only the append
			GrB_Vector *vs = malloc(n * sizeof(GrB_Vector));
			for(uint64_t i = 0; i < n; i++) {
				GrB_Vector_new(&vs[i], GrB_BOOL, GrB_INDEX_MAX);
				for(int e = 0; e < have; e++)
					GrB_Vector_setElement_BOOL(vs[i], true, 10 * i + e);
				GrB_wait(vs[i], GrB_MATERIALIZE);
			}
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < n; i++) {
				GrB_Vector_setElement_BOOL(vs[i], true, 10 * i + have);
				GrB_wait(vs[i], GrB_MATERIALIZE);
			}
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			ra.instr_per_op[rep] = (double)(i1 - i0) / n;
			ra.sec_per_op[rep]   = (t1 - t0) / n;
			for(uint64_t i = 0; i < n; i++) GrB_free(&vs[i]);
			free(vs);
		}
		report(&ra);
	}

	// --- batch insert of n brand-new inline (single-edge) pairs, the path the
	//     engine actually uses for bulk creation
	{
		Result rb = { .name = "batch insert n new inline pairs (per edge)", .ops = n };
		GrB_Index *rows = malloc(n * sizeof(GrB_Index));
		GrB_Index *cols = malloc(n * sizeof(GrB_Index));
		uint64_t  *vals = malloc(n * sizeof(uint64_t));
		for(uint64_t i = 0; i < n; i++) { rows[i] = cols[i] = i; vals[i] = i; }
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = Tensor_new(n + 1, n + 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			Tensor_SetElements(T, rows, cols, vals, n);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rb.instr_per_op[rep] = (double)(i1 - i0) / n;
			rb.sec_per_op[rep]   = (t1 - t0) / n;
			Tensor_free(&T);
		}
		report(&rb);
		free(rows); free(cols); free(vals);
	}

	// --- batch promotion: add a 2nd edge to every one of n single-edge pairs
	{
		Result rb = { .name = "batch promote n pairs (per promotion)", .ops = n };
		GrB_Index *rows = malloc(n * sizeof(GrB_Index));
		GrB_Index *cols = malloc(n * sizeof(GrB_Index));
		uint64_t  *vals = malloc(n * sizeof(uint64_t));
		for(uint64_t i = 0; i < n; i++) { rows[i] = cols[i] = i; vals[i] = n + i; }
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = build(n, 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			Tensor_SetElements(T, rows, cols, vals, n);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rb.instr_per_op[rep] = (double)(i1 - i0) / n;
			rb.sec_per_op[rep]   = (t1 - t0) / n;
			Tensor_free(&T);
		}
		report(&rb);
		free(rows); free(cols); free(vals);
	}

	// --- batch demotion: remove the 2nd edge of every one of n 2-edge pairs
	{
		Result rb = { .name = "batch demote n pairs (per demotion)", .ops = n };
		Edge *es = calloc(n, sizeof(Edge));
		for(uint64_t i = 0; i < n; i++) {
			es[i].src_id = i; es[i].dest_id = i; es[i].id = 2 * i + 1;
		}
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = build(n, 2);
			// Tensor_RemoveElements sorts the range in place; restore each rep
			for(uint64_t i = 0; i < n; i++) {
				es[i].src_id = i; es[i].dest_id = i; es[i].id = 2 * i + 1;
			}
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			Tensor_RemoveElements(T, es, n, NULL);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rb.instr_per_op[rep] = (double)(i1 - i0) / n;
			rb.sec_per_op[rep]   = (t1 - t0) / n;
			Tensor_free(&T);
		}
		report(&rb);
		free(es);
	}

	// --- reference point: inserting brand-new inline edges ONE AT A TIME.
	//     NOTE: this is O(pending) per call, not constant -- Tensor_SetElement
	//     reads T[row,col] first, and that read forces GraphBLAS to materialise
	//     delta-plus's pending tuples, so the i-th insert flushes i tuples.
	//     Reported per-op numbers therefore grow linearly with n.
	//     measured at two sizes to expose the linear growth
	for(uint64_t nn = 1000; nn <= 4000; nn *= 4) {
		char name[80];
		snprintf(name, sizeof(name), "insert 1-at-a-time, n=%-5llu (O(n)/op!)",
		         (unsigned long long)nn);
		Result ri = { .name = name, .ops = nn };
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = Tensor_new(nn + 1, nn + 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < nn; i++) Tensor_SetElement(T, i, i, i);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			ri.instr_per_op[rep] = (double)(i1 - i0) / nn;
			ri.sec_per_op[rep]   = (t1 - t0) / nn;
			Tensor_free(&T);
		}
		report(&ri);
	}

	// --- promotion: fresh all-single-edge tensor per rep, promote every pair
	{
		Result rp = { .name = "promote  single->2-edge (alloc container)", .ops = n };
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = build(n, 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < n; i++) Tensor_SetElement(T, i, i, n + i);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rp.instr_per_op[rep] = (double)(i1 - i0) / n;
			rp.sec_per_op[rep]   = (t1 - t0) / n;
			Tensor_free(&T);
		}
		report(&rp);
	}

	// --- demotion: fresh all-2-edge tensor per rep, demote every pair
	{
		Result rd = { .name = "demote   2-edge->single (free container)", .ops = n };
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = build(n, 2);
			// remove the 2nd edge of every pair; ids are 2*i and 2*i+1
			Edge *e = malloc(sizeof(Edge));
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < n; i++) {
				memset(e, 0, sizeof(Edge));
				e->src_id  = i;
				e->dest_id = i;
				e->id      = 2 * i + 1;
				Tensor_RemoveElements(T, e, 1, NULL);
			}
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			rd.instr_per_op[rep] = (double)(i1 - i0) / n;
			rd.sec_per_op[rep]   = (t1 - t0) / n;
			free(e);
			Tensor_free(&T);
		}
		report(&rd);
	}
}

// ---------------------------------------------------------------------------
// transition cost vs non-transitioning control on the same code path
// ---------------------------------------------------------------------------

// helper: one Tensor_SetElements call adding one edge to every pair of a
// tensor that already has `existing` edges per pair.
// existing == 1 -> every pair promotes (inline scalar -> container)
// existing >= 2 -> every pair already has a container, nothing transitions
static double mean3(const Result *r) {
	return (r->instr_per_op[0] + r->instr_per_op[1] + r->instr_per_op[2]) / 3.0;
}

static double bench_add_one_edge(uint64_t n, int existing, const char *name) {
	Result r = { .name = name, .ops = n };
	GrB_Index *rows = malloc(n * sizeof(GrB_Index));
	GrB_Index *cols = malloc(n * sizeof(GrB_Index));
	uint64_t  *vals = malloc(n * sizeof(uint64_t));
	for(uint64_t i = 0; i < n; i++) {
		rows[i] = cols[i] = i;
		// fresh id, above every id build() handed out
		vals[i] = (uint64_t)existing * n + i;
	}
	for(int rep = 0; rep < 3; rep++) {
		Tensor T = build(n, existing);
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		Tensor_SetElements(T, rows, cols, vals, n);
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		r.instr_per_op[rep] = (double)(i1 - i0) / n;
		r.sec_per_op[rep]   = (t1 - t0) / n;
		Tensor_free(&T);
	}
	report(&r);
	free(rows); free(cols); free(vals);
	return mean3(&r);
}

// helper: one Tensor_RemoveElements call removing one edge from every pair of a
// tensor that has `existing` edges per pair.
// existing == 2 -> every pair demotes (container -> inline scalar, container freed)
// existing >= 3 -> id is removed from the container, nothing transitions
static double bench_remove_one_edge(uint64_t n, int existing, const char *name) {
	Result r = { .name = name, .ops = n };
	Edge *es = calloc(n, sizeof(Edge));
	for(int rep = 0; rep < 3; rep++) {
		Tensor T = build(n, existing);
		// build() gives pair i the ids existing*i .. existing*i+existing-1;
		// drop the last one. Tensor_RemoveElements sorts in place, so refill.
		for(uint64_t i = 0; i < n; i++) {
			memset(&es[i], 0, sizeof(Edge));
			es[i].src_id  = i;
			es[i].dest_id = i;
			es[i].id      = (uint64_t)existing * i + (existing - 1);
		}
		uint64_t i0 = read_instructions();
		double  t0  = now_sec();
		Tensor_RemoveElements(T, es, n, NULL);
		double t1 = now_sec();
		uint64_t i1 = read_instructions();
		r.instr_per_op[rep] = (double)(i1 - i0) / n;
		r.sec_per_op[rep]   = (t1 - t0) / n;
		Tensor_free(&T);
	}
	report(&r);
	free(es);
	return mean3(&r);
}

// ---------------------------------------------------------------------------
// batch vs incremental construction of the same final tensor
// ---------------------------------------------------------------------------

// both variants end with N pairs x 2 edges; the difference is whether the two
// edges of a pair arrive in the same Tensor_SetElements call.
static void bench_build_paths(uint64_t n) {
	GrB_Index *rows = malloc(2 * n * sizeof(GrB_Index));
	GrB_Index *cols = malloc(2 * n * sizeof(GrB_Index));
	uint64_t  *vals = malloc(2 * n * sizeof(uint64_t));

	// --- incremental: pass 1 = first edge of every pair, pass 2 = second
	{
		Result r = { .name = "build incremental (2 calls) per pair", .ops = n };
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = Tensor_new(n + 1, n + 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			for(uint64_t i = 0; i < n; i++) { rows[i] = cols[i] = i; vals[i] = 2 * i; }
			Tensor_SetElements(T, rows, cols, vals, n);
			for(uint64_t i = 0; i < n; i++) { rows[i] = cols[i] = i; vals[i] = 2 * i + 1; }
			Tensor_SetElements(T, rows, cols, vals, n);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			r.instr_per_op[rep] = (double)(i1 - i0) / n;
			r.sec_per_op[rep]   = (t1 - t0) / n;
			verify_containers_once(T, rep, "incremental");
			Tensor_free(&T);
		}
		report(&r);
	}

	// --- batch: both edges of a pair in one call, adjacent in the array
	{
		Result r = { .name = "build batch (1 call, 2N tuples) per pair", .ops = n };
		for(uint64_t i = 0; i < n; i++) {
			rows[2*i] = cols[2*i] = i;         vals[2*i]   = 2 * i;
			rows[2*i+1] = cols[2*i+1] = i;     vals[2*i+1] = 2 * i + 1;
		}
		for(int rep = 0; rep < 3; rep++) {
			Tensor T = Tensor_new(n + 1, n + 1);
			uint64_t i0 = read_instructions();
			double  t0  = now_sec();
			Tensor_SetElements(T, rows, cols, vals, 2 * n);
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			r.instr_per_op[rep] = (double)(i1 - i0) / n;
			r.sec_per_op[rep]   = (t1 - t0) / n;
			verify_containers_once(T, rep, "batch");
			Tensor_free(&T);
		}
		report(&r);
	}

	free(rows); free(cols); free(vals);
}

// ---- container size -------------------------------------------------------

// a standalone container holding exactly k edge ids, as the always-materialised
// design would hold them (including k == 1, which the C tensor never stores)
static void report_container_size(void) {
	printf("container: GrB_Vector (struct GB_Vector_opaque), GrB_BOOL, "
	       "dim GrB_INDEX_MAX, edge id == vector index\n");
	printf("  %-4s %-16s %-16s %s\n", "ids", "malloc_size(h)", "memoryUsage", "marginal vs k-1");
	size_t prev = 0;
	int ks[] = { 0, 1, 2, 4, 8, 16 };
	for(unsigned t = 0; t < sizeof(ks) / sizeof(ks[0]); t++) {
		int k = ks[t];
		GrB_Vector V;
		GrB_Info info = GrB_Vector_new(&V, GrB_BOOL, GrB_INDEX_MAX);
		if(info != GrB_SUCCESS) { fprintf(stderr, "vector new failed\n"); exit(1); }
		for(int i = 0; i < k; i++) GrB_Vector_setElement_BOOL(V, true, 1000 + i);
		if(k > 0) GrB_wait(V, GrB_MATERIALIZE);
		size_t mem = 0;
		GxB_Vector_memoryUsage(&mem, V);
		printf("  %-4d %-16zu %-16zu %s%zd\n", k, malloc_size(V), mem,
		       t ? "+" : " ", t ? (ssize_t)(mem - prev) : (ssize_t)0);
		prev = mem;
		GrB_free(&V);
	}
	fflush(stdout);
}

// Tensor_memoryUsage for N pairs at k ids per pair.
// NOTE: k == 1 is stored INLINE by the C tensor (no container at all), so the
// k == 1 row is not "a container holding one id" -- see the standalone table.
static void report_space_vs_k(uint64_t n) {
	printf("\ntensor space, %llu pairs, k edge ids per pair:\n",
	       (unsigned long long)n);
	printf("  %-3s %-8s %-14s %-12s %-12s %s\n", "k", "cells", "Tensor_memUsage",
	       "B/pair", "B/edge", "delta vs k-1 per pair");
	double prev_per_pair = 0;
	int ks[] = { 1, 2, 4, 8 };
	for(unsigned t = 0; t < sizeof(ks) / sizeof(ks[0]); t++) {
		int k = ks[t];
		Tensor T = build(n, k);
		size_t sz = 0;
		Tensor_memoryUsage(&sz, T);
		double per_pair = (double)sz / n;
		printf("  %-3d %-8s %-14zu %-12.2f %-12.2f %s\n", k,
		       k == 1 ? "inline" : "cntnr", sz, per_pair, (double)sz / (n * k),
		       t == 0 ? "-" : "");
		if(t > 0) {
			printf("      %+.2f B/pair vs k=%d  (%+.2f B per added id)\n",
			       per_pair - prev_per_pair, ks[t-1],
			       (per_pair - prev_per_pair) / (k - ks[t-1]));
		}
		prev_per_pair = per_pair;
		Tensor_free(&T);
	}
	fflush(stdout);
}

// ---- the remaining entry points -------------------------------------------
//
// Degrees, the flat removal path and the mask-based clear were the C-side rows
// the paper listed as unmeasured. All four are per-pair or per-row costs the
// Rust side has direct counterparts for.

static void bench_degrees(Tensor T, const char *what, uint64_t reps) {
	{
		Result r = { .name = "row degree", .ops = reps };
		char nm[96]; snprintf(nm, sizeof nm, "row degree, %s", what); r.name = nm;
		for(int rep = 0; rep < 3; rep++) {
			uint64_t i0 = read_instructions(); double t0 = now_sec();
			uint64_t acc = 0;
			for(uint64_t i = 0; i < reps; i++) acc += Tensor_RowDegree(T, i % N);
			double t1 = now_sec(); uint64_t i1 = read_instructions();
			g_sink += acc;
			r.instr_per_op[rep] = (double)(i1 - i0) / reps;
			r.sec_per_op[rep]   = (t1 - t0) / reps;
		}
		report(&r);
	}
	{
		Result r = { .name = "col degree", .ops = reps };
		char nm[96]; snprintf(nm, sizeof nm, "col degree, %s", what); r.name = nm;
		for(int rep = 0; rep < 3; rep++) {
			uint64_t i0 = read_instructions(); double t0 = now_sec();
			uint64_t acc = 0;
			for(uint64_t i = 0; i < reps; i++) acc += Tensor_ColDegree(T, i % N);
			double t1 = now_sec(); uint64_t i1 = read_instructions();
			g_sink += acc;
			r.instr_per_op[rep] = (double)(i1 - i0) / reps;
			r.sec_per_op[rep]   = (t1 - t0) / reps;
		}
		report(&r);
	}
}

// `Tensor_RemoveElements_Flat` is the fast path: it assumes every entry is a
// scalar, so it can drop whole pairs without inspecting containers. Only valid
// on an all-single-edge tensor, which is the case it exists for.
static void bench_remove_flat(uint64_t n) {
	Result r = { .name = "remove_elements_flat (per pair)", .ops = n };
	Edge *es = calloc(n, sizeof(Edge));
	for(int rep = 0; rep < 3; rep++) {
		Tensor T = build(n, 1);
		for(uint64_t i = 0; i < n; i++) {
			memset(&es[i], 0, sizeof(Edge));
			es[i].src_id = i; es[i].dest_id = i; es[i].id = i;
		}
		uint64_t i0 = read_instructions(); double t0 = now_sec();
		Tensor_RemoveElements_Flat(T, es, n);
		double t1 = now_sec(); uint64_t i1 = read_instructions();
		r.instr_per_op[rep] = (double)(i1 - i0) / n;
		r.sec_per_op[rep]   = (t1 - t0) / n;
		Tensor_free(&T);
	}
	free(es);
	report(&r);
}

// `Tensor_ClearElements` takes a mask matrix and its transpose rather than an
// edge list, so its cost is a bulk matrix op rather than per pair. Measured per
// pair cleared so it lines up with the row above.
static void bench_clear_elements(uint64_t n, int edges_per_pair) {
	Result r = { .name = "clear_elements (per pair)", .ops = n };
	char nm[96];
	snprintf(nm, sizeof nm, "clear_elements, k=%d (per pair)", edges_per_pair);
	r.name = nm;
	for(int rep = 0; rep < 3; rep++) {
		Tensor T = build(n, edges_per_pair);
		GrB_Matrix A, AT;
		GrB_Matrix_new(&A,  GrB_BOOL, n + 1, n + 1);
		GrB_Matrix_new(&AT, GrB_BOOL, n + 1, n + 1);
		for(uint64_t i = 0; i < n; i++) {
			GrB_Matrix_setElement_BOOL(A,  true, i, i);
			GrB_Matrix_setElement_BOOL(AT, true, i, i);
		}
		GrB_Matrix_wait(A,  GrB_MATERIALIZE);
		GrB_Matrix_wait(AT, GrB_MATERIALIZE);
		uint64_t i0 = read_instructions(); double t0 = now_sec();
		Tensor_ClearElements(T, A, AT);
		double t1 = now_sec(); uint64_t i1 = read_instructions();
		r.instr_per_op[rep] = (double)(i1 - i0) / n;
		r.sec_per_op[rep]   = (t1 - t0) / n;
		GrB_Matrix_free(&A); GrB_Matrix_free(&AT);
		Tensor_free(&T);
	}
	report(&r);
}

// `Tensor_SetEdges` takes an array of Edge pointers rather than parallel
// coordinate arrays, so it is the path a query's CREATE takes rather than the
// bulk loader's.
static void bench_set_edges(uint64_t n) {
	Result r = { .name = "set_edges (per edge)", .ops = n };
	Edge **ptrs = malloc(n * sizeof(Edge *));
	Edge  *es   = calloc(n, sizeof(Edge));
	for(uint64_t i = 0; i < n; i++) {
		es[i].src_id = i; es[i].dest_id = i; es[i].id = i;
		ptrs[i] = &es[i];
	}
	for(int rep = 0; rep < 3; rep++) {
		Tensor T = Tensor_new(n + 1, n + 1);
		uint64_t i0 = read_instructions(); double t0 = now_sec();
		Tensor_SetEdges(T, (const Edge **)ptrs, n);
		double t1 = now_sec(); uint64_t i1 = read_instructions();
		r.instr_per_op[rep] = (double)(i1 - i0) / n;
		r.sec_per_op[rep]   = (t1 - t0) / n;
		Tensor_free(&T);
	}
	free(ptrs); free(es);
	report(&r);
}

// ---- working-set sweep (residency) ----------------------------------------

// Every other read here is warm: N pairs sits inside cache, so what they price
// is the code path, not the memory system.  This sweeps the working set from
// far inside the cache to far outside it, probing in a scrambled order so that
// above the cache the reads genuinely miss.
//
// The instruction column is the control and is expected to be flat: a cache
// miss retires no extra instruction.  The nanoseconds are the measurement.
static void bench_sweep(int edges_per_pair) {
	static const uint64_t sizes[] = {
		10000, 100000, 500000, 2000000, 4000000, 8000000
	};
	const uint64_t probes = 200000;
	printf("\n=== working-set sweep, %d edge(s) per pair, scrambled probe order ===\n",
	       edges_per_pair);
	printf("%12s  %12s  %10s\n", "pairs", "instr/op", "ns/op");
	for(size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
		uint64_t n = sizes[si];
		Tensor T = build(n, edges_per_pair);
		double best_ns = 1e300, best_instr = 0;
		for(int rep = 0; rep < 3; rep++) {
			uint64_t i0 = read_instructions();
			double t0 = now_sec();
			uint64_t acc = 0, x = 1;
			for(uint64_t j = 0; j < probes; j++) {
				// odd multiplier: walks the rows with no useful locality and
				// without a stored permutation, which would itself sit in cache
				x = x * 0x9E3779B97F4A7C15ull + 1;
				GrB_Index p = (x >> 32) % n;
				TensorIterator it;
				TensorIterator_ScanEntry(&it, T, p, p);
				uint64_t v;
				while(TensorIterator_next(&it, NULL, NULL, &v, NULL)) acc += v;
			}
			double t1 = now_sec();
			uint64_t i1 = read_instructions();
			g_sink += acc;
			double ns = 1e9 * (t1 - t0) / probes;
			if(ns < best_ns) { best_ns = ns; best_instr = (double)(i1 - i0) / probes; }
		}
		printf("%12llu  %12.1f  %10.1f\n",
		       (unsigned long long)n, best_instr, best_ns);
		fflush(stdout);
		Delta_Matrix_free(&T);
	}
}

int main(int argc, char **argv) {
	if(argc > 1) N = strtoull(argv[1], NULL, 10);
	uint64_t reps = 1000000;
	if(argc > 2) reps = strtoull(argv[2], NULL, 10);

	// same init sequence as tests/unit/test_delta_matrix.c setup()
	Alloc_Reset();  // route rm_malloc & friends to libc malloc

	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if(info != GrB_SUCCESS) { fprintf(stderr, "GrB_init failed %d\n", info); return 1; }
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
	Config_Option_set(Config_DELTA_MAX_PENDING_CHANGES, "10000", NULL);

	printf("GraphBLAS %d.%d.%d, N=%llu pairs, reps=%llu\n",
	       GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR,
	       GxB_IMPLEMENTATION_SUB, (unsigned long long)N,
	       (unsigned long long)reps);

	report_container_size();

	Tensor T1 = build(N, 1);
	Tensor T2 = build(N, 2);

	size_t sz1 = 0, sz2 = 0;
	Tensor_memoryUsage(&sz1, T1);
	Tensor_memoryUsage(&sz2, T2);
	printf("tensor memoryUsage: %llu single-edge pairs = %zu bytes (%.1f B/edge)\n",
	       (unsigned long long)N, sz1, (double)sz1 / N);
	printf("tensor memoryUsage: %llu 2-edge pairs      = %zu bytes (%.1f B/edge)\n",
	       (unsigned long long)N, sz2, (double)sz2 / (2 * N));
	verify_cells(T1, "T1 (1 edge/pair)");
	verify_cells(T2, "T2 (2 edges/pair)");
	printf("\n");

	bench_raw_read(T1, "raw GrB_Matrix read of M, single-edge", reps);
	bench_point_read(T1, "point read cell, single-edge (inline)", reps);
	bench_point_read(T2, "point read cell, 2-edge (tagged ptr)", reps);
	bench_point_read_edges(T1, "point read + edge ids, single-edge", reps);
	bench_point_read_edges(T2, "point read + edge ids, 2-edge", reps);

	uint64_t passes = (reps / N) ? (reps / N) : 1;
	bench_iterate(T1, "full iteration, all single-edge  (per edge)", N, passes);
	bench_iterate(T2, "full iteration, all 2-edge       (per edge)", 2 * N, passes);

	bench_iterate_t(T1, "transposed iteration, single-edge (per edge)", N, passes);
	bench_iterate_t(T2, "transposed iteration, 2-edge     (per edge)", 2 * N, passes);

	Tensor_free(&T1);
	Tensor_free(&T2);

	//--------------------------------------------------------------------------
	// fan-out: how reads and iteration scale past k = 2, which the paper listed
	// as unmeasured on this side.
	//--------------------------------------------------------------------------
	printf("\n-- fan-out: k edge ids per pair --\n");
	for(uint64_t k = 1; k <= 16; k *= 2) {
		Tensor Tk = build(N, k);
		char nm1[128], nm2[128], nm3[128];
		snprintf(nm1, sizeof nm1, "point read + all ids, k=%-2llu", (unsigned long long)k);
		snprintf(nm2, sizeof nm2, "full iteration (per edge), k=%-2llu", (unsigned long long)k);
		snprintf(nm3, sizeof nm3, "transposed iteration (per edge), k=%-2llu", (unsigned long long)k);
		bench_point_read_edges(Tk, nm1, reps);
		uint64_t pk = (reps / (N * k)) ? (reps / (N * k)) : 1;
		bench_iterate(Tk, nm2, N * k, pk);
		bench_iterate_t(Tk, nm3, N * k, pk);
		Tensor_free(&Tk);
	}

	//--------------------------------------------------------------------------
	// the entry points the paper listed as unmeasured on this side
	//--------------------------------------------------------------------------
	printf("\n-- remaining entry points --\n");
	{
		Tensor D1 = build(N, 1);
		Tensor D2 = build(N, 2);
		bench_degrees(D1, "single-edge", reps);
		bench_degrees(D2, "2-edge", reps);
		Tensor_free(&D1);
		Tensor_free(&D2);
	}
	bench_remove_flat(N);
	bench_clear_elements(N, 1);
	bench_clear_elements(N, 2);
	bench_set_edges(N);

	bench_sweep(1);
	bench_sweep(2);

	bench_promote_demote(N);

	//--------------------------------------------------------------------------
	// transition vs non-transitioning control on the same code path
	//--------------------------------------------------------------------------
	printf("\n-- transition vs non-transitioning control (same entry point) --\n");
	double promote = bench_add_one_edge(N, 1,
			"promote   +1 edge on 1-edge pairs (transitions)");
	double add3    = bench_add_one_edge(N, 2,
			"control   +1 edge on 2-edge pairs (no transition)");
	printf("  => promotion cost = %.1f - %.1f = %+.1f instr/pair (LOWER BOUND:\n"
	       "     the control's containers already hold 2N ids when measured,\n"
	       "     while the promote pass starts with none)\n",
	       promote, add3, promote - add3);

	double demote = bench_remove_one_edge(N, 2,
			"demote    -1 edge on 2-edge pairs (transitions)");
	double rem3   = bench_remove_one_edge(N, 3,
			"control   -1 edge on 3-edge pairs (no transition)");
	printf("  => demotion cost = %.1f - %.1f = %+.1f instr/pair (same lower-bound\n"
	       "     caveat: the control's containers are larger when measured)\n",
	       demote, rem3, demote - rem3);

	//--------------------------------------------------------------------------
	// batch vs incremental construction of the same final tensor
	//--------------------------------------------------------------------------
	printf("\n-- building the same N x 2-edge tensor two ways --\n");
	bench_build_paths(N);

	//--------------------------------------------------------------------------
	// space as a function of ids per pair
	//--------------------------------------------------------------------------
	report_space_vs_k(N);

	printf("\nsink=%llu\n", (unsigned long long)g_sink);
	return 0;
}
