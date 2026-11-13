// Reads n, nnz, xadj[n+1], adjncy[nnz] (int64 little-endian), runs METIS_NodeND with
// default options exactly as cch.c's CCH_EliminationOrder does, writes perm[n] (int64).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>
#include "metis.h"

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: nd in.bin perm.bin\n"); return 1; }
	FILE *f = fopen(argv[1], "rb");
	int64_t n, nnz;
	fread(&n, 8, 1, f); fread(&nnz, 8, 1, f);
	idx_t *xadj = malloc(sizeof(idx_t) * (n + 1));
	idx_t *adjncy = malloc(sizeof(idx_t) * nnz);
	fread(xadj, 8, n + 1, f); fread(adjncy, 8, nnz, f);
	fclose(f);
	struct rusage before; getrusage(RUSAGE_SELF, &before);
	idx_t *perm = malloc(sizeof(idx_t) * n), *iperm = malloc(sizeof(idx_t) * n);
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	idx_t nvtxs = n;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	int status = METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	struct rusage after; getrusage(RUSAGE_SELF, &after);
	double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	// macOS ru_maxrss is bytes
	printf("METIS_NodeND status %d, %.3fs, maxrss before %.1f MB after %.1f MB\n", status, secs,
		before.ru_maxrss / 1e6, after.ru_maxrss / 1e6);
	f = fopen(argv[2], "wb");
	fwrite(perm, 8, n, f);
	fclose(f);
	return 0;
}
