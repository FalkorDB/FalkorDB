#pragma once
#include <benchmark/benchmark.h>

// Define C++ things
#define restrict
extern "C" {
	#include "RG.h"
	#include "rax.h"
	#include "LAGraph.h"
	#include "LAGraphX.h"
	#include "src/globals.h"
	#include "util/simple_rand.h"
	#include "src/graph/graphcontext.h"
	#include "src/graph/tensor/tensor.h"
	#include "src/configuration/config.h"
	#include "src/graph/delta_matrix/delta_utils.h"
	#include "src/arithmetic/algebraic_expression.h"

	bool QueryCtx_Init(void) ;
	void QueryCtx_Free(void) ;
	void QueryCtx_SetGraphCtx (GraphContext *gc) ;
}
#undef restrict

void Delta_Random_Matrix
(
	Delta_Matrix *A,
	GrB_Type type,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) ;

// Make a random tensor
void Random_Tensor
(
	Tensor *A,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) ;
