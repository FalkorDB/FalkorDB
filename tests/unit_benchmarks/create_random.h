#pragma once
#include "RG.h"
#include <LAGraphX.h>
#include "GraphBLAS.h"
#include "util/simple_rand.h"
#include <benchmark/benchmark.h>

// Define C++ things
#define restrict 
extern "C" {
	#include "rax.h"
	#include "src/graph/tensor/tensor.h"
	#include "src/configuration/config.h"
	#include "src/graph/delta_matrix/delta_utils.h"
	#include "src/arithmetic/algebraic_expression.h"

	struct Global_ops {
		GrB_Scalar   empty;          // empty scalar
		GrB_BinaryOp not_zombie;     // binary operator to check if a value is not a zombie
		GrB_Semiring any_alive;      // semiring to check if any entry is alive
		GrB_UnaryOp  free_tensors;   // unary operator to free tensor entries
	};

	void Global_GrB_Ops_Init(void) ;
	void Global_GrB_Ops_Free(void) ;
	const struct Global_ops *Global_GrB_Ops_Get(void) ;
}

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
