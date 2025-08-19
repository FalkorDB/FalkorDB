#pragma once
#include "RG.h"
#include <LAGraphX.h>
#include "GraphBLAS.h"
#include "util/simple_rand.h"
#include <benchmark/benchmark.h>
extern "C" {
#include "globals.h"
#include "src/graph/tensor/tensor.h"
#include "src/configuration/config.h"
#include "src/graph/delta_matrix/delta_utils.h"
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
