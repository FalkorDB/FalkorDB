//----------------------------------------------------------------------------
// graph_zachary_karate.h - index arrays for Zachary Karate Club graph
// ----------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

// Contributed by Scott McMillan, SEI Carnegie Mellon University

//-----------------------------------------------------------------------------

#ifndef GRAPH_ZACHARY_KARATE_H
#define GRAPH_ZACHARY_KARATE_H

// The graph "karate" contains the network of friendships between the 34
// members of a karate club at a US university, as described by Wayne Zachary
// in 1977: W. W. Zachary, An information flow model for conflict and fission
// in small groups, Journal of Anthropological Research 33, 452-473 (1977).
// This data is also contained in the LAGraph/data/karate.mtx file, in
// MatrixMarket format.

//****************************************************************************
GrB_Index const ZACHARY_NUM_NODES = 34;
GrB_Index const ZACHARY_NUM_EDGES = 156;

GrB_Index const ZACHARY_I[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,
    4,4,4,
    5,5,5,5,
    6,6,6,6,
    7,7,7,7,
    8,8,8,8,8,
    9,9,
    10,10,10,
    11,
    12,12,
    13,13,13,13,13,
    14,14,
    15,15,
    16,16,
    17,17,
    18,18,
    19,19,19,
    20,20,
    21,21,
    22,22,
    23,23,23,23,23,
    24,24,24,
    25,25,25,
    26,26,
    27,27,27,27,
    28,28,28,
    29,29,29,29,
    30,30,30,30,
    31,31,31,31,31,31,
    32,32,32,32,32,32,32,32,32,32,32,32,
    33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

GrB_Index const ZACHARY_J[] = {
    1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,
    0,2,3,7,13,17,19,21,30,
    0,1,3,7,8,9,13,27,28,32,
    0,1,2,7,12,13,
    0,6,10,
    0,6,10,16,
    0,4,5,16,
    0,1,2,3,
    0,2,30,32,33,
    2,33,
    0,4,5,
    0,
    0,3,
    0,1,2,3,33,
    32,33,
    32,33,
    5,6,
    0,1,
    32,33,
    0,1,33,
    32,33,
    0,1,
    32,33,
    25,27,29,32,33,
    25,27,31,
    23,24,31,
    29,33,
    2,23,24,33,
    2,31,33,
    23,26,32,33,
    1,8,32,33,
    0,24,25,28,32,33,
    2,8,14,15,18,20,22,23,29,30,31,33,
    8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};

uint32_t const ZACHARY_V[] = {
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U};

#endif
