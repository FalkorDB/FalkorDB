//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_prejit.c: return list of PreJIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake from Config/GB_prejit.c.in, which has
// indexed the following 80 kernels in GraphBLAS/PreJIT:

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"
#include "jit_kernels/include/GB_jit_kernel_proto.h"
#include "include/GB_pedantic_disable.h"

//------------------------------------------------------------------------------
// prototypes for all PreJIT kernels
//------------------------------------------------------------------------------

JIT_DOT2 (GB_jit__AxB_dot2__0000000e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT2 (GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__0000400e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT2 (GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__0380400e894e89b7__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT2 (GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0000800e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT4 (GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0000c00e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT4 (GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0380c00e890e89c7__LG_MSF_tupleMin_int_LG_MSF_combine_int)
JIT_DOT4 (GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_SAX4 (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp)
JIT_SAX4 (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int)
JIT_ADD  (GB_jit__add__00000288808288a)
JIT_ADD  (GB_jit__add__00000288808388a)
JIT_ADD  (GB_jit__add__e3f002bbb0bbb45)
JIT_AP1  (GB_jit__apply_bind1st__000000e890ef9c3__LG_MSF_combine_int)
JIT_AP1  (GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp)
JIT_AP0  (GB_jit__apply_unop__0040008e08ee__LG_MSF_get_first_int)
JIT_AP0  (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_int)
JIT_AP0  (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_int)
JIT_AP0  (GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp)
JIT_AP0  (GB_jit__apply_unop__00400277076f)
JIT_AP0  (GB_jit__apply_unop__004002bb0b2e)
JIT_AP0  (GB_jit__apply_unop__004002bb0b3e)
JIT_AP0  (GB_jit__apply_unop__03c0008e08ed__LG_MSF_get_first_int)
JIT_AP0  (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_int)
JIT_AP0  (GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000003f000499b3)
JIT_SUB  (GB_jit__bitmap_assign_5_whole__0000100611101182)
JIT_BLD  (GB_jit__build__e9022222)
JIT_BLD  (GB_jit__build__e9033333)
JIT_BLD  (GB_jit__build__e90bbbbb)
JIT_EWFN (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp)
JIT_EWFN (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int)
JIT_MAS2 (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp)
JIT_MAS2 (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_int)
JIT_RED  (GB_jit__reduce__14b82)
JIT_RED  (GB_jit__reduce__14bb9)
JIT_ROWS (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp)
JIT_ROWS (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_int)
JIT_SELB (GB_jit__select_bitmap__003318e8a__LG_MSF_removeEdge_int)
JIT_SELB (GB_jit__select_bitmap__003318e8a__LG_MSF_selectEdge_int)
JIT_SELB (GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp)
JIT_SELB (GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp)
JIT_SELB (GB_jit__select_bitmap__00f318e8a__LG_MSF_removeEdge_int)
JIT_SELB (GB_jit__select_bitmap__00f318e8a__LG_MSF_selectEdge_int)
JIT_SEL1 (GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3ff318e85__LG_MSF_removeEdge_int)
JIT_SEL1 (GB_jit__select_phase1__3ff318e85__LG_MSF_selectEdge_int)
JIT_SEL1 (GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int)
JIT_SEL2 (GB_jit__select_phase2__3ff318e85__LG_MSF_selectEdge_int)
JIT_SEL2 (GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp)
JIT_SUB  (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp)
JIT_SUB  (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_int)
JIT_SUB  (GB_jit__subassign_08n__00000042999499f3)
JIT_SUB  (GB_jit__subassign_08n__00000050999499f3)
JIT_SUB  (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp)
JIT_SUB  (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_int)
JIT_BREF (GB_jit__subref_bitmap__00009f)
JIT_BREF (GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_fp)
JIT_BREF (GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_int)
JIT_BREF (GB_jit__subref_bitmap__000c9f)
JIT_BREF (GB_jit__subref_bitmap__000cef__LG_MSF_tuple_fp)
JIT_BREF (GB_jit__subref_bitmap__000cef__LG_MSF_tuple_int)
JIT_BREF (GB_jit__subref_bitmap__008c6f)
JIT_BREF (GB_jit__subref_bitmap__008c9f)
JIT_TR0  (GB_jit__trans_unop__00480288082a)
JIT_TR0  (GB_jit__trans_unop__00480288083a)
JIT_TR0  (GB_jit__trans_unop__1bc802bb0bb5)


//------------------------------------------------------------------------------
// prototypes for all PreJIT query kernels
//------------------------------------------------------------------------------

JIT_Q (GB_jit__AxB_dot2__0000000e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__0000400e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__0380400e894e89b7__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0000800e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0000c00e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0380c00e890e89c7__LG_MSF_tupleMin_int_LG_MSF_combine_int_query)
JIT_Q (GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp_query)
JIT_Q (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int_query)
JIT_Q (GB_jit__add__00000288808288a_query)
JIT_Q (GB_jit__add__00000288808388a_query)
JIT_Q (GB_jit__add__e3f002bbb0bbb45_query)
JIT_Q (GB_jit__apply_bind1st__000000e890ef9c3__LG_MSF_combine_int_query)
JIT_Q (GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp_query)
JIT_Q (GB_jit__apply_unop__0040008e08ee__LG_MSF_get_first_int_query)
JIT_Q (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_int_query)
JIT_Q (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_int_query)
JIT_Q (GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp_query)
JIT_Q (GB_jit__apply_unop__00400277076f_query)
JIT_Q (GB_jit__apply_unop__004002bb0b2e_query)
JIT_Q (GB_jit__apply_unop__004002bb0b3e_query)
JIT_Q (GB_jit__apply_unop__03c0008e08ed__LG_MSF_get_first_int_query)
JIT_Q (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_int_query)
JIT_Q (GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000003f000499b3_query)
JIT_Q (GB_jit__bitmap_assign_5_whole__0000100611101182_query)
JIT_Q (GB_jit__build__e9022222_query)
JIT_Q (GB_jit__build__e9033333_query)
JIT_Q (GB_jit__build__e90bbbbb_query)
JIT_Q (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp_query)
JIT_Q (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int_query)
JIT_Q (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_int_query)
JIT_Q (GB_jit__reduce__14b82_query)
JIT_Q (GB_jit__reduce__14bb9_query)
JIT_Q (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp_query)
JIT_Q (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_int_query)
JIT_Q (GB_jit__select_bitmap__003318e8a__LG_MSF_removeEdge_int_query)
JIT_Q (GB_jit__select_bitmap__003318e8a__LG_MSF_selectEdge_int_query)
JIT_Q (GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_bitmap__00f318e8a__LG_MSF_removeEdge_int_query)
JIT_Q (GB_jit__select_bitmap__00f318e8a__LG_MSF_selectEdge_int_query)
JIT_Q (GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3ff318e85__LG_MSF_removeEdge_int_query)
JIT_Q (GB_jit__select_phase1__3ff318e85__LG_MSF_selectEdge_int_query)
JIT_Q (GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int_query)
JIT_Q (GB_jit__select_phase2__3ff318e85__LG_MSF_selectEdge_int_query)
JIT_Q (GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_int_query)
JIT_Q (GB_jit__subassign_08n__00000042999499f3_query)
JIT_Q (GB_jit__subassign_08n__00000050999499f3_query)
JIT_Q (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp_query)
JIT_Q (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_int_query)
JIT_Q (GB_jit__subref_bitmap__00009f_query)
JIT_Q (GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_int_query)
JIT_Q (GB_jit__subref_bitmap__000c9f_query)
JIT_Q (GB_jit__subref_bitmap__000cef__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subref_bitmap__000cef__LG_MSF_tuple_int_query)
JIT_Q (GB_jit__subref_bitmap__008c6f_query)
JIT_Q (GB_jit__subref_bitmap__008c9f_query)
JIT_Q (GB_jit__trans_unop__00480288082a_query)
JIT_Q (GB_jit__trans_unop__00480288083a_query)
JIT_Q (GB_jit__trans_unop__1bc802bb0bb5_query)


//------------------------------------------------------------------------------
// GB_prejit_kernels: a list of function pointers to PreJIT kernels
//------------------------------------------------------------------------------

#if ( 80 > 0 )
static void *GB_prejit_kernels [80] =
{
GB_jit__AxB_dot2__0000000e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__0000400e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__0380400e894e89b7__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0000800e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0000c00e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0380c00e890e89c7__LG_MSF_tupleMin_int_LG_MSF_combine_int,
GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int,
GB_jit__add__00000288808288a,
GB_jit__add__00000288808388a,
GB_jit__add__e3f002bbb0bbb45,
GB_jit__apply_bind1st__000000e890ef9c3__LG_MSF_combine_int,
GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp,
GB_jit__apply_unop__0040008e08ee__LG_MSF_get_first_int,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_int,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_int,
GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp,
GB_jit__apply_unop__00400277076f,
GB_jit__apply_unop__004002bb0b2e,
GB_jit__apply_unop__004002bb0b3e,
GB_jit__apply_unop__03c0008e08ed__LG_MSF_get_first_int,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_int,
GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp,
GB_jit__bitmap_assign_2_whole__0000003f000499b3,
GB_jit__bitmap_assign_5_whole__0000100611101182,
GB_jit__build__e9022222,
GB_jit__build__e9033333,
GB_jit__build__e90bbbbb,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_int,
GB_jit__reduce__14b82,
GB_jit__reduce__14bb9,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_int,
GB_jit__select_bitmap__003318e8a__LG_MSF_removeEdge_int,
GB_jit__select_bitmap__003318e8a__LG_MSF_selectEdge_int,
GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp,
GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp,
GB_jit__select_bitmap__00f318e8a__LG_MSF_removeEdge_int,
GB_jit__select_bitmap__00f318e8a__LG_MSF_selectEdge_int,
GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase1__3ff318e85__LG_MSF_removeEdge_int,
GB_jit__select_phase1__3ff318e85__LG_MSF_selectEdge_int,
GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int,
GB_jit__select_phase2__3ff318e85__LG_MSF_selectEdge_int,
GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_int,
GB_jit__subassign_08n__00000042999499f3,
GB_jit__subassign_08n__00000050999499f3,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_int,
GB_jit__subref_bitmap__00009f,
GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_fp,
GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_int,
GB_jit__subref_bitmap__000c9f,
GB_jit__subref_bitmap__000cef__LG_MSF_tuple_fp,
GB_jit__subref_bitmap__000cef__LG_MSF_tuple_int,
GB_jit__subref_bitmap__008c6f,
GB_jit__subref_bitmap__008c9f,
GB_jit__trans_unop__00480288082a,
GB_jit__trans_unop__00480288083a,
GB_jit__trans_unop__1bc802bb0bb5
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_queries: a list of function pointers to PreJIT query kernels
//------------------------------------------------------------------------------

#if ( 80 > 0 )
static void *GB_prejit_queries [80] =
{
GB_jit__AxB_dot2__0000000e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__0000400e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__0380400e894e89b7__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0000800e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0000c00e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0380c00e890e89c7__LG_MSF_tupleMin_int_LG_MSF_combine_int_query,
GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp_query,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int_query,
GB_jit__add__00000288808288a_query,
GB_jit__add__00000288808388a_query,
GB_jit__add__e3f002bbb0bbb45_query,
GB_jit__apply_bind1st__000000e890ef9c3__LG_MSF_combine_int_query,
GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp_query,
GB_jit__apply_unop__0040008e08ee__LG_MSF_get_first_int_query,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_int_query,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_int_query,
GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp_query,
GB_jit__apply_unop__00400277076f_query,
GB_jit__apply_unop__004002bb0b2e_query,
GB_jit__apply_unop__004002bb0b3e_query,
GB_jit__apply_unop__03c0008e08ed__LG_MSF_get_first_int_query,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_int_query,
GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp_query,
GB_jit__bitmap_assign_2_whole__0000003f000499b3_query,
GB_jit__bitmap_assign_5_whole__0000100611101182_query,
GB_jit__build__e9022222_query,
GB_jit__build__e9033333_query,
GB_jit__build__e90bbbbb_query,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp_query,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int_query,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp_query,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_int_query,
GB_jit__reduce__14b82_query,
GB_jit__reduce__14bb9_query,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp_query,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_int_query,
GB_jit__select_bitmap__003318e8a__LG_MSF_removeEdge_int_query,
GB_jit__select_bitmap__003318e8a__LG_MSF_selectEdge_int_query,
GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp_query,
GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp_query,
GB_jit__select_bitmap__00f318e8a__LG_MSF_removeEdge_int_query,
GB_jit__select_bitmap__00f318e8a__LG_MSF_selectEdge_int_query,
GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase1__3ff318e85__LG_MSF_removeEdge_int_query,
GB_jit__select_phase1__3ff318e85__LG_MSF_selectEdge_int_query,
GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int_query,
GB_jit__select_phase2__3ff318e85__LG_MSF_selectEdge_int_query,
GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp_query,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp_query,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_int_query,
GB_jit__subassign_08n__00000042999499f3_query,
GB_jit__subassign_08n__00000050999499f3_query,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp_query,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_int_query,
GB_jit__subref_bitmap__00009f_query,
GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_fp_query,
GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_int_query,
GB_jit__subref_bitmap__000c9f_query,
GB_jit__subref_bitmap__000cef__LG_MSF_tuple_fp_query,
GB_jit__subref_bitmap__000cef__LG_MSF_tuple_int_query,
GB_jit__subref_bitmap__008c6f_query,
GB_jit__subref_bitmap__008c9f_query,
GB_jit__trans_unop__00480288082a_query,
GB_jit__trans_unop__00480288083a_query,
GB_jit__trans_unop__1bc802bb0bb5_query
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_names: a list of names of PreJIT kernels
//------------------------------------------------------------------------------

#if ( 80 > 0 )
static char *GB_prejit_names [80] =
{
"GB_jit__AxB_dot2__0000000e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__0000400e894e89bb__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__0380400e894e89b7__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0000800e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0000c00e890e89cb__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0380c00e890e89c7__LG_MSF_tupleMin_int_LG_MSF_combine_int",
"GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp",
"GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int",
"GB_jit__add__00000288808288a",
"GB_jit__add__00000288808388a",
"GB_jit__add__e3f002bbb0bbb45",
"GB_jit__apply_bind1st__000000e890ef9c3__LG_MSF_combine_int",
"GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp",
"GB_jit__apply_unop__0040008e08ee__LG_MSF_get_first_int",
"GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp",
"GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_int",
"GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp",
"GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_int",
"GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp",
"GB_jit__apply_unop__00400277076f",
"GB_jit__apply_unop__004002bb0b2e",
"GB_jit__apply_unop__004002bb0b3e",
"GB_jit__apply_unop__03c0008e08ed__LG_MSF_get_first_int",
"GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp",
"GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_int",
"GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp",
"GB_jit__bitmap_assign_2_whole__0000003f000499b3",
"GB_jit__bitmap_assign_5_whole__0000100611101182",
"GB_jit__build__e9022222",
"GB_jit__build__e9033333",
"GB_jit__build__e90bbbbb",
"GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp",
"GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int",
"GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp",
"GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_int",
"GB_jit__reduce__14b82",
"GB_jit__reduce__14bb9",
"GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp",
"GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_int",
"GB_jit__select_bitmap__003318e8a__LG_MSF_removeEdge_int",
"GB_jit__select_bitmap__003318e8a__LG_MSF_selectEdge_int",
"GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp",
"GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp",
"GB_jit__select_bitmap__00f318e8a__LG_MSF_removeEdge_int",
"GB_jit__select_bitmap__00f318e8a__LG_MSF_selectEdge_int",
"GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase1__3ff318e85__LG_MSF_removeEdge_int",
"GB_jit__select_phase1__3ff318e85__LG_MSF_selectEdge_int",
"GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int",
"GB_jit__select_phase2__3ff318e85__LG_MSF_selectEdge_int",
"GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp",
"GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp",
"GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_int",
"GB_jit__subassign_08n__00000042999499f3",
"GB_jit__subassign_08n__00000050999499f3",
"GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp",
"GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_int",
"GB_jit__subref_bitmap__00009f",
"GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_fp",
"GB_jit__subref_bitmap__0000ef__LG_MSF_tuple_int",
"GB_jit__subref_bitmap__000c9f",
"GB_jit__subref_bitmap__000cef__LG_MSF_tuple_fp",
"GB_jit__subref_bitmap__000cef__LG_MSF_tuple_int",
"GB_jit__subref_bitmap__008c6f",
"GB_jit__subref_bitmap__008c9f",
"GB_jit__trans_unop__00480288082a",
"GB_jit__trans_unop__00480288083a",
"GB_jit__trans_unop__1bc802bb0bb5"
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit: return list of PreJIT function pointers and function names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void ***Kernel_handle,  // return list of function pointers to kernels
    void ***Query_handle,   // return list of function pointers to queries
    char ***Name_handle     // return list of kernel names
)
{
    (*nkernels) = 80 ;
    #if ( 80 == 0 )
    (*Kernel_handle) = NULL ;
    (*Query_handle) = NULL ;
    (*Name_handle) = NULL ;
    #else
    (*Kernel_handle) = GB_prejit_kernels ;
    (*Query_handle) = GB_prejit_queries ;
    (*Name_handle) = GB_prejit_names ;
    #endif
}

