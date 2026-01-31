//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_prejit.c: return list of PreJIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake from Config/GB_prejit.c.in, which has
// indexed the following 129 kernels in GraphBLAS/PreJIT:

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"
#include "jit_kernels/include/GB_jit_kernel_proto.h"
#include "include/GB_pedantic_disable.h"

//------------------------------------------------------------------------------
// prototypes for all PreJIT kernels
//------------------------------------------------------------------------------

JIT_DOT2 (GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__0000400eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT2 (GB_jit__AxB_dot2__038120f1100110c7)
JIT_DOT2 (GB_jit__AxB_dot2__0384410b0b3b0ba6)
JIT_DOT4 (GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0000c00eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0004c10a0a0a0acb)
JIT_DOT4 (GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp)
JIT_DOT4 (GB_jit__AxB_dot4__0384c10a0a0a0ac7)
JIT_SAXB (GB_jit__AxB_saxbit__000420fbb00bb08f)
JIT_SAXB (GB_jit__AxB_saxbit__0384610b0b3b0ba6)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f110011000)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f110011041)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f990099000)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f990099004)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f990099041)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f120f990099045)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f4410b0b2b0b65)
JIT_SAX3 (GB_jit__AxB_saxpy3__e3f4410b0b3b0b65)
JIT_SAX3 (GB_jit__AxB_saxpy3__fff120f990399000)
JIT_SAX4 (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp)
JIT_ADD  (GB_jit__add__000203aaa0aaace)
JIT_ADD  (GB_jit__add__000203aaa0aaacf)
JIT_ADD  (GB_jit__add__000214aaa0aaacf)
JIT_ADD  (GB_jit__add__e3f001111019100)
JIT_ADD  (GB_jit__add__e3f101111019100)
JIT_AP1  (GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp)
JIT_AP2  (GB_jit__apply_bind2nd__000014aaa0a8fc8)
JIT_AP2  (GB_jit__apply_bind2nd__000014aaa0a8fcc)
JIT_AP0  (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp)
JIT_AP0  (GB_jit__apply_unop__004005bb0bbe)
JIT_AP0  (GB_jit__apply_unop__004007aa0aaf)
JIT_AP0  (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp)
JIT_AP0  (GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp)
JIT_AP0  (GB_jit__apply_unop__03c005bb0bbd)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000003f0002aaa3)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000003f000499b3)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000103f00020aa3)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000803f000299a2)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__0000a03f000301a0)
JIT_SUB  (GB_jit__bitmap_assign_2_whole__001c903f000301a0)
JIT_SUB  (GB_jit__bitmap_assign_4_whole__00e0203f00026890)
JIT_SUB  (GB_jit__bitmap_assign_4_whole__00fc103f00020180)
JIT_SUB  (GB_jit__bitmap_assign_5_whole__00000004bbb0bb82)
JIT_BLD  (GB_jit__build__e9011119)
JIT_BLD  (GB_jit__build__e9066668)
JIT_BLD  (GB_jit__build__e90bbbbb)
JIT_BLD  (GB_jit__build__f9066668)
JIT_EM2  (GB_jit__emult_02__e38005bbb0bbb46)
JIT_EMB  (GB_jit__emult_bitmap__000014bbb2bbbae)
JIT_EMB  (GB_jit__emult_bitmap__000214bbb2bbbae)
JIT_EWFN (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp)
JIT_EWFN (GB_jit__ewise_fulln__000014aaa0aaacf)
JIT_MAS1 (GB_jit__masker_phase1__dff32000)
JIT_MAS1 (GB_jit__masker_phase1__dff32045)
JIT_MAS1 (GB_jit__masker_phase1__dff33000)
JIT_MAS1 (GB_jit__masker_phase1__dff33041)
JIT_MAS1 (GB_jit__masker_phase1__dff33045)
JIT_MAS1 (GB_jit__masker_phase1__dff33054)
JIT_MAS2 (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp)
JIT_MAS2 (GB_jit__masker_phase2__fff02900)
JIT_MAS2 (GB_jit__masker_phase2__fff03900)
JIT_MAS2 (GB_jit__masker_phase2__fff03941)
JIT_RED  (GB_jit__reduce__14aa2)
JIT_RED  (GB_jit__reduce__14aa3)
JIT_ROWS (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp)
JIT_SELB (GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp)
JIT_SELB (GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp)
JIT_SELB (GB_jit__select_bitmap__00f31beba__LG_MSF_removeEdge_fp)
JIT_SELB (GB_jit__select_bitmap__00f31beba__LG_MSF_selectEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp)
JIT_SEL1 (GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp)
JIT_SEL2 (GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp)
JIT_SUB  (GB_jit__subassign_02__7f1c417f00001101)
JIT_SUB  (GB_jit__subassign_02__7f1c517f00000101)
JIT_SUB  (GB_jit__subassign_05__07e0207f00026850)
JIT_SUB  (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp)
JIT_SUB  (GB_jit__subassign_06n__071c007f00029971)
JIT_SUB  (GB_jit__subassign_06n__07e0007f00021100)
JIT_SUB  (GB_jit__subassign_06n__07e0007f00021110)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00020651)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00020900)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00021100)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00026651)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00029900)
JIT_SUB  (GB_jit__subassign_06n__07fc007f00029951)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00020100)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00020111)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00020140)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00020900)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00020940)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00029900)
JIT_SUB  (GB_jit__subassign_06n__07fc107f00029940)
JIT_SUB  (GB_jit__subassign_06s__7f1c407f00029965)
JIT_SUB  (GB_jit__subassign_06s__7fe0407f00021100)
JIT_SUB  (GB_jit__subassign_06s__7fe0407f00021110)
JIT_SUB  (GB_jit__subassign_06s__7fe0407f00029100)
JIT_SUB  (GB_jit__subassign_06s__7fe0407f00029110)
JIT_SUB  (GB_jit__subassign_06s__7ffc407f00036655)
JIT_SUB  (GB_jit__subassign_06s__7ffc407f00039900)
JIT_SUB  (GB_jit__subassign_06s__7ffc507f00020100)
JIT_SUB  (GB_jit__subassign_06s__7ffc507f00020110)
JIT_SUB  (GB_jit__subassign_08n__00000042999499f3)
JIT_SUB  (GB_jit__subassign_08n__00000050999499f3)
JIT_SUB  (GB_jit__subassign_13__7f00607f00030164)
JIT_SUB  (GB_jit__subassign_13__7fe0607f00030154)
JIT_SUB  (GB_jit__subassign_13__7fe0607f00050154)
JIT_SUB  (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp)
JIT_SUB  (GB_jit__subassign_23__00000044bbb0bbc2)
JIT_SUB  (GB_jit__subassign_23__00000052aaa0aac3)
JIT_SUB  (GB_jit__subassign_23__00001044bbb0bbc3)
JIT_SUB  (GB_jit__subassign_23__001c0044bbb0bbc1)
JIT_BREF (GB_jit__subref_bitmap__080009f)
JIT_BREF (GB_jit__subref_bitmap__08000ef__LG_MSF_tuple_fp)
JIT_BREF (GB_jit__subref_bitmap__0800c9f)
JIT_BREF (GB_jit__subref_bitmap__0800cef__LG_MSF_tuple_fp)
JIT_BREF (GB_jit__subref_bitmap__0808c6f)
JIT_SREF (GB_jit__subref_sparse__0bf8490)
JIT_SREF (GB_jit__subref_sparse__e3f8c90)


//------------------------------------------------------------------------------
// prototypes for all PreJIT query kernels
//------------------------------------------------------------------------------

JIT_Q (GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__0000400eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot2__038120f1100110c7_query)
JIT_Q (GB_jit__AxB_dot2__0384410b0b3b0ba6_query)
JIT_Q (GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0000c00eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0004c10a0a0a0acb_query)
JIT_Q (GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query)
JIT_Q (GB_jit__AxB_dot4__0384c10a0a0a0ac7_query)
JIT_Q (GB_jit__AxB_saxbit__000420fbb00bb08f_query)
JIT_Q (GB_jit__AxB_saxbit__0384610b0b3b0ba6_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f110011000_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f110011041_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f990099000_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f990099004_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f990099041_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f120f990099045_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f4410b0b2b0b65_query)
JIT_Q (GB_jit__AxB_saxpy3__e3f4410b0b3b0b65_query)
JIT_Q (GB_jit__AxB_saxpy3__fff120f990399000_query)
JIT_Q (GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp_query)
JIT_Q (GB_jit__add__000203aaa0aaace_query)
JIT_Q (GB_jit__add__000203aaa0aaacf_query)
JIT_Q (GB_jit__add__000214aaa0aaacf_query)
JIT_Q (GB_jit__add__e3f001111019100_query)
JIT_Q (GB_jit__add__e3f101111019100_query)
JIT_Q (GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp_query)
JIT_Q (GB_jit__apply_bind2nd__000014aaa0a8fc8_query)
JIT_Q (GB_jit__apply_bind2nd__000014aaa0a8fcc_query)
JIT_Q (GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp_query)
JIT_Q (GB_jit__apply_unop__004005bb0bbe_query)
JIT_Q (GB_jit__apply_unop__004007aa0aaf_query)
JIT_Q (GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp_query)
JIT_Q (GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp_query)
JIT_Q (GB_jit__apply_unop__03c005bb0bbd_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000003f0002aaa3_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000003f000499b3_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000103f00020aa3_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000803f000299a2_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__0000a03f000301a0_query)
JIT_Q (GB_jit__bitmap_assign_2_whole__001c903f000301a0_query)
JIT_Q (GB_jit__bitmap_assign_4_whole__00e0203f00026890_query)
JIT_Q (GB_jit__bitmap_assign_4_whole__00fc103f00020180_query)
JIT_Q (GB_jit__bitmap_assign_5_whole__00000004bbb0bb82_query)
JIT_Q (GB_jit__build__e9011119_query)
JIT_Q (GB_jit__build__e9066668_query)
JIT_Q (GB_jit__build__e90bbbbb_query)
JIT_Q (GB_jit__build__f9066668_query)
JIT_Q (GB_jit__emult_02__e38005bbb0bbb46_query)
JIT_Q (GB_jit__emult_bitmap__000014bbb2bbbae_query)
JIT_Q (GB_jit__emult_bitmap__000214bbb2bbbae_query)
JIT_Q (GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp_query)
JIT_Q (GB_jit__ewise_fulln__000014aaa0aaacf_query)
JIT_Q (GB_jit__masker_phase1__dff32000_query)
JIT_Q (GB_jit__masker_phase1__dff32045_query)
JIT_Q (GB_jit__masker_phase1__dff33000_query)
JIT_Q (GB_jit__masker_phase1__dff33041_query)
JIT_Q (GB_jit__masker_phase1__dff33045_query)
JIT_Q (GB_jit__masker_phase1__dff33054_query)
JIT_Q (GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__masker_phase2__fff02900_query)
JIT_Q (GB_jit__masker_phase2__fff03900_query)
JIT_Q (GB_jit__masker_phase2__fff03941_query)
JIT_Q (GB_jit__reduce__14aa2_query)
JIT_Q (GB_jit__reduce__14aa3_query)
JIT_Q (GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp_query)
JIT_Q (GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_bitmap__00f31beba__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_bitmap__00f31beba__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp_query)
JIT_Q (GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp_query)
JIT_Q (GB_jit__subassign_02__7f1c417f00001101_query)
JIT_Q (GB_jit__subassign_02__7f1c517f00000101_query)
JIT_Q (GB_jit__subassign_05__07e0207f00026850_query)
JIT_Q (GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subassign_06n__071c007f00029971_query)
JIT_Q (GB_jit__subassign_06n__07e0007f00021100_query)
JIT_Q (GB_jit__subassign_06n__07e0007f00021110_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00020651_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00020900_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00021100_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00026651_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00029900_query)
JIT_Q (GB_jit__subassign_06n__07fc007f00029951_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00020100_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00020111_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00020140_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00020900_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00020940_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00029900_query)
JIT_Q (GB_jit__subassign_06n__07fc107f00029940_query)
JIT_Q (GB_jit__subassign_06s__7f1c407f00029965_query)
JIT_Q (GB_jit__subassign_06s__7fe0407f00021100_query)
JIT_Q (GB_jit__subassign_06s__7fe0407f00021110_query)
JIT_Q (GB_jit__subassign_06s__7fe0407f00029100_query)
JIT_Q (GB_jit__subassign_06s__7fe0407f00029110_query)
JIT_Q (GB_jit__subassign_06s__7ffc407f00036655_query)
JIT_Q (GB_jit__subassign_06s__7ffc407f00039900_query)
JIT_Q (GB_jit__subassign_06s__7ffc507f00020100_query)
JIT_Q (GB_jit__subassign_06s__7ffc507f00020110_query)
JIT_Q (GB_jit__subassign_08n__00000042999499f3_query)
JIT_Q (GB_jit__subassign_08n__00000050999499f3_query)
JIT_Q (GB_jit__subassign_13__7f00607f00030164_query)
JIT_Q (GB_jit__subassign_13__7fe0607f00030154_query)
JIT_Q (GB_jit__subassign_13__7fe0607f00050154_query)
JIT_Q (GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp_query)
JIT_Q (GB_jit__subassign_23__00000044bbb0bbc2_query)
JIT_Q (GB_jit__subassign_23__00000052aaa0aac3_query)
JIT_Q (GB_jit__subassign_23__00001044bbb0bbc3_query)
JIT_Q (GB_jit__subassign_23__001c0044bbb0bbc1_query)
JIT_Q (GB_jit__subref_bitmap__080009f_query)
JIT_Q (GB_jit__subref_bitmap__08000ef__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subref_bitmap__0800c9f_query)
JIT_Q (GB_jit__subref_bitmap__0800cef__LG_MSF_tuple_fp_query)
JIT_Q (GB_jit__subref_bitmap__0808c6f_query)
JIT_Q (GB_jit__subref_sparse__0bf8490_query)
JIT_Q (GB_jit__subref_sparse__e3f8c90_query)


//------------------------------------------------------------------------------
// GB_prejit_kernels: a list of function pointers to PreJIT kernels
//------------------------------------------------------------------------------

#if ( 129 > 0 )
static void *GB_prejit_kernels [129] =
{
GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__0000400eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot2__038120f1100110c7,
GB_jit__AxB_dot2__0384410b0b3b0ba6,
GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0000c00eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0004c10a0a0a0acb,
GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp,
GB_jit__AxB_dot4__0384c10a0a0a0ac7,
GB_jit__AxB_saxbit__000420fbb00bb08f,
GB_jit__AxB_saxbit__0384610b0b3b0ba6,
GB_jit__AxB_saxpy3__e3f120f110011000,
GB_jit__AxB_saxpy3__e3f120f110011041,
GB_jit__AxB_saxpy3__e3f120f990099000,
GB_jit__AxB_saxpy3__e3f120f990099004,
GB_jit__AxB_saxpy3__e3f120f990099041,
GB_jit__AxB_saxpy3__e3f120f990099045,
GB_jit__AxB_saxpy3__e3f4410b0b2b0b65,
GB_jit__AxB_saxpy3__e3f4410b0b3b0b65,
GB_jit__AxB_saxpy3__fff120f990399000,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp,
GB_jit__add__000203aaa0aaace,
GB_jit__add__000203aaa0aaacf,
GB_jit__add__000214aaa0aaacf,
GB_jit__add__e3f001111019100,
GB_jit__add__e3f101111019100,
GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp,
GB_jit__apply_bind2nd__000014aaa0a8fc8,
GB_jit__apply_bind2nd__000014aaa0a8fcc,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp,
GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp,
GB_jit__apply_unop__004005bb0bbe,
GB_jit__apply_unop__004007aa0aaf,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp,
GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp,
GB_jit__apply_unop__03c005bb0bbd,
GB_jit__bitmap_assign_2_whole__0000003f0002aaa3,
GB_jit__bitmap_assign_2_whole__0000003f000499b3,
GB_jit__bitmap_assign_2_whole__0000103f00020aa3,
GB_jit__bitmap_assign_2_whole__0000803f000299a2,
GB_jit__bitmap_assign_2_whole__0000a03f000301a0,
GB_jit__bitmap_assign_2_whole__001c903f000301a0,
GB_jit__bitmap_assign_4_whole__00e0203f00026890,
GB_jit__bitmap_assign_4_whole__00fc103f00020180,
GB_jit__bitmap_assign_5_whole__00000004bbb0bb82,
GB_jit__build__e9011119,
GB_jit__build__e9066668,
GB_jit__build__e90bbbbb,
GB_jit__build__f9066668,
GB_jit__emult_02__e38005bbb0bbb46,
GB_jit__emult_bitmap__000014bbb2bbbae,
GB_jit__emult_bitmap__000214bbb2bbbae,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp,
GB_jit__ewise_fulln__000014aaa0aaacf,
GB_jit__masker_phase1__dff32000,
GB_jit__masker_phase1__dff32045,
GB_jit__masker_phase1__dff33000,
GB_jit__masker_phase1__dff33041,
GB_jit__masker_phase1__dff33045,
GB_jit__masker_phase1__dff33054,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp,
GB_jit__masker_phase2__fff02900,
GB_jit__masker_phase2__fff03900,
GB_jit__masker_phase2__fff03941,
GB_jit__reduce__14aa2,
GB_jit__reduce__14aa3,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp,
GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp,
GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp,
GB_jit__select_bitmap__00f31beba__LG_MSF_removeEdge_fp,
GB_jit__select_bitmap__00f31beba__LG_MSF_selectEdge_fp,
GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp,
GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp,
GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp,
GB_jit__subassign_02__7f1c417f00001101,
GB_jit__subassign_02__7f1c517f00000101,
GB_jit__subassign_05__07e0207f00026850,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp,
GB_jit__subassign_06n__071c007f00029971,
GB_jit__subassign_06n__07e0007f00021100,
GB_jit__subassign_06n__07e0007f00021110,
GB_jit__subassign_06n__07fc007f00020651,
GB_jit__subassign_06n__07fc007f00020900,
GB_jit__subassign_06n__07fc007f00021100,
GB_jit__subassign_06n__07fc007f00026651,
GB_jit__subassign_06n__07fc007f00029900,
GB_jit__subassign_06n__07fc007f00029951,
GB_jit__subassign_06n__07fc107f00020100,
GB_jit__subassign_06n__07fc107f00020111,
GB_jit__subassign_06n__07fc107f00020140,
GB_jit__subassign_06n__07fc107f00020900,
GB_jit__subassign_06n__07fc107f00020940,
GB_jit__subassign_06n__07fc107f00029900,
GB_jit__subassign_06n__07fc107f00029940,
GB_jit__subassign_06s__7f1c407f00029965,
GB_jit__subassign_06s__7fe0407f00021100,
GB_jit__subassign_06s__7fe0407f00021110,
GB_jit__subassign_06s__7fe0407f00029100,
GB_jit__subassign_06s__7fe0407f00029110,
GB_jit__subassign_06s__7ffc407f00036655,
GB_jit__subassign_06s__7ffc407f00039900,
GB_jit__subassign_06s__7ffc507f00020100,
GB_jit__subassign_06s__7ffc507f00020110,
GB_jit__subassign_08n__00000042999499f3,
GB_jit__subassign_08n__00000050999499f3,
GB_jit__subassign_13__7f00607f00030164,
GB_jit__subassign_13__7fe0607f00030154,
GB_jit__subassign_13__7fe0607f00050154,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp,
GB_jit__subassign_23__00000044bbb0bbc2,
GB_jit__subassign_23__00000052aaa0aac3,
GB_jit__subassign_23__00001044bbb0bbc3,
GB_jit__subassign_23__001c0044bbb0bbc1,
GB_jit__subref_bitmap__080009f,
GB_jit__subref_bitmap__08000ef__LG_MSF_tuple_fp,
GB_jit__subref_bitmap__0800c9f,
GB_jit__subref_bitmap__0800cef__LG_MSF_tuple_fp,
GB_jit__subref_bitmap__0808c6f,
GB_jit__subref_sparse__0bf8490,
GB_jit__subref_sparse__e3f8c90
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_queries: a list of function pointers to PreJIT query kernels
//------------------------------------------------------------------------------

#if ( 129 > 0 )
static void *GB_prejit_queries [129] =
{
GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__0000400eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot2__038120f1100110c7_query,
GB_jit__AxB_dot2__0384410b0b3b0ba6_query,
GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0000c00eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0004c10a0a0a0acb_query,
GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query,
GB_jit__AxB_dot4__0384c10a0a0a0ac7_query,
GB_jit__AxB_saxbit__000420fbb00bb08f_query,
GB_jit__AxB_saxbit__0384610b0b3b0ba6_query,
GB_jit__AxB_saxpy3__e3f120f110011000_query,
GB_jit__AxB_saxpy3__e3f120f110011041_query,
GB_jit__AxB_saxpy3__e3f120f990099000_query,
GB_jit__AxB_saxpy3__e3f120f990099004_query,
GB_jit__AxB_saxpy3__e3f120f990099041_query,
GB_jit__AxB_saxpy3__e3f120f990099045_query,
GB_jit__AxB_saxpy3__e3f4410b0b2b0b65_query,
GB_jit__AxB_saxpy3__e3f4410b0b3b0b65_query,
GB_jit__AxB_saxpy3__fff120f990399000_query,
GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp_query,
GB_jit__add__000203aaa0aaace_query,
GB_jit__add__000203aaa0aaacf_query,
GB_jit__add__000214aaa0aaacf_query,
GB_jit__add__e3f001111019100_query,
GB_jit__add__e3f101111019100_query,
GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp_query,
GB_jit__apply_bind2nd__000014aaa0a8fc8_query,
GB_jit__apply_bind2nd__000014aaa0a8fcc_query,
GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp_query,
GB_jit__apply_unop__004005bb0bbe_query,
GB_jit__apply_unop__004007aa0aaf_query,
GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp_query,
GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp_query,
GB_jit__apply_unop__03c005bb0bbd_query,
GB_jit__bitmap_assign_2_whole__0000003f0002aaa3_query,
GB_jit__bitmap_assign_2_whole__0000003f000499b3_query,
GB_jit__bitmap_assign_2_whole__0000103f00020aa3_query,
GB_jit__bitmap_assign_2_whole__0000803f000299a2_query,
GB_jit__bitmap_assign_2_whole__0000a03f000301a0_query,
GB_jit__bitmap_assign_2_whole__001c903f000301a0_query,
GB_jit__bitmap_assign_4_whole__00e0203f00026890_query,
GB_jit__bitmap_assign_4_whole__00fc103f00020180_query,
GB_jit__bitmap_assign_5_whole__00000004bbb0bb82_query,
GB_jit__build__e9011119_query,
GB_jit__build__e9066668_query,
GB_jit__build__e90bbbbb_query,
GB_jit__build__f9066668_query,
GB_jit__emult_02__e38005bbb0bbb46_query,
GB_jit__emult_bitmap__000014bbb2bbbae_query,
GB_jit__emult_bitmap__000214bbb2bbbae_query,
GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp_query,
GB_jit__ewise_fulln__000014aaa0aaacf_query,
GB_jit__masker_phase1__dff32000_query,
GB_jit__masker_phase1__dff32045_query,
GB_jit__masker_phase1__dff33000_query,
GB_jit__masker_phase1__dff33041_query,
GB_jit__masker_phase1__dff33045_query,
GB_jit__masker_phase1__dff33054_query,
GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp_query,
GB_jit__masker_phase2__fff02900_query,
GB_jit__masker_phase2__fff03900_query,
GB_jit__masker_phase2__fff03941_query,
GB_jit__reduce__14aa2_query,
GB_jit__reduce__14aa3_query,
GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp_query,
GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp_query,
GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp_query,
GB_jit__select_bitmap__00f31beba__LG_MSF_removeEdge_fp_query,
GB_jit__select_bitmap__00f31beba__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp_query,
GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp_query,
GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp_query,
GB_jit__subassign_02__7f1c417f00001101_query,
GB_jit__subassign_02__7f1c517f00000101_query,
GB_jit__subassign_05__07e0207f00026850_query,
GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp_query,
GB_jit__subassign_06n__071c007f00029971_query,
GB_jit__subassign_06n__07e0007f00021100_query,
GB_jit__subassign_06n__07e0007f00021110_query,
GB_jit__subassign_06n__07fc007f00020651_query,
GB_jit__subassign_06n__07fc007f00020900_query,
GB_jit__subassign_06n__07fc007f00021100_query,
GB_jit__subassign_06n__07fc007f00026651_query,
GB_jit__subassign_06n__07fc007f00029900_query,
GB_jit__subassign_06n__07fc007f00029951_query,
GB_jit__subassign_06n__07fc107f00020100_query,
GB_jit__subassign_06n__07fc107f00020111_query,
GB_jit__subassign_06n__07fc107f00020140_query,
GB_jit__subassign_06n__07fc107f00020900_query,
GB_jit__subassign_06n__07fc107f00020940_query,
GB_jit__subassign_06n__07fc107f00029900_query,
GB_jit__subassign_06n__07fc107f00029940_query,
GB_jit__subassign_06s__7f1c407f00029965_query,
GB_jit__subassign_06s__7fe0407f00021100_query,
GB_jit__subassign_06s__7fe0407f00021110_query,
GB_jit__subassign_06s__7fe0407f00029100_query,
GB_jit__subassign_06s__7fe0407f00029110_query,
GB_jit__subassign_06s__7ffc407f00036655_query,
GB_jit__subassign_06s__7ffc407f00039900_query,
GB_jit__subassign_06s__7ffc507f00020100_query,
GB_jit__subassign_06s__7ffc507f00020110_query,
GB_jit__subassign_08n__00000042999499f3_query,
GB_jit__subassign_08n__00000050999499f3_query,
GB_jit__subassign_13__7f00607f00030164_query,
GB_jit__subassign_13__7fe0607f00030154_query,
GB_jit__subassign_13__7fe0607f00050154_query,
GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp_query,
GB_jit__subassign_23__00000044bbb0bbc2_query,
GB_jit__subassign_23__00000052aaa0aac3_query,
GB_jit__subassign_23__00001044bbb0bbc3_query,
GB_jit__subassign_23__001c0044bbb0bbc1_query,
GB_jit__subref_bitmap__080009f_query,
GB_jit__subref_bitmap__08000ef__LG_MSF_tuple_fp_query,
GB_jit__subref_bitmap__0800c9f_query,
GB_jit__subref_bitmap__0800cef__LG_MSF_tuple_fp_query,
GB_jit__subref_bitmap__0808c6f_query,
GB_jit__subref_sparse__0bf8490_query,
GB_jit__subref_sparse__e3f8c90_query
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_names: a list of names of PreJIT kernels
//------------------------------------------------------------------------------

#if ( 129 > 0 )
static char *GB_prejit_names [129] =
{
"GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__0000400eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__0380000eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__0380400eb94eb9b7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot2__038120f1100110c7",
"GB_jit__AxB_dot2__0384410b0b3b0ba6",
"GB_jit__AxB_dot4__0000800eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0000c00eb90eb9cb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0004c10a0a0a0acb",
"GB_jit__AxB_dot4__0380800eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0380c00eb90eb9c7__LG_MSF_tupleMin_fp_LG_MSF_combine_fp",
"GB_jit__AxB_dot4__0384c10a0a0a0ac7",
"GB_jit__AxB_saxbit__000420fbb00bb08f",
"GB_jit__AxB_saxbit__0384610b0b3b0ba6",
"GB_jit__AxB_saxpy3__e3f120f110011000",
"GB_jit__AxB_saxpy3__e3f120f110011041",
"GB_jit__AxB_saxpy3__e3f120f990099000",
"GB_jit__AxB_saxpy3__e3f120f990099004",
"GB_jit__AxB_saxpy3__e3f120f990099041",
"GB_jit__AxB_saxpy3__e3f120f990099045",
"GB_jit__AxB_saxpy3__e3f4410b0b2b0b65",
"GB_jit__AxB_saxpy3__e3f4410b0b3b0b65",
"GB_jit__AxB_saxpy3__fff120f990399000",
"GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_fp_LG_MSF_tuple2nd_fp",
"GB_jit__add__000203aaa0aaace",
"GB_jit__add__000203aaa0aaacf",
"GB_jit__add__000214aaa0aaacf",
"GB_jit__add__e3f001111019100",
"GB_jit__add__e3f101111019100",
"GB_jit__apply_bind1st__000000eb90ef9c3__LG_MSF_combine_fp",
"GB_jit__apply_bind2nd__000014aaa0a8fc8",
"GB_jit__apply_bind2nd__000014aaa0a8fcc",
"GB_jit__apply_unop__0040009e09ee__LG_MSF_get_second_fp",
"GB_jit__apply_unop__0040009e09ef__LG_MSF_get_second_fp",
"GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp",
"GB_jit__apply_unop__004005bb0bbe",
"GB_jit__apply_unop__004007aa0aaf",
"GB_jit__apply_unop__03c0009e09ed__LG_MSF_get_second_fp",
"GB_jit__apply_unop__03c000be0bed__LG_MSF_get_first_fp",
"GB_jit__apply_unop__03c005bb0bbd",
"GB_jit__bitmap_assign_2_whole__0000003f0002aaa3",
"GB_jit__bitmap_assign_2_whole__0000003f000499b3",
"GB_jit__bitmap_assign_2_whole__0000103f00020aa3",
"GB_jit__bitmap_assign_2_whole__0000803f000299a2",
"GB_jit__bitmap_assign_2_whole__0000a03f000301a0",
"GB_jit__bitmap_assign_2_whole__001c903f000301a0",
"GB_jit__bitmap_assign_4_whole__00e0203f00026890",
"GB_jit__bitmap_assign_4_whole__00fc103f00020180",
"GB_jit__bitmap_assign_5_whole__00000004bbb0bb82",
"GB_jit__build__e9011119",
"GB_jit__build__e9066668",
"GB_jit__build__e90bbbbb",
"GB_jit__build__f9066668",
"GB_jit__emult_02__e38005bbb0bbb46",
"GB_jit__emult_bitmap__000014bbb2bbbae",
"GB_jit__emult_bitmap__000214bbb2bbbae",
"GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_fp",
"GB_jit__ewise_fulln__000014aaa0aaacf",
"GB_jit__masker_phase1__dff32000",
"GB_jit__masker_phase1__dff32045",
"GB_jit__masker_phase1__dff33000",
"GB_jit__masker_phase1__dff33041",
"GB_jit__masker_phase1__dff33045",
"GB_jit__masker_phase1__dff33054",
"GB_jit__masker_phase2__1c004e9e__LG_MSF_tuple_fp",
"GB_jit__masker_phase2__fff02900",
"GB_jit__masker_phase2__fff03900",
"GB_jit__masker_phase2__fff03941",
"GB_jit__reduce__14aa2",
"GB_jit__reduce__14aa3",
"GB_jit__rowscale__010200e1e0e1ec7__LG_MSF_tuple2nd_fp",
"GB_jit__select_bitmap__00331beba__LG_MSF_removeEdge_fp",
"GB_jit__select_bitmap__00331beba__LG_MSF_selectEdge_fp",
"GB_jit__select_bitmap__00f31beba__LG_MSF_removeEdge_fp",
"GB_jit__select_bitmap__00f31beba__LG_MSF_selectEdge_fp",
"GB_jit__select_phase1__3f331beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase1__3f331beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase1__3ff31beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase1__3ff31beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase2__3f331beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase2__3f331beb5__LG_MSF_selectEdge_fp",
"GB_jit__select_phase2__3ff31beb5__LG_MSF_removeEdge_fp",
"GB_jit__select_phase2__3ff31beb5__LG_MSF_selectEdge_fp",
"GB_jit__subassign_02__7f1c417f00001101",
"GB_jit__subassign_02__7f1c517f00000101",
"GB_jit__subassign_05__07e0207f00026850",
"GB_jit__subassign_05d__0000207f0004eef0__LG_MSF_tuple_fp",
"GB_jit__subassign_06n__071c007f00029971",
"GB_jit__subassign_06n__07e0007f00021100",
"GB_jit__subassign_06n__07e0007f00021110",
"GB_jit__subassign_06n__07fc007f00020651",
"GB_jit__subassign_06n__07fc007f00020900",
"GB_jit__subassign_06n__07fc007f00021100",
"GB_jit__subassign_06n__07fc007f00026651",
"GB_jit__subassign_06n__07fc007f00029900",
"GB_jit__subassign_06n__07fc007f00029951",
"GB_jit__subassign_06n__07fc107f00020100",
"GB_jit__subassign_06n__07fc107f00020111",
"GB_jit__subassign_06n__07fc107f00020140",
"GB_jit__subassign_06n__07fc107f00020900",
"GB_jit__subassign_06n__07fc107f00020940",
"GB_jit__subassign_06n__07fc107f00029900",
"GB_jit__subassign_06n__07fc107f00029940",
"GB_jit__subassign_06s__7f1c407f00029965",
"GB_jit__subassign_06s__7fe0407f00021100",
"GB_jit__subassign_06s__7fe0407f00021110",
"GB_jit__subassign_06s__7fe0407f00029100",
"GB_jit__subassign_06s__7fe0407f00029110",
"GB_jit__subassign_06s__7ffc407f00036655",
"GB_jit__subassign_06s__7ffc407f00039900",
"GB_jit__subassign_06s__7ffc507f00020100",
"GB_jit__subassign_06s__7ffc507f00020110",
"GB_jit__subassign_08n__00000042999499f3",
"GB_jit__subassign_08n__00000050999499f3",
"GB_jit__subassign_13__7f00607f00030164",
"GB_jit__subassign_13__7fe0607f00030154",
"GB_jit__subassign_13__7fe0607f00050154",
"GB_jit__subassign_23__00000040eee0eec3__LG_MSF_tupleMin_fp",
"GB_jit__subassign_23__00000044bbb0bbc2",
"GB_jit__subassign_23__00000052aaa0aac3",
"GB_jit__subassign_23__00001044bbb0bbc3",
"GB_jit__subassign_23__001c0044bbb0bbc1",
"GB_jit__subref_bitmap__080009f",
"GB_jit__subref_bitmap__08000ef__LG_MSF_tuple_fp",
"GB_jit__subref_bitmap__0800c9f",
"GB_jit__subref_bitmap__0800cef__LG_MSF_tuple_fp",
"GB_jit__subref_bitmap__0808c6f",
"GB_jit__subref_sparse__0bf8490",
"GB_jit__subref_sparse__e3f8c90"
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
    (*nkernels) = 129 ;
    #if ( 129 == 0 )
    (*Kernel_handle) = NULL ;
    (*Query_handle) = NULL ;
    (*Name_handle) = NULL ;
    #else
    (*Kernel_handle) = GB_prejit_kernels ;
    (*Query_handle) = GB_prejit_queries ;
    (*Name_handle) = GB_prejit_names ;
    #endif
}

