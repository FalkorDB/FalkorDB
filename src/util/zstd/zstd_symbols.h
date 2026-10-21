/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

//------------------------------------------------------------------------------
// zstd symbol isolation
//------------------------------------------------------------------------------
//
// This renames every global symbol in our vendored zstd to FDB_*, because
// GraphBLAS links its own copy: deps/GraphBLAS/zstd/zstd_subset, at 1.5.5,
// with its globals renamed to GB_ZSTD_* by the GBZSTD() macro in
// deps/GraphBLAS/Source/zstd_wrapper/GB_zstd.h.
//
// Their comment there is worth reading, because it is why this file can exist:
//
//     "It's possible that the user application has its own copy of the ZSTD
//      library ... To avoid any conflict between multiple copies of the ZSTD
//      library, all global symbols ZSTD_* are renamed"
//
// They renamed to make room for exactly this case. Their list has five gaps.
//
//------------------------------------------------------------------------------
// the five, measured
//------------------------------------------------------------------------------
//
// Do not take this list on trust and do not grep for it - it is the
// intersection of our global symbols with the defined globals of a built
// libgraphblas.a, and reproducing it is two commands:
//
//     nm -g libgraphblas.a | grep -E '^[0-9a-f]+ [TDSBC] ' | awk '{print $3}' \
//       | sed 's/^_//' | sort -u                                  > gb.txt
//     grep '^#define' zstd_symbols.h | awk '{print $2}' | sort -u > ours.txt
//     comm -12 ours.txt gb.txt
//
//   ZSTD_isError                     hard duplicate-symbol link error
//   FSE_isError                      hard duplicate-symbol link error
//   HUF_isError                      hard duplicate-symbol link error
//   g_debuglevel                     hard duplicate-symbol link error
//   g_ZSTD_threading_useless_symbol  'C' (common) our side, 'S' theirs.
//                                    Common symbols merge instead of
//                                    erroring, and the name says what it is:
//                                    a placeholder keeping a translation unit
//                                    non-empty. Harmless.
//
// So FOUR hard collisions, not one, and none of them is optional: all four
// live in objects GraphBLAS's own serialize path pulls into every link.
// Verified by deliberately building without the rename - the linker reports
// three duplicate symbols (the fourth, ZSTD_isError, is renamed even then, by
// the undef fix-up described below) - and then with it, where both copies
// coexist as _FDB_ZSTD_compress and _GB_ZSTD_compress in one image.
//
// WHY EVERY EARLIER ANALYSIS OF THIS, INCLUDING TWO OF MINE, SAID "one
// symbol": they searched for the string "zstd". Three of the five - FSE_isError,
// HUF_isError, g_debuglevel - do not contain it. A name-shaped search cannot
// find a name-shaped gap. The intersection above can, which is why the list
// is derived rather than written.
//
//------------------------------------------------------------------------------
// why the list is generated
//------------------------------------------------------------------------------
//
// The rename covers all 333 globals, not the five, and the list is produced by
// compiling the amalgamation and asking nm:
//
//     nm -g zstd.o | grep -E '^[0-9a-f]+ [TDSBC] ' | awk '{print $3}' | sed 's/^_//'
//
// regenerate.sh then FAILS if any global survives unprefixed. That is a check
// rather than a judgement about what GraphBLAS currently renames, and it has
// already caught three things a hand-written list would have missed:
//
//   * the three non-"zstd" collisions above.
//
//   * ZSTD_isError escaping the rename even when named explicitly, because
//     zstd_common.c does '#undef ZSTD_isError' before defining the real
//     exported function, which removes our macro along with zstd's own.
//     regenerate.sh re-applies the rename after any such undef. Renaming
//     ZSTD_isError by hand and stopping there would have left the collision
//     fully intact while looking fixed.
//
//   * the 'C' class. A [TDSB] filter does not match a common symbol, so
//     g_ZSTD_threading_useless_symbol escaped unnoticed until the filter was
//     widened to [TDSBC].
//
// See PROVENANCE.md for the version, hash, and regeneration command.
//
//------------------------------------------------------------------------------

#define ERR_getErrorString                             FDB_ERR_getErrorString
#define FSE_NCountWriteBound                           FDB_FSE_NCountWriteBound
#define FSE_buildCTable_rle                            FDB_FSE_buildCTable_rle
#define FSE_buildCTable_wksp                           FDB_FSE_buildCTable_wksp
#define FSE_buildDTable_wksp                           FDB_FSE_buildDTable_wksp
#define FSE_compressBound                              FDB_FSE_compressBound
#define FSE_compress_usingCTable                       FDB_FSE_compress_usingCTable
#define FSE_decompress_wksp_bmi2                       FDB_FSE_decompress_wksp_bmi2
#define FSE_getErrorName                               FDB_FSE_getErrorName
#define FSE_isError                                    FDB_FSE_isError
#define FSE_normalizeCount                             FDB_FSE_normalizeCount
#define FSE_optimalTableLog                            FDB_FSE_optimalTableLog
#define FSE_optimalTableLog_internal                   FDB_FSE_optimalTableLog_internal
#define FSE_readNCount                                 FDB_FSE_readNCount
#define FSE_readNCount_bmi2                            FDB_FSE_readNCount_bmi2
#define FSE_versionNumber                              FDB_FSE_versionNumber
#define FSE_writeNCount                                FDB_FSE_writeNCount
#define HIST_add                                       FDB_HIST_add
#define HIST_count                                     FDB_HIST_count
#define HIST_countFast                                 FDB_HIST_countFast
#define HIST_countFast_wksp                            FDB_HIST_countFast_wksp
#define HIST_count_simple                              FDB_HIST_count_simple
#define HIST_count_wksp                                FDB_HIST_count_wksp
#define HIST_isError                                   FDB_HIST_isError
#define HUF_buildCTable_wksp                           FDB_HUF_buildCTable_wksp
#define HUF_cardinality                                FDB_HUF_cardinality
#define HUF_compress1X_repeat                          FDB_HUF_compress1X_repeat
#define HUF_compress1X_usingCTable                     FDB_HUF_compress1X_usingCTable
#define HUF_compress4X_repeat                          FDB_HUF_compress4X_repeat
#define HUF_compress4X_usingCTable                     FDB_HUF_compress4X_usingCTable
#define HUF_compressBound                              FDB_HUF_compressBound
#define HUF_decompress1X1_DCtx_wksp                    FDB_HUF_decompress1X1_DCtx_wksp
#define HUF_decompress1X2_DCtx_wksp                    FDB_HUF_decompress1X2_DCtx_wksp
#define HUF_decompress1X_DCtx_wksp                     FDB_HUF_decompress1X_DCtx_wksp
#define HUF_decompress1X_usingDTable                   FDB_HUF_decompress1X_usingDTable
#define HUF_decompress4X_hufOnly_wksp                  FDB_HUF_decompress4X_hufOnly_wksp
#define HUF_decompress4X_usingDTable                   FDB_HUF_decompress4X_usingDTable
#define HUF_estimateCompressedSize                     FDB_HUF_estimateCompressedSize
#define HUF_getErrorName                               FDB_HUF_getErrorName
#define HUF_getNbBitsFromCTable                        FDB_HUF_getNbBitsFromCTable
#define HUF_isError                                    FDB_HUF_isError
#define HUF_minTableLog                                FDB_HUF_minTableLog
#define HUF_optimalTableLog                            FDB_HUF_optimalTableLog
#define HUF_readCTable                                 FDB_HUF_readCTable
#define HUF_readCTableHeader                           FDB_HUF_readCTableHeader
#define HUF_readDTableX1_wksp                          FDB_HUF_readDTableX1_wksp
#define HUF_readDTableX2_wksp                          FDB_HUF_readDTableX2_wksp
#define HUF_readStats                                  FDB_HUF_readStats
#define HUF_readStats_wksp                             FDB_HUF_readStats_wksp
#define HUF_selectDecoder                              FDB_HUF_selectDecoder
#define HUF_validateCTable                             FDB_HUF_validateCTable
#define HUF_writeCTable_wksp                           FDB_HUF_writeCTable_wksp
#define POOL_add                                       FDB_POOL_add
#define POOL_create                                    FDB_POOL_create
#define POOL_create_advanced                           FDB_POOL_create_advanced
#define POOL_free                                      FDB_POOL_free
#define POOL_joinJobs                                  FDB_POOL_joinJobs
#define POOL_resize                                    FDB_POOL_resize
#define POOL_sizeof                                    FDB_POOL_sizeof
#define POOL_tryAdd                                    FDB_POOL_tryAdd
#define ZSTD_CCtxParams_getParameter                   FDB_ZSTD_CCtxParams_getParameter
#define ZSTD_CCtxParams_init                           FDB_ZSTD_CCtxParams_init
#define ZSTD_CCtxParams_init_advanced                  FDB_ZSTD_CCtxParams_init_advanced
#define ZSTD_CCtxParams_registerSequenceProducer       FDB_ZSTD_CCtxParams_registerSequenceProducer
#define ZSTD_CCtxParams_reset                          FDB_ZSTD_CCtxParams_reset
#define ZSTD_CCtxParams_setParameter                   FDB_ZSTD_CCtxParams_setParameter
#define ZSTD_CCtx_getParameter                         FDB_ZSTD_CCtx_getParameter
#define ZSTD_CCtx_loadDictionary                       FDB_ZSTD_CCtx_loadDictionary
#define ZSTD_CCtx_loadDictionary_advanced              FDB_ZSTD_CCtx_loadDictionary_advanced
#define ZSTD_CCtx_loadDictionary_byReference           FDB_ZSTD_CCtx_loadDictionary_byReference
#define ZSTD_CCtx_refCDict                             FDB_ZSTD_CCtx_refCDict
#define ZSTD_CCtx_refPrefix                            FDB_ZSTD_CCtx_refPrefix
#define ZSTD_CCtx_refPrefix_advanced                   FDB_ZSTD_CCtx_refPrefix_advanced
#define ZSTD_CCtx_refThreadPool                        FDB_ZSTD_CCtx_refThreadPool
#define ZSTD_CCtx_reset                                FDB_ZSTD_CCtx_reset
#define ZSTD_CCtx_setCParams                           FDB_ZSTD_CCtx_setCParams
#define ZSTD_CCtx_setFParams                           FDB_ZSTD_CCtx_setFParams
#define ZSTD_CCtx_setParameter                         FDB_ZSTD_CCtx_setParameter
#define ZSTD_CCtx_setParametersUsingCCtxParams         FDB_ZSTD_CCtx_setParametersUsingCCtxParams
#define ZSTD_CCtx_setParams                            FDB_ZSTD_CCtx_setParams
#define ZSTD_CCtx_setPledgedSrcSize                    FDB_ZSTD_CCtx_setPledgedSrcSize
#define ZSTD_CCtx_trace                                FDB_ZSTD_CCtx_trace
#define ZSTD_CStreamInSize                             FDB_ZSTD_CStreamInSize
#define ZSTD_CStreamOutSize                            FDB_ZSTD_CStreamOutSize
#define ZSTD_DCtx_getParameter                         FDB_ZSTD_DCtx_getParameter
#define ZSTD_DCtx_loadDictionary                       FDB_ZSTD_DCtx_loadDictionary
#define ZSTD_DCtx_loadDictionary_advanced              FDB_ZSTD_DCtx_loadDictionary_advanced
#define ZSTD_DCtx_loadDictionary_byReference           FDB_ZSTD_DCtx_loadDictionary_byReference
#define ZSTD_DCtx_refDDict                             FDB_ZSTD_DCtx_refDDict
#define ZSTD_DCtx_refPrefix                            FDB_ZSTD_DCtx_refPrefix
#define ZSTD_DCtx_refPrefix_advanced                   FDB_ZSTD_DCtx_refPrefix_advanced
#define ZSTD_DCtx_reset                                FDB_ZSTD_DCtx_reset
#define ZSTD_DCtx_setFormat                            FDB_ZSTD_DCtx_setFormat
#define ZSTD_DCtx_setMaxWindowSize                     FDB_ZSTD_DCtx_setMaxWindowSize
#define ZSTD_DCtx_setParameter                         FDB_ZSTD_DCtx_setParameter
#define ZSTD_DDict_dictContent                         FDB_ZSTD_DDict_dictContent
#define ZSTD_DDict_dictSize                            FDB_ZSTD_DDict_dictSize
#define ZSTD_DStreamInSize                             FDB_ZSTD_DStreamInSize
#define ZSTD_DStreamOutSize                            FDB_ZSTD_DStreamOutSize
#define ZSTD_adjustCParams                             FDB_ZSTD_adjustCParams
#define ZSTD_buildBlockEntropyStats                    FDB_ZSTD_buildBlockEntropyStats
#define ZSTD_buildCTable                               FDB_ZSTD_buildCTable
#define ZSTD_buildFSETable                             FDB_ZSTD_buildFSETable
#define ZSTD_cParam_getBounds                          FDB_ZSTD_cParam_getBounds
#define ZSTD_checkCParams                              FDB_ZSTD_checkCParams
#define ZSTD_checkContinuity                           FDB_ZSTD_checkContinuity
#define ZSTD_compress                                  FDB_ZSTD_compress
#define ZSTD_compress2                                 FDB_ZSTD_compress2
#define ZSTD_compressBegin                             FDB_ZSTD_compressBegin
#define ZSTD_compressBegin_advanced                    FDB_ZSTD_compressBegin_advanced
#define ZSTD_compressBegin_advanced_internal           FDB_ZSTD_compressBegin_advanced_internal
#define ZSTD_compressBegin_usingCDict                  FDB_ZSTD_compressBegin_usingCDict
#define ZSTD_compressBegin_usingCDict_advanced         FDB_ZSTD_compressBegin_usingCDict_advanced
#define ZSTD_compressBegin_usingCDict_deprecated       FDB_ZSTD_compressBegin_usingCDict_deprecated
#define ZSTD_compressBegin_usingDict                   FDB_ZSTD_compressBegin_usingDict
#define ZSTD_compressBlock                             FDB_ZSTD_compressBlock
#define ZSTD_compressBlock_btlazy2                     FDB_ZSTD_compressBlock_btlazy2
#define ZSTD_compressBlock_btlazy2_dictMatchState      FDB_ZSTD_compressBlock_btlazy2_dictMatchState
#define ZSTD_compressBlock_btlazy2_extDict             FDB_ZSTD_compressBlock_btlazy2_extDict
#define ZSTD_compressBlock_btopt                       FDB_ZSTD_compressBlock_btopt
#define ZSTD_compressBlock_btopt_dictMatchState        FDB_ZSTD_compressBlock_btopt_dictMatchState
#define ZSTD_compressBlock_btopt_extDict               FDB_ZSTD_compressBlock_btopt_extDict
#define ZSTD_compressBlock_btultra                     FDB_ZSTD_compressBlock_btultra
#define ZSTD_compressBlock_btultra2                    FDB_ZSTD_compressBlock_btultra2
#define ZSTD_compressBlock_btultra_dictMatchState      FDB_ZSTD_compressBlock_btultra_dictMatchState
#define ZSTD_compressBlock_btultra_extDict             FDB_ZSTD_compressBlock_btultra_extDict
#define ZSTD_compressBlock_deprecated                  FDB_ZSTD_compressBlock_deprecated
#define ZSTD_compressBlock_doubleFast                  FDB_ZSTD_compressBlock_doubleFast
#define ZSTD_compressBlock_doubleFast_dictMatchState   FDB_ZSTD_compressBlock_doubleFast_dictMatchState
#define ZSTD_compressBlock_doubleFast_extDict          FDB_ZSTD_compressBlock_doubleFast_extDict
#define ZSTD_compressBlock_fast                        FDB_ZSTD_compressBlock_fast
#define ZSTD_compressBlock_fast_dictMatchState         FDB_ZSTD_compressBlock_fast_dictMatchState
#define ZSTD_compressBlock_fast_extDict                FDB_ZSTD_compressBlock_fast_extDict
#define ZSTD_compressBlock_greedy                      FDB_ZSTD_compressBlock_greedy
#define ZSTD_compressBlock_greedy_dedicatedDictSearch  FDB_ZSTD_compressBlock_greedy_dedicatedDictSearch
#define ZSTD_compressBlock_greedy_dedicatedDictSearch_row FDB_ZSTD_compressBlock_greedy_dedicatedDictSearch_row
#define ZSTD_compressBlock_greedy_dictMatchState       FDB_ZSTD_compressBlock_greedy_dictMatchState
#define ZSTD_compressBlock_greedy_dictMatchState_row   FDB_ZSTD_compressBlock_greedy_dictMatchState_row
#define ZSTD_compressBlock_greedy_extDict              FDB_ZSTD_compressBlock_greedy_extDict
#define ZSTD_compressBlock_greedy_extDict_row          FDB_ZSTD_compressBlock_greedy_extDict_row
#define ZSTD_compressBlock_greedy_row                  FDB_ZSTD_compressBlock_greedy_row
#define ZSTD_compressBlock_lazy                        FDB_ZSTD_compressBlock_lazy
#define ZSTD_compressBlock_lazy2                       FDB_ZSTD_compressBlock_lazy2
#define ZSTD_compressBlock_lazy2_dedicatedDictSearch   FDB_ZSTD_compressBlock_lazy2_dedicatedDictSearch
#define ZSTD_compressBlock_lazy2_dedicatedDictSearch_row FDB_ZSTD_compressBlock_lazy2_dedicatedDictSearch_row
#define ZSTD_compressBlock_lazy2_dictMatchState        FDB_ZSTD_compressBlock_lazy2_dictMatchState
#define ZSTD_compressBlock_lazy2_dictMatchState_row    FDB_ZSTD_compressBlock_lazy2_dictMatchState_row
#define ZSTD_compressBlock_lazy2_extDict               FDB_ZSTD_compressBlock_lazy2_extDict
#define ZSTD_compressBlock_lazy2_extDict_row           FDB_ZSTD_compressBlock_lazy2_extDict_row
#define ZSTD_compressBlock_lazy2_row                   FDB_ZSTD_compressBlock_lazy2_row
#define ZSTD_compressBlock_lazy_dedicatedDictSearch    FDB_ZSTD_compressBlock_lazy_dedicatedDictSearch
#define ZSTD_compressBlock_lazy_dedicatedDictSearch_row FDB_ZSTD_compressBlock_lazy_dedicatedDictSearch_row
#define ZSTD_compressBlock_lazy_dictMatchState         FDB_ZSTD_compressBlock_lazy_dictMatchState
#define ZSTD_compressBlock_lazy_dictMatchState_row     FDB_ZSTD_compressBlock_lazy_dictMatchState_row
#define ZSTD_compressBlock_lazy_extDict                FDB_ZSTD_compressBlock_lazy_extDict
#define ZSTD_compressBlock_lazy_extDict_row            FDB_ZSTD_compressBlock_lazy_extDict_row
#define ZSTD_compressBlock_lazy_row                    FDB_ZSTD_compressBlock_lazy_row
#define ZSTD_compressBound                             FDB_ZSTD_compressBound
#define ZSTD_compressCCtx                              FDB_ZSTD_compressCCtx
#define ZSTD_compressContinue                          FDB_ZSTD_compressContinue
#define ZSTD_compressContinue_public                   FDB_ZSTD_compressContinue_public
#define ZSTD_compressEnd                               FDB_ZSTD_compressEnd
#define ZSTD_compressEnd_public                        FDB_ZSTD_compressEnd_public
#define ZSTD_compressLiterals                          FDB_ZSTD_compressLiterals
#define ZSTD_compressRleLiteralsBlock                  FDB_ZSTD_compressRleLiteralsBlock
#define ZSTD_compressSequences                         FDB_ZSTD_compressSequences
#define ZSTD_compressSequencesAndLiterals              FDB_ZSTD_compressSequencesAndLiterals
#define ZSTD_compressStream                            FDB_ZSTD_compressStream
#define ZSTD_compressStream2                           FDB_ZSTD_compressStream2
#define ZSTD_compressStream2_simpleArgs                FDB_ZSTD_compressStream2_simpleArgs
#define ZSTD_compressSuperBlock                        FDB_ZSTD_compressSuperBlock
#define ZSTD_compress_advanced                         FDB_ZSTD_compress_advanced
#define ZSTD_compress_advanced_internal                FDB_ZSTD_compress_advanced_internal
#define ZSTD_compress_usingCDict                       FDB_ZSTD_compress_usingCDict
#define ZSTD_compress_usingCDict_advanced              FDB_ZSTD_compress_usingCDict_advanced
#define ZSTD_compress_usingDict                        FDB_ZSTD_compress_usingDict
#define ZSTD_convertBlockSequences                     FDB_ZSTD_convertBlockSequences
#define ZSTD_copyCCtx                                  FDB_ZSTD_copyCCtx
#define ZSTD_copyDCtx                                  FDB_ZSTD_copyDCtx
#define ZSTD_copyDDictParameters                       FDB_ZSTD_copyDDictParameters
#define ZSTD_createCCtx                                FDB_ZSTD_createCCtx
#define ZSTD_createCCtxParams                          FDB_ZSTD_createCCtxParams
#define ZSTD_createCCtx_advanced                       FDB_ZSTD_createCCtx_advanced
#define ZSTD_createCDict                               FDB_ZSTD_createCDict
#define ZSTD_createCDict_advanced                      FDB_ZSTD_createCDict_advanced
#define ZSTD_createCDict_advanced2                     FDB_ZSTD_createCDict_advanced2
#define ZSTD_createCDict_byReference                   FDB_ZSTD_createCDict_byReference
#define ZSTD_createCStream                             FDB_ZSTD_createCStream
#define ZSTD_createCStream_advanced                    FDB_ZSTD_createCStream_advanced
#define ZSTD_createDCtx                                FDB_ZSTD_createDCtx
#define ZSTD_createDCtx_advanced                       FDB_ZSTD_createDCtx_advanced
#define ZSTD_createDDict                               FDB_ZSTD_createDDict
#define ZSTD_createDDict_advanced                      FDB_ZSTD_createDDict_advanced
#define ZSTD_createDDict_byReference                   FDB_ZSTD_createDDict_byReference
#define ZSTD_createDStream                             FDB_ZSTD_createDStream
#define ZSTD_createDStream_advanced                    FDB_ZSTD_createDStream_advanced
#define ZSTD_crossEntropyCost                          FDB_ZSTD_crossEntropyCost
#define ZSTD_cycleLog                                  FDB_ZSTD_cycleLog
#define ZSTD_dParam_getBounds                          FDB_ZSTD_dParam_getBounds
#define ZSTD_decodeLiteralsBlock_wrapper               FDB_ZSTD_decodeLiteralsBlock_wrapper
#define ZSTD_decodeSeqHeaders                          FDB_ZSTD_decodeSeqHeaders
#define ZSTD_decodingBufferSize_min                    FDB_ZSTD_decodingBufferSize_min
#define ZSTD_decompress                                FDB_ZSTD_decompress
#define ZSTD_decompressBegin                           FDB_ZSTD_decompressBegin
#define ZSTD_decompressBegin_usingDDict                FDB_ZSTD_decompressBegin_usingDDict
#define ZSTD_decompressBegin_usingDict                 FDB_ZSTD_decompressBegin_usingDict
#define ZSTD_decompressBlock                           FDB_ZSTD_decompressBlock
#define ZSTD_decompressBlock_deprecated                FDB_ZSTD_decompressBlock_deprecated
#define ZSTD_decompressBlock_internal                  FDB_ZSTD_decompressBlock_internal
#define ZSTD_decompressBound                           FDB_ZSTD_decompressBound
#define ZSTD_decompressContinue                        FDB_ZSTD_decompressContinue
#define ZSTD_decompressDCtx                            FDB_ZSTD_decompressDCtx
#define ZSTD_decompressStream                          FDB_ZSTD_decompressStream
#define ZSTD_decompressStream_simpleArgs               FDB_ZSTD_decompressStream_simpleArgs
#define ZSTD_decompress_usingDDict                     FDB_ZSTD_decompress_usingDDict
#define ZSTD_decompress_usingDict                      FDB_ZSTD_decompress_usingDict
#define ZSTD_decompressionMargin                       FDB_ZSTD_decompressionMargin
#define ZSTD_dedicatedDictSearch_lazy_loadDictionary   FDB_ZSTD_dedicatedDictSearch_lazy_loadDictionary
#define ZSTD_defaultCLevel                             FDB_ZSTD_defaultCLevel
#define ZSTD_encodeSequences                           FDB_ZSTD_encodeSequences
#define ZSTD_endStream                                 FDB_ZSTD_endStream
#define ZSTD_estimateCCtxSize                          FDB_ZSTD_estimateCCtxSize
#define ZSTD_estimateCCtxSize_usingCCtxParams          FDB_ZSTD_estimateCCtxSize_usingCCtxParams
#define ZSTD_estimateCCtxSize_usingCParams             FDB_ZSTD_estimateCCtxSize_usingCParams
#define ZSTD_estimateCDictSize                         FDB_ZSTD_estimateCDictSize
#define ZSTD_estimateCDictSize_advanced                FDB_ZSTD_estimateCDictSize_advanced
#define ZSTD_estimateCStreamSize                       FDB_ZSTD_estimateCStreamSize
#define ZSTD_estimateCStreamSize_usingCCtxParams       FDB_ZSTD_estimateCStreamSize_usingCCtxParams
#define ZSTD_estimateCStreamSize_usingCParams          FDB_ZSTD_estimateCStreamSize_usingCParams
#define ZSTD_estimateDCtxSize                          FDB_ZSTD_estimateDCtxSize
#define ZSTD_estimateDDictSize                         FDB_ZSTD_estimateDDictSize
#define ZSTD_estimateDStreamSize                       FDB_ZSTD_estimateDStreamSize
#define ZSTD_estimateDStreamSize_fromFrame             FDB_ZSTD_estimateDStreamSize_fromFrame
#define ZSTD_fillDoubleHashTable                       FDB_ZSTD_fillDoubleHashTable
#define ZSTD_fillHashTable                             FDB_ZSTD_fillHashTable
#define ZSTD_findDecompressedSize                      FDB_ZSTD_findDecompressedSize
#define ZSTD_findFrameCompressedSize                   FDB_ZSTD_findFrameCompressedSize
#define ZSTD_flushStream                               FDB_ZSTD_flushStream
#define ZSTD_frameHeaderSize                           FDB_ZSTD_frameHeaderSize
#define ZSTD_freeCCtx                                  FDB_ZSTD_freeCCtx
#define ZSTD_freeCCtxParams                            FDB_ZSTD_freeCCtxParams
#define ZSTD_freeCDict                                 FDB_ZSTD_freeCDict
#define ZSTD_freeCStream                               FDB_ZSTD_freeCStream
#define ZSTD_freeDCtx                                  FDB_ZSTD_freeDCtx
#define ZSTD_freeDDict                                 FDB_ZSTD_freeDDict
#define ZSTD_freeDStream                               FDB_ZSTD_freeDStream
#define ZSTD_fseBitCost                                FDB_ZSTD_fseBitCost
#define ZSTD_generateSequences                         FDB_ZSTD_generateSequences
#define ZSTD_get1BlockSummary                          FDB_ZSTD_get1BlockSummary
#define ZSTD_getBlockSize                              FDB_ZSTD_getBlockSize
#define ZSTD_getCParams                                FDB_ZSTD_getCParams
#define ZSTD_getCParamsFromCCtxParams                  FDB_ZSTD_getCParamsFromCCtxParams
#define ZSTD_getCParamsFromCDict                       FDB_ZSTD_getCParamsFromCDict
#define ZSTD_getDecompressedSize                       FDB_ZSTD_getDecompressedSize
#define ZSTD_getDictID_fromCDict                       FDB_ZSTD_getDictID_fromCDict
#define ZSTD_getDictID_fromDDict                       FDB_ZSTD_getDictID_fromDDict
#define ZSTD_getDictID_fromDict                        FDB_ZSTD_getDictID_fromDict
#define ZSTD_getDictID_fromFrame                       FDB_ZSTD_getDictID_fromFrame
#define ZSTD_getErrorCode                              FDB_ZSTD_getErrorCode
#define ZSTD_getErrorName                              FDB_ZSTD_getErrorName
#define ZSTD_getErrorString                            FDB_ZSTD_getErrorString
#define ZSTD_getFrameContentSize                       FDB_ZSTD_getFrameContentSize
#define ZSTD_getFrameHeader                            FDB_ZSTD_getFrameHeader
#define ZSTD_getFrameHeader_advanced                   FDB_ZSTD_getFrameHeader_advanced
#define ZSTD_getFrameProgression                       FDB_ZSTD_getFrameProgression
#define ZSTD_getParams                                 FDB_ZSTD_getParams
#define ZSTD_getSeqStore                               FDB_ZSTD_getSeqStore
#define ZSTD_getcBlockSize                             FDB_ZSTD_getcBlockSize
#define ZSTD_initCStream                               FDB_ZSTD_initCStream
#define ZSTD_initCStream_advanced                      FDB_ZSTD_initCStream_advanced
#define ZSTD_initCStream_internal                      FDB_ZSTD_initCStream_internal
#define ZSTD_initCStream_srcSize                       FDB_ZSTD_initCStream_srcSize
#define ZSTD_initCStream_usingCDict                    FDB_ZSTD_initCStream_usingCDict
#define ZSTD_initCStream_usingCDict_advanced           FDB_ZSTD_initCStream_usingCDict_advanced
#define ZSTD_initCStream_usingDict                     FDB_ZSTD_initCStream_usingDict
#define ZSTD_initDStream                               FDB_ZSTD_initDStream
#define ZSTD_initDStream_usingDDict                    FDB_ZSTD_initDStream_usingDDict
#define ZSTD_initDStream_usingDict                     FDB_ZSTD_initDStream_usingDict
#define ZSTD_initStaticCCtx                            FDB_ZSTD_initStaticCCtx
#define ZSTD_initStaticCDict                           FDB_ZSTD_initStaticCDict
#define ZSTD_initStaticCStream                         FDB_ZSTD_initStaticCStream
#define ZSTD_initStaticDCtx                            FDB_ZSTD_initStaticDCtx
#define ZSTD_initStaticDDict                           FDB_ZSTD_initStaticDDict
#define ZSTD_initStaticDStream                         FDB_ZSTD_initStaticDStream
#define ZSTD_insertAndFindFirstIndex                   FDB_ZSTD_insertAndFindFirstIndex
#define ZSTD_insertBlock                               FDB_ZSTD_insertBlock
#define ZSTD_invalidateRepCodes                        FDB_ZSTD_invalidateRepCodes
#define ZSTD_isError                                   FDB_ZSTD_isError
#define ZSTD_isFrame                                   FDB_ZSTD_isFrame
#define ZSTD_isSkippableFrame                          FDB_ZSTD_isSkippableFrame
#define ZSTD_ldm_adjustParameters                      FDB_ZSTD_ldm_adjustParameters
#define ZSTD_ldm_blockCompress                         FDB_ZSTD_ldm_blockCompress
#define ZSTD_ldm_fillHashTable                         FDB_ZSTD_ldm_fillHashTable
#define ZSTD_ldm_generateSequences                     FDB_ZSTD_ldm_generateSequences
#define ZSTD_ldm_getMaxNbSeq                           FDB_ZSTD_ldm_getMaxNbSeq
#define ZSTD_ldm_getTableSize                          FDB_ZSTD_ldm_getTableSize
#define ZSTD_ldm_skipRawSeqStoreBytes                  FDB_ZSTD_ldm_skipRawSeqStoreBytes
#define ZSTD_ldm_skipSequences                         FDB_ZSTD_ldm_skipSequences
#define ZSTD_loadCEntropy                              FDB_ZSTD_loadCEntropy
#define ZSTD_loadDEntropy                              FDB_ZSTD_loadDEntropy
#define ZSTD_maxCLevel                                 FDB_ZSTD_maxCLevel
#define ZSTD_mergeBlockDelimiters                      FDB_ZSTD_mergeBlockDelimiters
#define ZSTD_minCLevel                                 FDB_ZSTD_minCLevel
#define ZSTD_nextInputType                             FDB_ZSTD_nextInputType
#define ZSTD_nextSrcSizeToDecompress                   FDB_ZSTD_nextSrcSizeToDecompress
#define ZSTD_noCompressLiterals                        FDB_ZSTD_noCompressLiterals
#define ZSTD_readSkippableFrame                        FDB_ZSTD_readSkippableFrame
#define ZSTD_referenceExternalSequences                FDB_ZSTD_referenceExternalSequences
#define ZSTD_registerSequenceProducer                  FDB_ZSTD_registerSequenceProducer
#define ZSTD_resetCStream                              FDB_ZSTD_resetCStream
#define ZSTD_resetDStream                              FDB_ZSTD_resetDStream
#define ZSTD_resetSeqStore                             FDB_ZSTD_resetSeqStore
#define ZSTD_reset_compressedBlockState                FDB_ZSTD_reset_compressedBlockState
#define ZSTD_row_update                                FDB_ZSTD_row_update
#define ZSTD_selectBlockCompressor                     FDB_ZSTD_selectBlockCompressor
#define ZSTD_selectEncodingType                        FDB_ZSTD_selectEncodingType
#define ZSTD_seqToCodes                                FDB_ZSTD_seqToCodes
#define ZSTD_sequenceBound                             FDB_ZSTD_sequenceBound
#define ZSTD_sizeof_CCtx                               FDB_ZSTD_sizeof_CCtx
#define ZSTD_sizeof_CDict                              FDB_ZSTD_sizeof_CDict
#define ZSTD_sizeof_CStream                            FDB_ZSTD_sizeof_CStream
#define ZSTD_sizeof_DCtx                               FDB_ZSTD_sizeof_DCtx
#define ZSTD_sizeof_DDict                              FDB_ZSTD_sizeof_DDict
#define ZSTD_sizeof_DStream                            FDB_ZSTD_sizeof_DStream
#define ZSTD_splitBlock                                FDB_ZSTD_splitBlock
#define ZSTD_toFlushNow                                FDB_ZSTD_toFlushNow
#define ZSTD_updateTree                                FDB_ZSTD_updateTree
#define ZSTD_versionNumber                             FDB_ZSTD_versionNumber
#define ZSTD_versionString                             FDB_ZSTD_versionString
#define ZSTD_writeLastEmptyBlock                       FDB_ZSTD_writeLastEmptyBlock
#define ZSTD_writeSkippableFrame                       FDB_ZSTD_writeSkippableFrame
#define g_ZSTD_threading_useless_symbol                FDB_g_ZSTD_threading_useless_symbol
#define g_debuglevel                                   FDB_g_debuglevel
