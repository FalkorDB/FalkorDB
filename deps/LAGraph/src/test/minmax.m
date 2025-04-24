% LAGraph/src/test/minmax.m

% LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
% SPDX-License-Identifier: BSD-2-Clause
% See additional acknowledgments in the LICENSE file,
% or contact permission@sei.cmu.edu for the full terms.

files = {
    'A2.mtx',
    'A.mtx',
    'bcsstk13.mtx',
    'comments_cover.mtx',
    'comments_full.mtx',
    'comments_west0067.mtx',
    'complex.mtx',
    'cover.mtx',
    'cover_structure.mtx',
    'cryg2500.mtx',
    'empty.mtx',
    'full.mtx',
    'full_noheader.mtx',
    'full_symmetric.mtx',
    'jagmesh7.mtx',
    'karate.mtx',
    'ldbc-cdlp-directed-example.mtx',
    'ldbc-cdlp-undirected-example.mtx',
    'ldbc-directed-example-bool.mtx',
    'ldbc-directed-example.mtx',
    'ldbc-directed-example-unweighted.mtx',
    'ldbc-undirected-example-bool.mtx',
    'ldbc-undirected-example.mtx',
    'ldbc-undirected-example-unweighted.mtx',
    'ldbc-wcc-example.mtx',
    'LFAT5.mtx',
    'LFAT5_two.mtx',
    'lp_afiro.mtx',
    'lp_afiro_structure.mtx',
    'matrix_bool.mtx',
    'matrix_fp32.mtx',
    'matrix_fp32_structure.mtx',
    'matrix_fp64.mtx',
    'matrix_int16.mtx',
    'matrix_int32.mtx',
    'matrix_int64.mtx',
    'matrix_int8.mtx',
    'matrix_uint16.mtx',
    'matrix_uint32.mtx',
    'matrix_uint64.mtx',
    'matrix_uint8.mtx',
    'msf1.mtx',
    'msf2.mtx',
    'msf3.mtx',
    'olm1000.mtx',
    'pushpull.mtx',
    'sample2.mtx',
    'sample.mtx',
    'skew_fp32.mtx',
    'skew_fp64.mtx',
    'skew_int16.mtx',
    'skew_int32.mtx',
    'skew_int64.mtx',
    'skew_int8.mtx',
    'sources_7.mtx',
    'structure.mtx',
    'test_BF.mtx',
    'test_FW_1000.mtx',
    'test_FW_2003.mtx',
    'test_FW_2500.mtx',
    'tree-example.mtx',
    'west0067_jumbled.mtx',
    'west0067.mtx',
    'west0067_noheader.mtx',
    'zenios.mtx' } ;

nmat = length (files) ;

for k = 1:nmat
    file = files {k} ;
    [A Z] = mread (['../../data/' file]) ;
    [i j x] = find (A) ;
    znz = nnz (Z) ;

    if (~isreal (A))
        fprintf ('// %s is complex\n', file) ;
    else
        emin = 0 ;
        emax = 0 ;
        if (length (x) > 0)
            emin = min (x) ;
            emax = max (x) ;
        end
        if (znz > 0)
            emin = min (emin, 0) ;
            emax = max (emax, 0) ;
        end
        fprintf ('{ %18.16g, %18.16g, "%s"} ,\n', emin, emax, file) ;
    end
end

