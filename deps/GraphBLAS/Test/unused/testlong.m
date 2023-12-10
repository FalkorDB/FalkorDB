function testlong
%TESTLONG run long GraphBLAS tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_init ;

testall_time = tic ;

if (nargin < 2)
    % run the shorter tests by default
    longtests = 0 ;
end

if (nargin < 1)
    threads = [ ] ;
end
if (isempty (threads))
    threads {1} = [4 1] ;
end
t = threads ;

% single thread
s {1} = [1 1] ;

extra {1} = [4 1] ;
extra {2} = [1 1] ;

% clear the statement coverage counts
clear global GraphBLAS_grbcov

% use built-in complex data types by default
GB_builtin_complex_set (true) ;

% many of the tests use spok in SuiteSparse, a copy of which is
% included here in GraphBLAS/Test/spok.
addpath ('../Test/spok') ;

try
    spok (sparse (1)) ;
catch
    here = pwd ;
    cd ../Test/spok ;
    spok_install ;
    cd (here) ;
end

logstat ;             % start the log.txt
hack = GB_mex_hack ;

% start with the Werk stack enabled
hack (2) = 0 ; GB_mex_hack (hack) ;

%===============================================================================
% The following tests are not required for statement coverage.  Some need
% other packages in SuiteSparse (CSparse, SSMULT, ssget).  By default, these
% tests are not run.  To install them, see test_other.m.  Timing is with malloc
% debugging turned off.

% ------------------------ % ---- % ------------------------------
% test script              % time % description
% ------------------------ % ---- % ------------------------------

logstat ('test03' ,t) ;    %    0 % random matrix tests
logstat ('test03' ,s) ;    %    0 % random matrix tests
logstat ('test05',t) ;     %      % quick setElement test, with typecasting
logstat ('test06(936)',t); %      % performance test GrB_mxm on all semirings
logstat ('test07',t) ;     %    0 % quick test GB_mex_subassign
logstat ('test07',s) ;     %    0 % quick test GB_mex_subassign
logstat ('test07b',t) ;    %      % quick test GB_mex_assign
logstat ('test09b',t) ;    %      % duplicate I,J test of GB_mex_assign

logstat ('test13',t) ;     %      % simple tests of GB_mex_transpose
logstat ('test15',t) ;            % simple test of GB_mex_AxB
logstat ('test16' ,t) ;    %  177 % user-defined complex operators

logstat ('test20',t) ;            % quick test of GB_mex_mxm on a few semirings
logstat ('test20(1)',t) ;  %      % test of GB_mex_mxm on all built-in semirings
logstat ('test21',s) ;     %   41 % quick test of GB_mex_subassign
logstat ('test21(1)',t) ;  %      % exhaustive test of GB_mex_subassign
logstat ('test22',t) ;     %      % quick test of GB_mex_transpose
logstat ('test23(1)',t) ;  %      % exhaustive test of GB_*_build
logstat ('test24',t) ;     %   42 % test of GrB_Matrix_reduce
logstat ('test24(1)',t) ;  %      % exhaustive test of GrB_Matrix_reduce
logstat ('test25',t) ;     %      % long test of GxB_select
logstat ('test26',t) ;     %   .6 % quick test of GxB_select
logstat ('test26(1)',t) ;  %      % performance test of GxB_select (use ssget)
logstat ('test27',t) ;     %   13 % quick test of GxB_select (LoHi_band)
logstat ('test28',t) ;     %    1 % mxm with aliased inputs, C<C> = accum(C,C*C)

logstat ('test30') ;       %   11 % GB_mex_subassign, scalar expansion
logstat ('test30b') ;      %    9 % performance GB_mex_assign, scalar expansion
logstat ('test31',t) ;     %      % simple tests of GB_mex_transpose
logstat ('test32',t) ;     %      % quick GB_mex_mxm test
logstat ('test33',t) ;     %      % create a semiring
logstat ('test34',t) ;     %      % quick GB_mex_Matrix_eWiseAdd test
logstat ('test35') ;       %      % performance test for GrB_extractTuples
logstat ('test36') ;       %      % performance test for GB_mex_Matrix_subref
logstat ('test38',t) ;     %      % GB_mex_transpose with matrix collection
logstat ('test39') ;       %      % GrB_transpose, GB_*_add and eWiseAdd
logstat ('test39(0)') ;    %   55 % GrB_transpose, GB_*_add and eWiseAdd

logstat ('test40',t) ;     %      % GrB_Matrix_extractElement, and Vector
logstat ('test41',t) ;     %      % test of GB_mex_AxB
logstat ('test42') ;       %      % performance tests for GB_mex_Matrix_build
logstat ('test43',t) ;     %      % performance tests for GB_mex_Matrix_subref
logstat ('test44',t) ;     %    5 % test qsort
logstat ('test45(0)',t) ;  %  334 % test GB_mex_setElement and build
logstat ('test46') ;       %      % performance test GB_mex_subassign
logstat ('test46b') ;      %      % performance test GB_mex_assign
logstat ('test47',t) ;     %      % performance test of GrB_vxm
logstat ('test48') ;       %      % performance test of GrB_mxm
logstat ('test49') ;       %      % performance test of GrB_mxm (dot, A'*B)

logstat ('test50',t) ;     %      % test GB_mex_AxB on larger matrix
logstat ('test51') ;       %      % performance test GB_mex_subassign
logstat ('test51b') ;      %      % performance test GB_mex_assign, multiple ops
logstat ('test52',t) ;     %      % performance of A*B with tall mtx, AdotB, AxB
logstat ('test53',t) ;     %      % exhaustive test of GB_mex_Matrix_extract
logstat ('test55',t) ;     %      % GxB_subassign, dupl, built-in vs GraphBLAS
logstat ('test55b',t) ;    %      % GrB_assign, duplicates, built-in vs GrB
logstat ('test56',t) ;     %      % test GrB_*_build
logstat ('test57',t) ;     %      % test operator on large uint32 values
logstat ('test58(0)') ;    %      % longer GB_mex_Matrix_eWiseAdd performance
logstat ('test58') ;       %      % test GrB_eWiseAdd
logstat ('test59',t) ;     %      % test GrB_mxm

logstat ('test60',t) ;     %      % test min and max operators with NaNs
logstat ('test61') ;       %      % performance test of GrB_eWiseMult
logstat ('test62',t) ;     %      % exhaustive test of GrB_apply
logstat ('test63',t) ;     %      % GB_mex_op and operator tests
logstat ('test64',t) ;     %      % GB_mex_subassign, scalar expansion
logstat ('test64b',t) ;    %      % GrB_*_assign, scalar expansion
logstat ('test65',t) ;     %      % test type casting
logstat ('test66',t) ;     %      % quick test for GrB_Matrix_reduce
logstat ('test67',t) ;     %      % quick test for GrB_apply
logstat ('test68',t) ;

logstat ('test72' ,t) ;    %    0 % several special cases
logstat ('test73',t) ;     %      % performance of C = A*B, with mask
logstat ('test75',t) ;     %      % test GrB_mxm A'*B on all semirings
logstat ('test77',t) ;     %  450 % long tests of GrB_kronecker
logstat ('test78',t) ;     %    1 % quick test of hypersparse subref
logstat ('test79',t) ;     %      % run all in SuiteSparse Collection

logstat ('test85',t) ;     %    0 % GrB_transpose (1-by-n with typecasting)
logstat ('test86',t) ;     %      % performance test of of GrB_Matrix_extract
logstat ('test87',t) ;     %      % performance test of GrB_mxm
logstat ('test88',t) ;            % hypersparse matrices with hash-based method
logstat ('test89',t) ;     %      % performance test of complex A*B

logstat ('test90',t) ;     %    1 % test user-defined semirings
logstat ('test91',t) ;     %      % test subref performance on dense vectors
logstat ('test92' ,t) ;    %   .1 % GB_subref: symbolic case
logstat ('test95',t) ;     %      % performance test for GrB_transpose
logstat ('test96',t) ;     %   16 % A*B using dot product
logstat ('test97',t) ;     %    0 % GB_mex_assign, scalar expansion and zombies
logstat ('test98',t) ;     %      % GB_mex_mxm, typecast on the fly
logstat ('test99',t) ;     %   20 % GB_mex_transpose w/ explicit 0s in the Mask

logstat ('test101',t) ;    %    1 % import and export
logstat ('test102',t);     %    1 % GB_AxB_saxpy3_flopcount
logstat ('test103',t) ;    %      % GrB_transpose aliases
logstat ('test105',t) ;    %    2 % eWiseAdd for hypersparse
logstat ('test106',t) ;    %    4 % GxB_subassign with alias
logstat ('test107',t) ;    %    2 % monoids with terminal values

logstat ('test110',t) ;    %    0 % binary search of M(:,j) in accum/mask
logstat ('test111',t) ;    %      % performance test for eWiseAdd
logstat ('test112',t) ;    %      % test row/col scale
logstat ('test113',t) ;    %      % performance tests for GrB_kron
logstat ('test114',t) ;    %      % performance of reduce-to-scalar
logstat ('test115',t) ;    %   10 % GrB_assign with duplicate indices
logstat ('test116',t) ;    %      % performance tests for GrB_assign
logstat ('test117',t) ;    %      % performance tests for GrB_assign
logstat ('test118',t) ;    %      % performance tests for GrB_assign
logstat ('test119',t) ;    %      % performance tests for GrB_assign

logstat ('test120',t) ;    %      % performance tests for GrB_assign
logstat ('test121',t) ;    %      % performance tests for GrB_assign
logstat ('test122',t) ;    %      % performance tests for GrB_assign
logstat ('test126',t) ;    %    7 % GrB_reduce to vector; very sparse matrix 

logstat ('test131',t) ;    %   .1 % GrB_Matrix_clear
logstat ('test134',t) ;    %  105 % quick test of GxB_select

logstat ('test143',t) ;    %   37 % mxm, special cases
logstat ('test146',t) ;    %   .1 % expand scalar
logstat ('test147',t) ;           % C<M>=A*B with very sparse M
logstat ('test149',t) ;           % test fine hash tasks for C<!M>=A*B

logstat ('test158',t) ;    %  10  % test colscale and rowscale

logstat ('test161',t) ;    %      % test A*B*E
logstat ('test163',t) ;    %   .6 % test C<!M>=A'*B where C and M are sparse
logstat ('test164',t) ;    %    0 % test dot method
logstat ('test166',t) ;    %   .1 % test GxB_select with a dense matrix
logstat ('test167',t) ;    %   .2 % test C<M>=A*B with very sparse M, different types
logstat ('test168',t) ;           % test C=A+B with C and B full, A bitmap
logstat ('test169',t) ;    %    0 % test C<!M>=A+B with C sparse, M hyper, A and B sparse

logstat ('test171',t) ;    %    1 % test conversion and GB_memset
logstat ('test175',t) ;    %    8 % test142 updated
logstat ('test177',t) ;    %  1.2 % test C<!M>=A*B, C and B bitmap, M and A sparse

logstat ('test180',t) ;    %  16  % test assign and subassign (multi threaded)

logstat ('test190',t) ;    %   .3 % test dense matrix for C<!M>=A*B
logstat ('test198',t) ;    %   .1 % test apply with C=op(C)

logstat ('test205',t) ;    %    0 % test iso kron
logstat ('test217',t) ;    %    0 % test C<repl>(I,J)=A, bitmap assign
logstat ('test218',t) ;    %    0 % test C=A+B, C and A are full, B is bitmap

logstat ('testc1',t) ;     %      % test complex operators
logstat ('testc2',t) ;     %      % test complex A*B, A'*B, A*B', A'*B', A+B
logstat ('testc3',t) ;     %      % test complex GrB_extract
logstat ('testc4',t) ;     %      % test complex extractElement and setElement
logstat ('testc5',t) ;     %      % test complex subref
logstat ('testc6',t) ;     %      % test complex apply
logstat ('testc7',t) ;     %      % test complex assign
logstat ('testc8',t) ;     %      % test complex eWiseAdd and eWiseMult
logstat ('testc9',t) ;     %      % test complex extractTuples
logstat ('testca',t) ;     %      % test complex mxm, mxv, and vxm
logstat ('testcb',t) ;     %      % test complex reduce
logstat ('testcc',t) ;     %      % test complex transpose

