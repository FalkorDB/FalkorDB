function test251(tasks)
%TEST251 test dot4 for plus-pair semirings
% GB_AxB_dot4 computes C+=A'*B when C is dense.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: test coverage should start with an empty JIT cache.

fprintf ('test251 ------------ C+=A''*B when C is dense (plus-pair)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    { 0, 1, 1}, ... % ( 98,  98)
    { 1, 1, 1}, ... % ( 22, 120)
    { 2, 1, 1}, ... % ( 13, 133)
    { 3, 1, 1}, ... % ( 31, 164)
    { 4, 1, 1}, ... % ( 11, 175)
    { 5, 1, 1}, ... % ( 11, 186)
    { 6, 1, 1}, ... % ( 13, 199)
    { 7, 1, 1}, ... % ( 18, 217)
    { 8, 1, 1}, ... % ( 13, 230)
    { 9, 1, 1}, ... % (  9, 239)
    {10, 1, 1}, ... % (  9, 248)
    {11, 1, 1}, ... % (  4, 252)
    {12, 1, 1}, ... % ( 16, 268)
    {13, 1, 1}, ... % ( 17, 285)
    { 0, 1, 2}, ... % (  7, 292)
    { 0, 1, 4}, ... % ( 16, 308)
    { 1, 1, 4}, ... % (  2, 310)
    { 2, 1, 4}, ... % (  1, 311)
    { 3, 1, 4}, ... % (  1, 312)
    { 4, 1, 4}, ... % (  1, 313)
    { 5, 1, 4}, ... % (  1, 314)
    { 6, 1, 4}, ... % (  1, 315)
    { 7, 1, 4}, ... % (  1, 316)
    { 8, 1, 4}, ... % (  1, 317)
    { 9, 1, 4}, ... % (  1, 318)
    {10, 1, 4}, ... % (  1, 319)
    {11, 1, 4}, ... % (  1, 320)
    {12, 1, 4}, ... % (  1, 321)
    {13, 1, 4}, ... % (  1, 322)
    { 0, 1, 8}, ... % (  5, 327)
    { 1, 1, 8}, ... % (  1, 328)
    { 2, 1, 8}, ... % (  2, 330)
    { 4, 1, 8}, ... % (  2, 332)
    { 5, 1, 8}, ... % (  2, 334)
    { 0, 2, 1}, ... % (  5, 339)
    { 0, 2, 2}, ... % (  1, 340)
    { 0, 4, 1}, ... % (  8, 348)
    { 1, 4, 1}, ... % (  3, 351)
    { 2, 4, 1}, ... % (  1, 352)
    { 3, 4, 1}, ... % (  1, 353)
    { 4, 4, 1}, ... % (  1, 354)
    { 5, 4, 1}, ... % (  1, 355)
    { 6, 4, 1}, ... % (  1, 356)
    { 7, 4, 1}, ... % (  1, 357)
    { 8, 4, 1}, ... % (  1, 358)
    { 9, 4, 1}, ... % (  1, 359)
    {10, 4, 1}, ... % (  1, 360)
    {11, 4, 1}, ... % (  1, 361)
    {12, 4, 1}, ... % (  1, 362)
    {13, 4, 1}, ... % (  1, 363)
    { 1, 4, 2}, ... % (  2, 365)
    { 0, 4, 4}, ... % ( 14, 379)
    { 2, 4, 4}, ... % (  1, 380)
    { 3, 4, 4}, ... % (  1, 381)
    { 4, 4, 4}, ... % (  1, 382)
    { 5, 4, 4}, ... % (  1, 383)
    { 6, 4, 4}, ... % (  1, 384)
    { 7, 4, 4}, ... % (  1, 385)
    { 8, 4, 4}, ... % (  1, 386)
    { 9, 4, 4}, ... % (  1, 387)
    {10, 4, 4}, ... % (  1, 388)
    {11, 4, 4}, ... % (  1, 389)
    {12, 4, 4}, ... % (  1, 390)
    {13, 4, 4}, ... % (  1, 391)
    { 0, 4, 8}, ... % (  3, 394)
    { 0, 8, 1}, ... % (  5, 399)
    { 1, 8, 1}, ... % (  4, 403)
    { 2, 8, 1}, ... % (  2, 405)
    { 4, 8, 1}, ... % (  2, 407)
    { 5, 8, 1}, ... % (  2, 409)
    { 0, 8, 2}, ... % (  1, 410)
    { 1, 8, 2}, ... % (  1, 411)
    { 0, 8, 4}, ... % (  3, 414)
    { 0, 8, 8}, ... % (  3, 417)
    { 1, 8, 8}, ... % (  1, 418)
    { 2, 8, 8}, ... % (  2, 420)
    { 4, 8, 8}, ... % (  2, 422)
    { 5, 8, 8}, ... % (  2, 424)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

rng ('default') ;

plus_pair.add = 'plus' ;
plus_pair.multiply = 'oneb' ;   % same as pair
[~, ~, ~, types, ~, ~, ~,] = GB_spec_opsall ;
types = types.all ;

add_op.opname = 'plus' ;
dtn_dot = struct ('axb', 'dot', 'inp0', 'tran') ;
dtn_sax = struct ('axb', 'saxpy', 'inp0', 'tran') ;

n = 20 ;
C = GB_spec_random (n, n, inf, 100, 'double') ;
C.sparsity = 8 ;
C0.matrix = sparse (n, n) ;

% create the test matrices
A_matrices = cell (8) ;
B_matrices = cell (8,8) ;
for A_sparsity = [1 2 4 8]
    if (A_sparsity == 8)
        A = GB_spec_random (n, n, inf, 100, 'double') ;
    else
        A = GB_spec_random (n, n, 0.1, 100, 'double') ;
    end
    A_matrices {A_sparsity} = A ;
    for B_sparsity = [1 2 4 8]
        if (B_sparsity == 8)
            B = GB_spec_random (n, n, inf, 100, 'double') ;
        else
            B = GB_spec_random (n, n, 0.1, 100, 'double') ;
        end
        B_matrices {A_sparsity,B_sparsity} = B ;
    end
end

for kk = 1:length(tasks)
    task = tasks {kk} ;
    k = task {1} ;
    A_sparsity = task {2} ;
    B_sparsity = task {3} ;

    A = A_matrices {A_sparsity} ;
    A.sparsity = A_sparsity ;

    B = B_matrices {A_sparsity,B_sparsity} ;
    B.sparsity = B_sparsity ;

    if (k == 0)
        type = 'logical' ;
        add_op.opname = 'xor' ;
        plus_pair.add = 'xor' ;
    else
        type = types {k} ;
        add_op.opname = 'plus' ;
        plus_pair.add = 'plus' ;
    end
    plus_pair.class = type ;
    add_op.optype = type ;
    if (test_contains (type, 'single'))
        tol = 1e-5 ;
    else
        tol = 1e-10 ;
    end

    A.class = type ;
    B.class = type ;
    C0.class = type ;
    C.class = type ;

    % X = C + A'*B using dot4
    X2 = GB_mex_mxm  (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
    X1 = GB_spec_mxm (C, [ ], add_op, plus_pair, A, B, dtn_dot) ;
    GB_spec_compare (X1, X2, 0, tol) ;

    % X = A'*B using dot2/dot3
    X2 = GB_mex_mxm  (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
    X1 = GB_spec_mxm (C0, [ ], [ ], plus_pair, A, B, dtn_dot) ;
    GB_spec_compare (X1, X2, 0, tol) ;

    % X = C + A'*B using saxpy
    X2 = GB_mex_mxm  (C, [ ], add_op, plus_pair, A, B, dtn_sax) ;
    X1 = GB_spec_mxm (C, [ ], add_op, plus_pair, A, B, dtn_sax) ;
    GB_spec_compare (X1, X2) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            fprintf ('{%2d, %d, %d},', ...
                k, A_sparsity, B_sparsity) ;
            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end

end

fprintf ('\n') ;
fprintf ('test251: all tests passed\n') ;

