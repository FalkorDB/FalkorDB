function test14(tasks)
%TEST14 test GrB_reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest14: reduce to column and scalar\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {   'min',  1, 0, 0}, ... % ( 19,  19)
    {   'max',  1, 0, 0}, ... % ( 10,  29)
    {   'any',  1, 0, 0}, ... % ( 10,  39)
    {    'or',  1, 0, 0}, ... % (  1,  40)
    {   'and',  1, 0, 0}, ... % (  1,  41)
    {   'xor',  1, 0, 0}, ... % (  8,  49)
    {    'eq',  1, 0, 0}, ... % ( 10,  59)
    {   'min',  1, 0, 1}, ... % (  7,  66)
    {   'min',  1, 1, 0}, ... % (  2,  68)
    {   'min',  2, 0, 0}, ... % ( 11,  79)
    {   'max',  2, 0, 0}, ... % (  9,  88)
    {  'plus',  2, 0, 0}, ... % (  8,  96)
    { 'times',  2, 0, 0}, ... % (  9, 105)
    {   'any',  2, 0, 0}, ... % (  9, 114)
    {   'min',  2, 0, 1}, ... % (  2, 116)
    {   'min',  3, 0, 0}, ... % ( 30, 146)
    {   'max',  3, 0, 0}, ... % ( 17, 163)
    {  'plus',  3, 0, 0}, ... % ( 13, 176)
    { 'times',  3, 0, 0}, ... % ( 24, 200)
    {   'any',  3, 0, 0}, ... % ( 16, 216)
    {   'min',  4, 0, 0}, ... % ( 10, 226)
    {   'max',  4, 0, 0}, ... % (  9, 235)
    {  'plus',  4, 0, 0}, ... % (  8, 243)
    { 'times',  4, 0, 0}, ... % (  8, 251)
    {   'any',  4, 0, 0}, ... % (  9, 260)
    {   'min',  5, 0, 0}, ... % ( 10, 270)
    {   'max',  5, 0, 0}, ... % (  9, 279)
    {  'plus',  5, 0, 0}, ... % (  8, 287)
    { 'times',  5, 0, 0}, ... % (  9, 296)
    {   'any',  5, 0, 0}, ... % (  9, 305)
    {   'min',  6, 0, 0}, ... % ( 12, 317)
    {   'max',  6, 0, 0}, ... % (  9, 326)
    {  'plus',  6, 0, 0}, ... % (  8, 334)
    { 'times',  6, 0, 0}, ... % (  9, 343)
    {   'any',  6, 0, 0}, ... % (  9, 352)
    {   'bor',  6, 0, 0}, ... % ( 14, 366)
    {  'band',  6, 0, 0}, ... % ( 12, 378)
    {  'bxor',  6, 0, 0}, ... % (  3, 381)
    { 'bxnor',  6, 0, 0}, ... % ( 10, 391)
    {   'min',  7, 0, 0}, ... % ( 17, 408)
    {   'max',  7, 0, 0}, ... % ( 13, 421)
    {  'plus',  7, 0, 0}, ... % (  7, 428)
    { 'times',  7, 0, 0}, ... % (  9, 437)
    {   'any',  7, 0, 0}, ... % (  9, 446)
    {   'bor',  7, 0, 0}, ... % (  4, 450)
    {  'band',  7, 0, 0}, ... % (  3, 453)
    {  'bxor',  7, 0, 0}, ... % (  2, 455)
    { 'bxnor',  7, 0, 0}, ... % (  2, 457)
    {   'min',  8, 0, 0}, ... % ( 10, 467)
    {   'max',  8, 0, 0}, ... % (  9, 476)
    {  'plus',  8, 0, 0}, ... % (  8, 484)
    { 'times',  8, 0, 0}, ... % (  9, 493)
    {   'any',  8, 0, 0}, ... % (  9, 502)
    {   'bor',  8, 0, 0}, ... % (  4, 506)
    {  'band',  8, 0, 0}, ... % (  3, 509)
    {  'bxor',  8, 0, 0}, ... % (  2, 511)
    { 'bxnor',  8, 0, 0}, ... % (  2, 513)
    {   'min',  8, 0, 1}, ... % (  2, 515)
    {   'min',  9, 0, 0}, ... % ( 12, 527)
    {   'max',  9, 0, 0}, ... % (  9, 536)
    {  'plus',  9, 0, 0}, ... % (  8, 544)
    { 'times',  9, 0, 0}, ... % (  9, 553)
    {   'any',  9, 0, 0}, ... % (  9, 562)
    {   'bor',  9, 0, 0}, ... % (  5, 567)
    {  'band',  9, 0, 0}, ... % (  3, 570)
    {  'bxor',  9, 0, 0}, ... % (  2, 572)
    { 'bxnor',  9, 0, 0}, ... % (  2, 574)
    {   'min', 10, 0, 0}, ... % ( 12, 586)
    {   'max', 10, 0, 0}, ... % (  9, 595)
    {  'plus', 10, 0, 0}, ... % (  8, 603)
    { 'times', 10, 0, 0}, ... % (  9, 612)
    {   'any', 10, 0, 0}, ... % (  9, 621)
    {   'min', 11, 0, 0}, ... % (  9, 630)
    {   'max', 11, 0, 0}, ... % (  6, 636)
    {  'plus', 11, 0, 0}, ... % (  6, 642)
    { 'times', 11, 0, 0}, ... % (  9, 651)
    {   'any', 11, 0, 0}, ... % (  9, 660)
    {  'plus', 12, 0, 0}, ... % ( 11, 671)
    { 'times', 12, 0, 0}, ... % (  8, 679)
    {   'any', 12, 0, 0}, ... % (  9, 688)
    {  'plus', 13, 0, 0}, ... % ( 10, 698)
    { 'times', 13, 0, 0}, ... % (  7, 705)
    {   'any', 13, 0, 0}, ... % (  9, 714)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

[~, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

m = 8 ;
n = 4 ;
dt = struct ('inp0', 'tran') ;

ntypes = length (types) ;
A_matrices = cell (ntypes,1) ;
B_matrices = cell (ntypes,1) ;
w_matrices = cell (ntypes,1) ;
m_matrices = cell (ntypes,1) ;

rng ('default') ;
for k1 = 1:length(types)
    atype = types {k1} ;
    A_matrices {k1} = GB_spec_random (m, n, 0.3, 100, atype) ;
    B_matrices {k1} = GB_spec_random (n, m, 0.3, 100, atype) ;
    w_matrices {k1} = GB_spec_random (m, 1, 0.3, 100, atype) ;
    m_matrices {k1} = GB_random_mask (m, 1, 0.5, true, false) ;
end

for kk = 1:length(tasks)
    task = tasks {kk} ;
    op = task {1} ;
    k1 = task {2} ;
    A_is_hyper = task {3} ;
    A_is_csc = task {4} ;

    atype = types {k1} ;

    A = A_matrices {k1} ;
    B = B_matrices {k1} ;
    w = w_matrices {k1} ;
    cin = GB_mex_cast (0, atype) ;

    clear S_input
    S_input.matrix = cin ;
    S_input.pattern = true ;
    S_input.class = atype ;

    clear E_input
    E_input.matrix = sparse (0) ;
    E_input.pattern = false ;
    E_input.class = atype ;

    mask = m_matrices {k1} ;

    is_float = test_contains (atype, 'single') || ...
               test_contains (atype, 'double') ;

    A.is_csc = A_is_csc ; A.is_hyper = A_is_hyper ;
    B.is_csc = A_is_csc ; B.is_hyper = A_is_hyper ;

    if (isequal (op, 'any'))
        tol = [ ] ;
    elseif (test_contains (atype, 'single'))
        tol = 1e-5 ;
    elseif (test_contains (atype, 'double'))
        tol = 1e-12 ;
    else
        tol = 0 ;
    end

    try
        GB_spec_operator (op, atype) ;
        identity = GB_spec_identity (op, atype) ;
    catch
        continue
    end

    % no mask
    w1 = GB_spec_reduce_to_vector (w, [], [], op, A, []) ;
    w2 = GB_mex_reduce_to_vector  (w, [], [], op, A, []) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % no mask, with accum
    w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, A, []) ;
    w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, A, []) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % with mask
    w1 = GB_spec_reduce_to_vector (w, mask, [], op, A, []) ;
    w2 = GB_mex_reduce_to_vector  (w, mask, [], op, A, []) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % with mask and accum
    w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, A, []) ;
    w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, A, []) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % no mask, transpose
    w1 = GB_spec_reduce_to_vector (w, [], [], op, B, dt) ;
    w2 = GB_mex_reduce_to_vector  (w, [], [], op, B, dt) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % no mask, with accum, transpose
    w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, B, dt) ;
    w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, B, dt) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % with mask, transpose
    w1 = GB_spec_reduce_to_vector (w, mask, [], op, B, dt) ;
    w2 = GB_mex_reduce_to_vector  (w, mask, [], op, B, dt) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % with mask and accum, transpose
    w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, B, dt) ;
    w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, B, dt) ;
    GB_spec_compare (w1, w2, identity, tol) ;

    % GB_spec_reduce_to_scalar always operates column-wise, but GrB_reduce
    % operates in whatever order it is given: by column if CSC or by row if
    % CSR.  The result can vary slightly because of different round off
    % errors.  A_flip causes GB_spec_reduce_to_scalar to operate in the
    % same order as GrB_reduce.

    A_flip = A ;
    if (~A.is_csc && is_float)
        A_flip.matrix = A_flip.matrix.' ;
        A_flip.pattern = A_flip.pattern' ;
        A_flip.is_csc = true ;
    end

    % Parallel reduction leads to different roundoff.  So even with A_flip,
    % c1 and c2 can only be compared to within round-off error.

    % to scalar
    c2 = GB_mex_reduce_to_scalar  (cin, [ ], op, A) ;
    if (isequal (op, 'any'))
        X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
        assert (any (X == c2)) ;
    else
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % to GrB_Scalar
    S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], op, A) ;
    c2 = S.matrix ;
    if (isequal (op, 'any'))
        X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
        assert (any (X == c2)) ;
    else
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % to GrB_Scalar
    S = GB_mex_reduce_to_GrB_Scalar (E_input, [ ], op, A) ;
    c2 = S.matrix ;
    if (isequal (op, 'any'))
        X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
        assert (any (X == c2)) ;
    else
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % vector to GrB_Scalar
    S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], op, w) ;
    c2 = S.matrix ;
    if (isequal (op, 'any'))
        X = GB_mex_cast (full (w.matrix (w.pattern)), w.class) ;
        assert (any (X == c2)) ;
    else
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, w) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % to scalar, with accum
    c2 = GB_mex_reduce_to_scalar (cin, 'plus', op, A) ;
    if (~isequal (op, 'any'))
        c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, A_flip) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % to GrB_Scalar, with accum
    S = GB_mex_reduce_to_GrB_Scalar (S_input, 'plus', op, A) ;
    c2 = S.matrix ;
    if (~isequal (op, 'any'))
        c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, A_flip) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    % vector to GrB_Scalar, with accum
    S = GB_mex_reduce_to_GrB_Scalar (S_input, 'plus', op, w) ;
    c2 = S.matrix ;
    if (~isequal (op, 'any'))
        c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, w) ;
        if (is_float)
            assert (abs (c1-c2) < tol *  (abs(c1) + 1))
        else
            assert (isequal (c1, c2)) ;
        end
    end

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            oo = sprintf ('''%s''', op) ;
            fprintf ('{%8s, %2d, %d, %d},', ...
                oo, k1, A_is_hyper, A_is_csc) ;
            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end

end

%-------------------------------------------------------------------------------
% final test
%-------------------------------------------------------------------------------

clear A
A.matrix = sparse (4,5) ;
A.pattern = false (4,5) ;
A.class = 'double' ;

clear S_input
S_input.matrix = 1 ;
S_input.pattern = true ;
S_input.class = 'double' ;

% empty matrix to GrB_Scalar
S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], 'plus', A) ;
assert (nnz (S.matrix) == 0) ;

fprintf ('\ntest14: all tests passed\n') ;

