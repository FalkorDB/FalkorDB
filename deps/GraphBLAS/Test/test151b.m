function test151b
%TEST151B test bitshift operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test151b: test bshift operator\n') ;

[~, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.int ;
ops2 = { 'bshift' } ;

int_nbits = [ 8, 16, 32, 64, 8, 16, 32, 64 ] ;

rng ('default') ;
Cin = sparse (4,4) ;
C10 = sparse (10,10) ;
desc.mask = 'complement' ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

for k = 1:8
    type = types {k} ;
    nbits = int_nbits (k) ;
    for trial = 1:4

        % create the test matrices
        imax = double (intmax (type) / 4) ;
        A = GB_mex_cast (imax * rand (4), type) ;
        B = GB_mex_cast ((nbits-1) * rand (4), type) + 1 ;
        clear A2 ; A2.matrix = sparse (double (A)) ; A2.class = type ;
        clear B2 ; B2.matrix = sparse (double (B)) ; B2.class = 'int8' ;
        A2.pattern = logical (spones (A)) ;
        B2.pattern = logical (spones (B)) ;
        M = sparse (mod (magic (4), 2)) ;
        clear M2 ; M2.matrix = M ; M2.class = 'logical' ;

        % determine the tests to run
        Sparsities = { } ;
        switch (k)
            case 1
                if (trial == 1)
                    Sparsities = {
                    { 1, 1, 1, 1, 1}, ... % (  3,    3)
                    { 1, 1, 1, 1, 4}, ... % (  3,    6)
                    { 1, 1, 1, 4, 1}, ... % (  9,   15)
                    { 1, 1, 1, 4, 4}, ... % (  8,   23)
                    { 1, 1, 1, 8, 1}, ... % (  4,   27)
                    { 1, 1, 4, 1, 1}, ... % (  6,   33)
                    { 1, 1, 4, 1, 4}, ... % (  6,   39)
                    { 1, 1, 4, 4, 1}, ... % ( 10,   49)
                    { 1, 1, 4, 4, 4}, ... % (  1,   50)
                    { 1, 1, 4, 8, 1}, ... % (  4,   54)
                    { 1, 1, 8, 1, 1}, ... % (  2,   56)
                    { 1, 1, 8, 8, 1}, ... % (  3,   59)
                    { 1, 1, 8, 8, 4}, ... % (  2,   61)
                    } ;
                elseif (trial == 4)
                    Sparsities = {
                    { 1, 4, 1, 1, 1}, ... % (  1,   62)
                    { 1, 4, 1, 1, 4}, ... % (  2,   64)
                    { 1, 4, 4, 1, 1}, ... % (  2,   66)
                    { 1, 4, 4, 1, 4}, ... % (  1,   67)
                    { 1, 4, 4, 4, 4}, ... % (  1,   68)
                    { 1, 4, 4, 8, 1}, ... % (  4,   72)
                    } ;
                end
            case 2
                if (trial == 1)
                    Sparsities = {
                    { 2, 1, 1, 1, 1}, ... % (  5,   77)
                    { 2, 1, 1, 4, 1}, ... % (  1,   78)
                    { 2, 1, 4, 1, 1}, ... % (  1,   79)
                    { 2, 1, 4, 4, 1}, ... % (  2,   81)
                    } ;
                end
            case 3
                if (trial == 1)
                    Sparsities = {
                    { 3, 1, 1, 1, 1}, ... % (  3,   84)
                    { 3, 1, 1, 4, 1}, ... % (  1,   85)
                    { 3, 1, 4, 1, 1}, ... % (  1,   86)
                    { 3, 1, 4, 4, 1}, ... % (  2,   88)
                    } ;
                end
            case 4
                if (trial == 1)
                    Sparsities = {
                    { 4, 1, 1, 1, 1}, ... % (  3,   91)
                    { 4, 1, 1, 4, 1}, ... % (  1,   92)
                    { 4, 1, 4, 1, 1}, ... % (  1,   93)
                    { 4, 1, 4, 4, 1}, ... % (  2,   95)
                    } ;
                end
            case 5
                if (trial == 1)
                    Sparsities = {
                    { 5, 1, 1, 1, 1}, ... % (  3,   98)
                    { 5, 1, 1, 4, 1}, ... % (  1,   99)
                    { 5, 1, 4, 1, 1}, ... % (  1,  100)
                    { 5, 1, 4, 4, 1}, ... % (  2,  102)
                    } ;
                end
            case 6
                if (trial == 1)
                    Sparsities = {
                    { 6, 1, 1, 1, 1}, ... % (  3,  105)
                    { 6, 1, 1, 4, 1}, ... % (  1,  106)
                    { 6, 1, 4, 1, 1}, ... % (  1,  107)
                    { 6, 1, 4, 4, 1}, ... % (  2,  109)
                    } ;
                end
            case 7
                if (trial == 1)
                    Sparsities = {
                    { 7, 1, 1, 1, 1}, ... % (  3,  112)
                    { 7, 1, 1, 4, 1}, ... % (  1,  113)
                    { 7, 1, 4, 1, 1}, ... % (  1,  114)
                    { 7, 1, 4, 4, 1}, ... % (  2,  116)
                    } ;
                end
            case 8
                if (trial == 1)
                    Sparsities = {
                    { 8, 1, 1, 1, 1}, ... % (  3,  119)
                    { 8, 1, 1, 4, 1}, ... % (  1,  120)
                    { 8, 1, 4, 1, 1}, ... % (  1,  121)
                    { 8, 1, 4, 4, 1}, ... % (  2,  123)
                    } ;
                end
        end

        % run the tests
        for kk = 1:length(Sparsities)
            Sparsity = Sparsities {kk} ;
            A_sparsity = Sparsity {3} ;
            B_sparsity = Sparsity {4} ;
            M_sparsity = Sparsity {5} ;

            A2.sparsity = A_sparsity ;
            B2.sparsity = B_sparsity ;
            M2.sparsity = M_sparsity ;

            opname = 'bshift' ;
            op.opname = opname ; op.optype = type ;

            % C1 = bitop (A, B) ;
            C1 = GB_spec_Matrix_eWiseMult(Cin, [], [], op, A2, B2, []) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, [], [], op, A2, B2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, [], [], op, A2, B2, []) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, [], [], op, A2, B2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion(Cin, [], [], op, A2, 3, B2, 2, []) ;
            C2 = GB_mex_Matrix_eWiseUnion (Cin, [], [], op, A2, 3, B2, 2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, [], [], op, B2, A2, []) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, [], [], op, B2, A2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion(Cin, [], [], op, B2, 3, A2, 2, []) ;
            C2 = GB_mex_Matrix_eWiseUnion (Cin, [], [], op, B2, 3, A2, 2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseMult(Cin, M2, [], op, A2, B2, []) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, M2, [], op, A2, B2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, M2, [], op, A2, B2, []) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, M2, [], op, A2, B2, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion (Cin, M2, [], op, A2, 1, B2, 3, []) ;
            C2 = GB_mex_Matrix_eWiseUnion  (Cin, M2, [], op, A2, 1, B2, 3, []) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseMult(Cin, M2, [], op, A2, B2, desc) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, M2, [], op, A2, B2, desc) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, M2, [], op, A2, B2, desc) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, M2, [], op, A2, B2, desc) ;
            GB_spec_compare (C1, C2) ;

            if (track_coverage)
                c = sum (GraphBLAS_grbcov > 0) ;
                d = c - clast ;
                if (d > 0)
                    fprintf (...
                    '{%2d,%2d,%2d,%2d,%2d}, ... ', ...
                    k, trial, A_sparsity, B_sparsity, M_sparsity) ;
                    fprintf ('%% (%3d, %4d)\n', d, c - cfirst) ;
                end
                clast = c ;
            else
                fprintf ('.') ;
            end

        end
    end
end

fprintf ('\ntest151b: all tests passed\n') ;

