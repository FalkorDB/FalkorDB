function test242
%TEST242 test GxB_Iterator for matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;
ntypes = length (types) ;

m = 20 ;
n = 30 ;
GB_builtin_complex_set (true) ;

% yin is dense and all zero
yin.matrix = sparse (zeros (m, 1)) ;
yin.pattern = true (m, 1) ;

for k = 1:(ntypes + 1)

    if (k == ntypes + 1)
        % user-defined double complex
        GB_builtin_complex_set (false) ;
        type = 'double complex'  ;
        fprintf ('\nComplex (UDT):\n') ;
    else
        % built-in types
        type = types {k}  ;
        fprintf ('\n%s:\n', type) ;
    end
    
    if (test_contains (type, 'single'))
        tol = 1e-5 ;
    elseif (test_contains (type, 'double'))
        tol = 1e-14 ;
    else
        tol = 0 ;
    end

    A = [ ] ;

    if (isequal (type, 'bool'))
        accum.opname = 'lor' ;
        semiring.add = 'lor' ;
        semiring.multiply = 'land' ;
    else
        accum.opname = 'plus' ;
        semiring.add = 'plus' ;
        semiring.multiply = 'times' ;
    end
    accum.optype = type ;
    semiring.class = type ;

    for sparsity = [1 2 4 8]

        if (sparsity == 8)
            fprintf ('full') ;
            ntrials = 1 ;
        elseif (sparsity == 4)
            fprintf ('bitmap') ;
            ntrials = 51 ;
        elseif (sparsity == 2)
            fprintf ('sparse') ;
            ntrials = 51 ;
        elseif (sparsity == 1)
            fprintf ('hyper') ;
            ntrials = 51 ;
        end

        for trial = 1:ntrials

            fprintf ('.') ;
            if (sparsity == 8 || trial == ntrials)
                A = GB_spec_random (m, n, inf, 100, type) ;
            else
                d = trial / 200 ;
                A = GB_spec_random (m, n, d, 100, type) ;
            end
            A.sparsity = sparsity ;

            % x must be full
            x = GB_spec_random (n, 1, inf, 2, type) ;
            x.sparsity = 8 ;

            % yin is empty
            yin.class = type ;
            yin.iso = false ;

            y0 = GB_spec_mxv (yin, [ ], accum, semiring, A, x, [ ]) ;

            % CSR with row iterator
            A.is_csc = false ;
            y1 = GB_mex_mxv_iterator (A, x, 0) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 0+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSR with entry iterator
            y1 = GB_mex_mxv_iterator (A, x, 2) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 2+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSR with row iterator: backwards kseek
            y1 = GB_mex_mxv_iterator (A, x, 3) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 3+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSR with row iterator: with seekRow
            y1 = GB_mex_mxv_iterator (A, x, 5) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 5+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSR with entry iterator, seek
            y1 = GB_mex_mxv_iterator (A, x, 7) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 7+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSC with col iterator
            A.is_csc = true ;
            y1 = GB_mex_mxv_iterator (A, x, 1) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 1+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSC with entry iterator
            y1 = GB_mex_mxv_iterator (A, x, 2) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 2+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSC with col iterator: backwards kseek
            y1 = GB_mex_mxv_iterator (A, x, 4) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 4+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSC with col iterator: with seekCol
            y1 = GB_mex_mxv_iterator (A, x, 6) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 6+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

            % CSC with entry iterator, seek
            y1 = GB_mex_mxv_iterator (A, x, 7) ;    % with macros
            GB_spec_compare (y0, y1, 0, tol) ;
            y1 = GB_mex_mxv_iterator (A, x, 7+8) ;  % with functions
            GB_spec_compare (y0, y1, 0, tol) ;

        end
        fprintf ('\n') ;
    end
end
GB_builtin_complex_set (true) ;
fprintf ('\ntest242: all tests passed\n') ;

