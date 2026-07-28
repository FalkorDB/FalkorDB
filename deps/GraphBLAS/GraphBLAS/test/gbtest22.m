function gbtest22 (ghb)
%GBTEST22 test reduce to scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

desc.kind = 'sparse' ;

A = magic (3) ;
types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    if (isequal (type, 'logical'))
        c = false ;
        c = gtb_reduce (ghb, c, '|', '|', gbtest_cast (A, 'logical')) ; %#ok<*NASGU>
    else
        % c = ones (1, 1, type) ;
        is_double_complex = isequal (type, 'double complex') ;
        is_single_complex = isequal (type, 'single complex') ;

        if (is_double_complex)
            c = complex (ones (1, 1)) ;
        elseif (is_single_complex)
            c = complex (ones (1, 1, 'single')) ;
        else
            c = ones (1, 1, type) ;
        end

        c = gtb_reduce (ghb, c, '+', '+', gbtest_cast (A, type)) ;
        assert (c == sum (sum (A)) + 1) ;
    end
end


for trial = 1:10
    fprintf ('.') ;
    for m = 0:5
        for n = 0:5
            A = 100 * sprand (m, n, 0.5) ;
            G = gtb (ghb, A) ;
            [i, j, x] = find (A) ; %#ok<*ASGLU>

            % c1 = sum (A, 'all') ;
            c1 = sum (sum (A)) ;
            c2 = gtb_reduce (ghb, '+', A) ;
            c3 = sum (G, 'all') ;
            c4 = gtb_reduce (ghb, '+', A, desc) ;
            c5 = gzb_reduce (ghb, '+', G) ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;
            assert (norm (c1-c3,1) <= 1e-12 * norm (c1,1)) ;
            assert (norm (c1-c4,1) <= 1e-12 * norm (c1,1)) ;
            assert (isequal (class (c4), 'double')) ;
            assert (norm (c1-c5,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = pi + sum (A, 'all') ;
            c1 = pi + sum (sum (A)) ;
            c2 = gtb_reduce (ghb, pi, '+', '+', A) ;
            c3 = pi + sum (G, 'all') ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;
            assert (norm (c1-c3,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = prod (x, 'all') ;
            if (isempty (x))
                c1 = sparse (0) ;
            else
                c1 = prod (x) ;
            end
            c2 = gtb_reduce (ghb, '*', A) ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = prod (A, 'all') ;
            if (nnz (A) == 0)
                c1 = sparse (0) ;
            else
                c1 = prod (prod (A)) ;
            end
            c2 = prod (G, 'all') ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = pi + prod (x, 'all') ;
            if (nnz (A) == 0)
                c1 = pi ;
            else
                c1 = pi + prod (x) ;
            end
            c2 = gtb_reduce (ghb, pi, '+', '*', A) ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = max (A, [ ], 'all') ;
            c1 = max (max (A)) ;
            c2 = gtb_reduce (ghb, 'max', A) ;
            if (nnz (A) < m*n)
                c2 = max (full (c2), 0) ;
            end
            c3 = max (G, [ ], 'all') ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;
            assert (norm (c1-c3,1) <= 1e-12 * norm (c1,1)) ;

            % c1 = min (A, [ ], 'all') ;
            c1 = min (min (A)) ;
            c2 = gtb_reduce (ghb, 'min', A) ;
            if (nnz (A) < m*n)
                c2 = min (full (c2), 0) ;
            end
            c3 = min (G, [ ], 'all') ;
            assert (norm (c1-c2,1) <= 1e-12 * norm (c1,1)) ;
            assert (norm (c1-c3,1) <= 1e-12 * norm (c1,1)) ;

            B = logical (A) ;
            G = gtb (ghb, B) ;

            % c1 = any (A, 'all') ;
            c1 = any (any (A)) ;
            c2 = gtb_reduce (ghb, '|.logical', A) ;
            c3 = any (G, 'all') ;
            assert (c1 == logical (c2)) ;
            assert (c1 == logical (c3)) ;

            % c1 = all (A, 'all') ;
            if (isempty (A))
                c1 = sparse (false) ;
            else
                c1 = all (all (A)) ;
            end
            c3 = all (G, 'all') ;
            assert (c1 == logical (c3)) ;

            [i, j, x] = find (A) ;
            % c1 = all (x, 'all') ;
            if (isempty (x))
                c1 = 0 ;
            else
                c1 = all (x) ;
            end
            c2 = gtb_reduce (ghb, '&.logical', A) ;
            assert (c1 == logical (c2)) ;

        end
    end
end

fprintf ('\ngbtest22 (%d): all tests passed\n', ghb) ;

