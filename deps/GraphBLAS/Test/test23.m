function test23(fulltest)
%TEST23 test GrB_*_build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

if (nargin < 1)
    % do a short test, by default
    fulltest = 0 ;
end

ops = {
'first',  0, % z = x
'second', 0, % z = y
'pair',   1, % z = 1
'oneb',   1, % z = 1 (same as pair)
'any',    1, % z = pick x or y
'min',    1, % z = min(x,y)
'max',    1, % z = max(x,y)
'plus',   1, % z = x + y
'times',  1, % z = x * y
'iseq',   1, % z = x == y
'or',     1, % z = x || y
'and',    1, % z = x && y
'xor'     1, % z = x != y
} ;


if (fulltest)
    fprintf ('\n==== exhaustive test for GrB_Matrix_build and GrB_Vector_build:\n');
    problems = [
        10,    8,   40,  -5, 100
        100, 200, 1000, -99, 200
         50,  50,  500,  -2, 3
        ] ;
else
    fprintf ('\n==== quick test for GrB_Matrix_build and GrB_Vector_build:\n');
    problems = [
        10,    8,   40,  -5, 100
        ] ;
end

% try several problems
for k0 = 1:size (problems,1) ;

    % create nnz triplets for a matrix of size nrows-by-nrows
    nrows = problems (k0,1) ;
    ncols = problems (k0,2) ;
    nnz = problems (k0,3) ;
    y1 = problems (k0,4) ;
    y2 = problems (k0,5) ;

    rng ('default') ;
    I = irand (0, nrows-1, nnz, 1) ;
    J = irand (0, ncols-1, nnz, 1) ;
    Y = y2 * rand (nnz, 1) + y1 ;

    fprintf ('\nnrows: %d ncols %d nnz %d ymin %g ymax %g\n', ...
        nrows, ncols, nnz, min (Y), max (Y)) ;

    % try every operator
    for k1 = 1:size (ops,1)
        op.opname = ops {k1,1} ;
        is_associative = ops {k1,2} ;
        op_is_any = isequal (op.opname, 'any') ;

        fprintf ('\n%-14s ', op.opname) ;

        % try every operator type
        for k2 = 1:length (types)
            op.optype = types {k2} ;
            z = GB_mex_cast (1, op.optype) ;
            opint = isinteger (z) || islogical (z) ;

            try
                GB_spec_operator (op) ;
            catch
                continue
            end

            % the non-boolean logical operators are not associative
            if (isequal (op.opname, 'or')  || ...
                isequal (op.opname, 'and')  || ...
                isequal (op.opname, 'iseq')  || ...
                isequal (op.opname, 'xor'))
                if (~isequal (op.optype, 'logical'))
                    is_associative = false ;
                end
            end

            if (test_contains (op.optype, 'single'))
                epsilon = 1e-5 ;
            elseif (test_contains (op.optype, 'double'))
                epsilon = 1e-12 ;
            else
                epsilon = 0 ;
            end

            if (fulltest)
                k3list = 1:length(types) ;
            else
                k3list = unique ([k2 randperm(11,2)]) ;
            end

            % try every type for X
            for k3 = k3list % 1:length (types)
                xtype = types {k3} ;
                X = GB_mex_cast (Y, xtype) ;
                fprintf ('.') ;

                if (fulltest)
                    k4list = 1:length(types) ;
                else
                    k4list = unique ([k3 randperm(11,2)]) ;
                end

                % try every type for the result
                for k4 = k4list % 1:length (types)
                    ctype = types {k4} ;

                    % build the matrix in the natural order
                    % fprintf ('\n-------------------------------op: %s ', ...
                    % op.opname) ;
                    % fprintf ('optype: %s ', op.optype) ;
                    % fprintf ('xtype: %s ', xtype) ;

                    for A_is_csc   = 0:1

                        A = GB_mex_Matrix_build (I, J, X, nrows, ncols, op, ...
                            ctype, A_is_csc) ;
                        % A is sparse but may have explicit zeros
                        if (~GB_spok (A.matrix*1))
                            fprintf ('test failure: invalid sparse matrix\n') ;
                            assert (false) ;
                        end
                        A.matrix = full (double (A.matrix)) ;
                        if (~op_is_any)
                            S = GB_spec_build (I, J, X, nrows, ncols, op, 'natural', ctype) ;
                            if (~isequalwithequalnans (A.matrix, double (S.matrix))) ;
                                fprintf ('test failure: does not match spec\n') ;
                                assert (false) ;
                            end
                            assert (isequal (S.class, A.class)) ;
                        end

                        % build in random order, for associative operators.
                        if (is_associative)
                            [S2 p] = GB_spec_build (I, J, X, nrows, ncols, ...
                                op, 'random', ctype) ;
                            if (op_is_any)
                                % 'any' reduction
                            elseif (opint)
                                % integers are perfectly associative
                                if (~isequal (A.matrix, double (S2.matrix)))
                                    fprintf ('fail: int non-associative\n') ;
                                    assert (false) ;
                                end
                            else
                                % floating point is approximately associative
                                tol = norm (double (S2.matrix)) * epsilon ;
                                ok = isequal (isnan (A.matrix), isnan (S2.matrix)) ;
                                A.matrix (isnan (A.matrix)) = 0 ;
                                S2.matrix (isnan (S2.matrix)) = 0 ;
                                ok = ok & (norm (double (A.matrix - double (S2.matrix))) < tol) ;
                                if (~ok)
                                    fprintf ('fail: float non-associative\n') ;
                                    assert (false) ;
                                end
                            end
                        end
                    end

                    % build a vector in the natural order (discard J)
                    % fprintf ('\n-------------------------------op: %s ', op) ;
                    % fprintf ('optype: %s ', optype) ;
                    % fprintf ('xtype: %s\n', xtype) ;
                    A = GB_mex_Vector_build (I, X, nrows, op, ctype) ;
                    % A is sparse but may have explicit zeros
                    if (~GB_spok (A.matrix*1))
                        fprintf ('test failure: invalid sparse matrix\n') ;
                        assert (false) ;
                    end
                    if (~op_is_any)
                        A.matrix = full (double (A.matrix)) ;
                        S = GB_spec_build (I, [ ], X, nrows, 1, op, 'natural', ctype) ;
                        if (~isequalwithequalnans (A.matrix, double (S.matrix))) ;
                            fprintf ('test failure: does not match spec\n') ;
                            assert (false) ;
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest23: all tests passed\n') ;

