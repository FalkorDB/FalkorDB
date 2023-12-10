function test155
%TEST155 test GrB_*_setElement and GrB_*_removeElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
mlist  = [ 1  1  10  20    5  ] ;
nlist  = [ 1 10   1  10    5  ]  ;
nzlist = [ 5 100 100 1000  100] ;

for trial = 1:4

    % fprintf ('trial: %d\n', trial) ;
    m = mlist (trial) ;
    n = nlist (trial) ;
    nz = nzlist (trial) ;
    I = irand (1, m, nz, 1) ;
    J = irand (1, n, nz, 1) ;
    X = rand (nz, 1) ;
    Action = rand (nz, 1) ; % > 0.4) ;

    %---------------------------------------------------------------------------
    % starting with an empty matrix:
    %---------------------------------------------------------------------------

    % do the work with built-in methods
    C1 = sparse (m, n) ;
    for k = 1:nz
        if (Action (k) <= 0.4)
            C1 (I (k), J (k)) = sparse (0) ;
        else
            C1 (I (k), J (k)) = sparse (X (k)) ;
        end
    end

    % do the work in GraphBLAS (default input matrix)
    C2 = GB_mex_edit (sparse (m, n), I, J, X, Action) ;
    assert (isequal (C1, C2)) ;

    % do the work in GraphBLAS (all hyper / csc/csr cases)
    clear C0
    for is_hyper = 0:1
        for is_csc = 0:1
            C0.matrix = sparse (m, n) ;
            C0.is_hyper = is_hyper ;
            C0.is_csc = is_csc ;
            C2 = GB_mex_edit (C0, I, J, X, Action) ;
            assert (isequal (C1, C2)) ;
        end
    end

    %---------------------------------------------------------------------------
    % starting with a full matrix:
    %---------------------------------------------------------------------------

    % do the work with built-in methods
    C1 = rand (m, n) ;
    C1_start = C1 ;
    for k = 1:nz
        if (Action (k) <= 0.4)
            C1 (I (k), J (k)) = sparse (0) ;
        else
            C1 (I (k), J (k)) = sparse (X (k)) ;
        end
    end

    % do the work in GraphBLAS, testing all sparsity control options
    C0.matrix = C1_start ;
    for sparsity_control = 1:15
        C0.sparsity = sparsity_control ;
        C2 = GB_mex_edit (C0, I, J, X, Action) ;
        assert (isequal (C1, C2)) ;
    end

end

fprintf ('test155: all tests passed\n') ;

