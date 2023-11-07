function test19(fulltest)
%TEST19 test GxB_subassign and GrB_*_setElement with many pending operations

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    fulltest = 0 ;
end
if (fulltest)
    nt = 3000 ;
else
    % check if malloc debugging is enabled
    debug_status = stat ;
    if (debug_status)
        % exhaustive malloc debugging
        nt = 50 ; % was 500, which takes too long
    else
        nt = 500 ;
    end
end

fprintf ('\nGxB_subassign and setElement test, many pending computations\n') ;

for problem = 0:2

    clear Work Work2
    switch (problem)
        case 0
            Corig = sparse (5,5) ;
            d = 1 ;
            ntrials = 100 ;
        case 1
            Corig = sprandn (10,20,0.1) ;
            d = 0.3 ;
            ntrials = nt ;
        case 2
            Corig = sparse (rand (10, 20)) ;
            d = 0.3 ;
            ntrials = nt ;
    end

    rng ('default') ;
    [m n] = size (Corig) ;
    fprintf ('problem %d: C is %d-by-%d, # assign/setElement to do: %d\n', ...
        problem, m, n, ntrials) ;

    if (problem > 0)
        fprintf ('... please wait\n') ;
    end

    for k = 1:ntrials

        c = randi (10,1) ;
        if (c == 10)
            d = struct ('outp', 'replace') ;
        elseif (c == 9)
            d = struct ('outp', 'replace', 'mask', 'complement') ;
        elseif (c == 8)
            d = struct ('mask', 'complement') ;
        else
            d = [ ] ;
        end

%       c = randi (10,1) ;
%       if (c == 10)
%           d = struct ('outp', 'replace') ;
%       else
%           d = [ ] ;
%       end

        c = randi (3,1) ;
        switch (c)
            case 1
                accum = '' ;
            case 2
                accum = 'plus' ;
            case 3
                accum = 'second' ;
        end

        c = randi (10,1) ;
        if (c < 8)
            ni = randi (3,1) ;
            nj = randi (3,1) ;
            J = randperm (n, nj) ;
            I = randperm (m, ni) ;
            A = sprand (ni, nj, 0.3) ;
            scalar = 0 ;
        elseif (c == 8)
            % scalar expansion
            ni = 2 ;
            nj = 2 ;
            J = randperm (n, nj) ;
            I = randperm (m, ni) ;
            A = sparse (rand (1)) ;
            scalar = 1 ;
        elseif (c == 9)
            ni = 1 ;
            nj = 1 ;
            I = randperm (m,1) ;
            J = randperm (n,1) ;
            A = sparse (rand (1)) ;
            scalar = 0 ;
        else
            ni = 2 ;
            nj = 2 ;
            I = randperm (m,2) ;
            J = randperm (n,2) ;
            A = sparse (rand (2)) ;
            scalar = 0 ;
        end

        c = randi (2,1) ;
        switch (c)
            case 1
                Mask = [ ] ;
            case 2
                Mask = (sprand (ni, nj, 0.3) ~= 0) ;
        end

        Work (k).A = A ;
        Work (k).I = I ;
        Work (k).J = J ;
        Work (k).Mask = Mask ;
        Work (k).accum = accum ;
        Work (k).desc = d ;
        Work (k).scalar = scalar ;

        Work2 (k).A = A ;
        Work2 (k).I = uint64 (I-1) ;
        Work2 (k).J = uint64 (J-1) ;
        Work2 (k).Mask = Mask ;
        Work2 (k).accum = accum ;
        Work2 (k).desc = d ;

    end

    C3 = Corig ;
    for k = 1:ntrials
        J = Work (k).J ;
        I = Work (k).I ;
        A = Work (k).A ;
        M = Work (k).Mask ;
        accum = Work (k).accum ;
        d = Work (k).desc ;
        scalar = Work (k).scalar ;
        C3 = GB_spec_subassign (C3, M, accum, A, I, J, d, scalar) ;
    end

    % default sparsity
    C2 = GB_mex_subassign (Corig, Work2) ;  % WORK_ASSIGN
    GB_spec_compare (C2, C3) ;

    % with sparsity control
    for C_sparsity_control = [1 2 4 8]
        for M_sparsity_control = [2 12]
            ctrl = [C_sparsity_control M_sparsity_control] ;
            C2 = GB_mex_subassign (Corig, Work2, ctrl) ; % WORK_ASSIGN
            GB_spec_compare (C2, C3) ;
        end
    end

    % with no accum

    C3 = Corig ;
    for k = 1:ntrials
        J = Work (k).J ;
        I = Work (k).I ;
        A = Work (k).A ;
        M = Work (k).Mask ;
        d = Work (k).desc ;
        scalar = Work (k).scalar ;
        C3 = GB_spec_subassign (C3, M, [ ], A, I, J, d, scalar) ;
        Work2 (k).accum = [ ] ;
    end

    % default sparsity
    C2 = GB_mex_subassign (Corig, Work2) ;      % WORK_ASSIGN
    GB_spec_compare (C2, C3) ;

    % with sparsity control
    for C_sparsity_control = [1 2 4 8]
        for M_sparsity_control = [2 12]
            ctrl = [C_sparsity_control M_sparsity_control] ;
            C2 = GB_mex_subassign (Corig, Work2, ctrl) ;    % WORK_ASSIGN
            GB_spec_compare (C2, C3) ;
        end
    end

end

fprintf ('\ntest19: all tests passed\n') ;

