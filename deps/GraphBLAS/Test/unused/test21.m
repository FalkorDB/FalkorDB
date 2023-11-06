function test21(fulltest)
%TEST21 test GxB_subassign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    % do a short test, by default
    fulltest = 0 ;
end

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
accum_ops = binops.all ; 
types = types.real ;

dn = struct ;
dt = struct ( 'inp0', 'tran' ) ;

if (fulltest)
    fprintf ('\ntest21 --------------exhaustive test of GB_mex_subassign\n') ;
    k1test = 0:length(accum_ops) ;
else
    fprintf ('\ntest21 --------------quick test of GB_mex_subassign\n') ;
    k1test = [ 4 0] ; % Was [4 0 2 7] ;
end

% try all accum
for k1 = k1test
    if (k1 == 0)
        accum_op = ''  ;
        ntypes = 1 ;
    else
        accum_op = accum_ops {k1}  ;
        ntypes = length (types) ;
    end
    fprintf ('\naccum: [%s]', accum_op) ;

    if (fulltest)
        k2test = 1:ntypes ;
    else
        k2test = 11 ; % Was [1 2 11] ;
    end

    % try all types
    for k2 = k2test % 1:ntypes
    clear accum
    if (~isempty (accum_op))
        accum_type = types {k2}  ;
        accum.opname = accum_op ;
        accum.optype = accum_type ;
    else
        accum = '' ;
        accum_type = '' ;
    end

    if (GB_spec_is_positional (accum))
        continue ;
    end

    try
        GB_spec_operator (accum) ;
    catch
        continue ;
    end
    fprintf (' %s', accum_type) ;

    rng (k1 * 100 + k2, 'v4') ;

    for Mask_complement = [false true]

    if (Mask_complement)
        dn.mask = 'complement' ;
        dt.mask = 'complement' ;
    else
        dn.mask = 'default' ;
        dt.mask = 'default' ;
    end

    for C_replace = [false true]

    if (C_replace)
        dn.outp = 'replace' ;
        dt.outp = 'replace' ;
    else
        dn.outp = 'default' ;
        dt.outp = 'default' ;
    end

    kk3 = randperm (length (types), 1) ;

    % try all matrix types, to test casting
    for k3 = kk3 % 1:length (types)
    atype = types {k3}  ;

    % try some matrices
    for m = [1 10] % Was [1 5 10 ]
    for n = [1 10] % Was [ 1 5 10 ]
    for sm = [0 1 5] % Was [ 0 1 5 10 ]
    if (sm > m)
        continue
    end
    for sn = [0 1 5] % Was [ 0 1 5 10 ]
    if (sn > n)
        continue
    end
    for scalar = [false true]

    %---------------------------------------


    if (sm == 0)
        I = [ ] ;
        am = m ;
    else
        I = randperm (m,sm) ; % I = I(1:sm);
        am = sm ;
    end
    I0 = uint64 (I-1) ;
    if (sn == 0)
        J = [ ] ;
        an = n ;
    else
        J = randperm (n,sn) ; % J = J(1:sn);
        an = sn ;
    end
    J0 = uint64 (J-1) ;

    fprintf ('.') ;

    for A_is_hyper = 0:1
    for A_is_csc   = 0:1
    for C_is_hyper = 0:1
    for C_is_csc   = 0:1
    for M_is_hyper = 0:1
    for M_is_csc   = 0:1

    if (scalar)
        % test scalar expansion
        % fprintf ('test expansion\n') ;
        A.matrix = sparse (rand (1)) * 100 ;
        A.pattern = sparse (logical (true));
        A.class = atype ;
        if (A_is_hyper || ~A_is_csc)
            continue
        end
    else
        A = GB_spec_random (am,an,0.2,100,atype, A_is_csc, A_is_hyper) ;
    end

    C = GB_spec_random (m,n,0.2,100,atype, C_is_csc, C_is_hyper) ;
    Mask = GB_random_mask (am,an,0.2, M_is_csc, M_is_hyper) ;

    % C(I,J) = accum (C (I,J),A)
    % Mask = [ ] ;

    C0 = GB_spec_subassign (C, [ ], accum, A, I, J, dn, scalar);
    C1 = GB_mex_subassign  (C, [ ], accum, A, I0, J0, dn);
    GB_spec_compare (C0, C1) ;

    % C(I,J)<Mask> = accum (C (I,J),A)
    C0 = GB_spec_subassign (C, Mask, accum, A, I, J, dn, scalar);
    C1 = GB_mex_subassign  (C, Mask, accum, A, I0, J0, dn);
    GB_spec_compare (C0, C1) ;

    %---------------------------------------

    % C(I,J)<Mask> = accum (C(I,J), A');
    % note transposing twice
    clear AT
    AT.matrix  = A.matrix' ;
    AT.pattern = A.pattern' ;
    AT.class = A.class ;

    C0 = GB_spec_subassign (C, [ ], accum, AT, I, J, dt, scalar);
    C1 = GB_mex_subassign  (C, [ ], accum, AT, I0, J0, dt);
    GB_spec_compare (C0, C1) ;

    C0 = GB_spec_subassign (C, Mask, accum, AT, I, J, dt, scalar);
    C1 = GB_mex_subassign  (C, Mask, accum, AT, I0, J0, dt);
    GB_spec_compare (C0, C1) ;

    %---------------------------------------
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end

fprintf ('\ntest21: all tests passed\n') ;

