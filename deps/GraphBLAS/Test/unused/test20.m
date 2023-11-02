function test20(fulltest)
%TEST20 test GrB_mxm, mxv, and vxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

tic

if (nargin < 1)
    fulltest = 0 ;
end

if (fulltest == 2)
    fprintf ('test20: lengthy tests of GrB_mxm, mxv, and vxm\n') ;
    n_semirings_max = inf ;
else
    fprintf ('test20: quick test of GrB_mxm, mxv, and vxm\n') ;
    n_semirings_max = 1 ;
end

% types to test:
kk = 1 ;
% ops to test
aa = 1 ;

if (fulltest > 0)
    k1_list = 1:length(mult_ops) ;
    k2_list = 1:length(add_ops) ;
    k3_list = 1:length(types) ;
else
    k1_list = [ 9 ] ;   % times
    k2_list = [ 3 ] ;   % plus
    k3_list = [ 11 ] ;  % double
end

kk = min (kk, length (types)) ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

n_semirings = 0 ;
ntrials = 0 ;

for k1 = k1_list % 1:length(mult_ops)
    mulop = mult_ops {k1} ;
    % if (~GB_spec_is_positional (mulop))
    %     continue ;
    % end

    for k2 = k2_list % 1:length(add_ops)

    addop = add_ops {k2} ;
    fprintf ('\nsemiring %s:%s ', addop, mulop) ;

    for k3 = k3_list % 1:length (types)

    rng ('default') ;
    type = types {k3} ;

    semiring.multiply = mulop ;
    semiring.add = addop ;
    semiring.class = type ;
    if (n_semirings_max == 1)
        semiring
    end

    % create the semiring.  some are not valid because the or,and,xor
    % monoids can only be used when z is boolean for z=mult(x,y).
    try
        [mult_op add_op id] = GB_spec_semiring (semiring) ;
        [mult_opname mult_optype ztype] = GB_spec_operator (mult_op);
        [ add_opname  add_optype] = GB_spec_operator (add_op) ;
        identity = GB_spec_identity (semiring.add, add_optype) ;
    catch
        continue
    end

    if (n_semirings+1 > n_semirings_max)
        fprintf ('\ntest20: all quick tests passed\n') ;
        return ;
    end

    % fprintf ('\n%4d ', n_semirings) ;
    % fprintf ('%12.2f mxm semiring %s:%s:%s ', toc,addop,mulop,type) ;
    % fprintf (' id: %g ', double (identity)) ;
    n_semirings = n_semirings + 1 ;

    % for k4 = [0 5 11 15] % 0:length(mult_ops)
    for k4 = [ 0 randperm(length(mult_ops), aa)]

    if (k4 == 0)
        accum_op = '' ;
        ntypes = 1 ;
    else
        accum_op = mult_ops {k4} ;
        % ntypes = [1 2 8 ] ; % length (types) ;
        % ntypes = 1:length (types) ;
        ntypes = randperm (length (types),kk) ;
    end

    for k5 = ntypes

    clear accum
    if (~isempty (accum_op))
        accum_type = types {k5} ;
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
        continue
    end

    for Mask_complement = [false true]

    if (Mask_complement)
        dnn.mask = 'complement' ;
        dtn.mask = 'complement' ;
        dnt.mask = 'complement' ;
        dtt.mask = 'complement' ;
    else
        dnn.mask = 'default' ;
        dtn.mask = 'default' ;
        dnt.mask = 'default' ;
        dtt.mask = 'default' ;
    end

    for C_replace = [false true]

    if (C_replace)
        dnn.outp = 'replace' ;
        dtn.outp = 'replace' ;
        dnt.outp = 'replace' ;
        dtt.outp = 'replace' ;
    else
        dnn.outp = 'default' ;
        dtn.outp = 'default' ;
        dnt.outp = 'default' ;
        dtt.outp = 'default' ;
    end

    % pick a random class, and int32
    atypes = randperm(length(types),kk) ;
    %  1:length (types)
    atypes = unique ([atypes 6]) ;

    % try all matrix types, to test casting
    for k6 = atypes

    aclas = types {k6} ;

    if (isequal (aclas, 'int32') && ...
        mod (n_semirings, 100) == 1)
        % single or double would lead to
        % different roundoff errors
        hyper_range = 0:1 ;
        csc_range   = 0:1 ;
    else
        hyper_range = 0 ;
        csc_range   = 1 ;
    end

    % try some matrices
    for m = 8 % [1 5 10 ]

    for n = 5 % [ 1 5 10 ]

    for s = 4 % [ 1 5 10 ]
    for density = [0.1 0.2 0.3 0.5]

    % try all combinations of hyper/
    % non-hyper and CSR/CSC
    for A_is_hyper = hyper_range
    for A_is_csc   = csc_range
    for B_is_hyper = hyper_range
    for B_is_csc   = csc_range
    for C_is_hyper = hyper_range
    for C_is_csc   = csc_range
    for M_is_hyper = hyper_range
    for M_is_csc   = csc_range

    if (mod (ntrials, 23) == 0)
        fprintf ('.') ;
    end

    %-----------------------------------
    % A*B
    %-----------------------------------

    A = GB_spec_random (m,s,density,100,aclas, A_is_csc, A_is_hyper) ;
    B = GB_spec_random (s,n,density,100,aclas, B_is_csc, B_is_hyper) ;
    C = GB_spec_random (m,n,density,100,aclas, C_is_csc, C_is_hyper) ;

    % C = A*B, no mask
    Mask = [ ] ;
    C0 = GB_spec_mxm (C, [ ], accum, semiring, A, B, dnn);
    C1 = GB_mex_mxm  (C, [ ], accum, semiring, A, B, dnn);
    GB_spec_compare (C0, C1, identity) ;

    % w = A*u, no mask
    w = GB_spec_random (m,1,density,100,type) ;
    u = GB_spec_random (s,1,density,100,type) ;

    w0 = GB_spec_mxv (w, [ ], accum, semiring, A, u, dnn);
    w1 = GB_mex_mxv  (w, [ ], accum, semiring, A, u, dnn);
    GB_spec_compare (w0, w1, identity) ;

    % w' = u'*A' no mask
    w0 = GB_spec_vxm (w, [ ], accum, semiring, u, A, dnt);
    w1 = GB_mex_vxm  (w, [ ], accum, semiring, u, A, dnt);
    GB_spec_compare (w0, w1, identity) ;

    % C = A*B with mask
    % Mask = sprandn (m,n,0.2) ~= 0 ;
    Mask = GB_random_mask(m,n,0.2, M_is_csc, M_is_hyper) ;
    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dnn);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dnn);
    GB_spec_compare (C0, C1, identity) ;

    % C = A*B with mask (with explicit zero entries)
    Mask1 = sprandn (m,n,0.2) ~= 0 ;
    Mask2 = Mask1 .* spones (sprandn (m,n,0.5)) ;
    S = sparse (m,n) ;
    Mask3 = GB_mex_Matrix_eWiseAdd (S,[ ],[ ],'minus',Mask1,Mask2) ;
    clear Mask
    Mask.matrix = Mask3.matrix ;
    Mask.is_csc = M_is_csc ;
    Mask.is_hyper = M_is_hyper ;
    clear Mask1 Mask2 Mask3
    % the Mask matrix will not pass GB_spok(Mask) test since
    % it will have explicit zeros

    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dnn);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dnn);
    GB_spec_compare (C0, C1, identity) ;

    % w = A*u with mask
    % mask = sprandn (m,1,0.2) ~= 0 ;
    mask = GB_random_mask (m,1,0.2) ;
    w0 = GB_spec_mxv (w, mask, accum, semiring, A, u, dnn);
    w1 = GB_mex_mxv  (w, mask, accum, semiring, A, u, dnn);
    GB_spec_compare (w0, w1, identity) ;

    % w' = u'*A' with mask
    w0 = GB_spec_vxm (w, mask, accum, semiring, u, A, dnt) ;
    w1 = GB_mex_vxm  (w, mask, accum, semiring, u, A, dnt) ;
    GB_spec_compare (w0, w1, identity) ;

    %-----------------------------------
    % A'*B
    %-----------------------------------

    A = GB_spec_random (s,m,density,100,aclas, A_is_csc, A_is_hyper) ;
    B = GB_spec_random (s,n,density,100,aclas, B_is_csc, B_is_hyper) ;
    C = GB_spec_random (m,n,density,100,aclas, C_is_csc, C_is_hyper) ;

    % C = A'*B, no Mask
    C0 = GB_spec_mxm (C, [ ], accum, semiring, A, B, dtn);
    C1 = GB_mex_mxm  (C, [ ], accum, semiring, A, B, dtn);
    GB_spec_compare (C0, C1, identity) ;

    % w = A'*u, no mask

    w0 = GB_spec_mxv (w, [ ], accum, semiring, A, u, dtn);
    w1 = GB_mex_mxv  (w, [ ], accum, semiring, A, u, dtn);
    GB_spec_compare (w0, w1, identity) ;

    % w' = u'*A, no mask
    w0 = GB_spec_vxm (w, [ ], accum, semiring, u, A, dnn);
    w1 = GB_mex_vxm  (w, [ ], accum, semiring, u, A, dnn);
    GB_spec_compare (w0, w1, identity) ;

    % C = A'*B with mask
    % Mask = sprandn (m,n,0.2) ~= 0 ;
    Mask = GB_random_mask (m,n,0.2, M_is_csc, M_is_hyper) ;
    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dtn);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dtn);
    GB_spec_compare (C0, C1, identity) ;

    % C = A'*B with mask
    Mask1 = sprandn (m,n,0.2) ~= 0 ;
    Mask2 = Mask1 .* spones (sprandn (m,n,0.5)) ;
    S = sparse (m,n) ;
    Mask3 = GB_mex_Matrix_eWiseAdd (S,[ ],[ ],'minus',Mask1,Mask2) ;
    clear Mask
    Mask.matrix = Mask3.matrix ;
    Mask.is_csc = M_is_csc ;
    Mask.is_hyper = M_is_hyper ;
    clear Mask1 Mask2 Mask3
    % the Mask matrix will not pass GB_spok(Mask) test since
    % it will have explicit zeros
    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dtn);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dtn);
    GB_spec_compare (C0, C1, identity) ;

    % w = A'*u, with mask
    % mask = sprandn (m,1,0.2) ~= 0 ;
    mask = GB_random_mask (m,1,0.2) ;
    w0 = GB_spec_mxv (w, mask, accum, semiring, A, u, dtn);
    w1 = GB_mex_mxv  (w, mask, accum, semiring, A, u, dtn);
    GB_spec_compare (w0, w1, identity) ;

    % w' = u'*A, with mask
    w0 = GB_spec_vxm (w, mask, accum, semiring, u, A, dnn);
    w1 = GB_mex_vxm  (w, mask, accum, semiring, u, A, dnn);
    GB_spec_compare (w0, w1, identity) ;

    %-----------------------------------
    % A*B'
    %-----------------------------------

    % no mask

    A = GB_spec_random (m,s,density,100,aclas, A_is_csc, A_is_hyper) ;
    B = GB_spec_random (n,s,density,100,aclas, B_is_csc, B_is_hyper) ;
    C = GB_spec_random (m,n,density,100,aclas, C_is_csc, C_is_hyper) ;

    C0 = GB_spec_mxm (C, [ ], accum, semiring, A, B, dnt);
    C1 = GB_mex_mxm  (C, [ ], accum, semiring, A, B, dnt);
    GB_spec_compare (C0, C1, identity) ;

    % with mask
    % Mask = sprandn (m,n,0.2) ~= 0 ;
    Mask = GB_random_mask (m,n,0.2, M_is_csc, M_is_hyper) ;
    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dnt);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dnt);
    GB_spec_compare (C0, C1, identity) ;

    %-----------------------------------
    % A'*B'
    %-----------------------------------

    % no Mask

    A = GB_spec_random (s,m,density,100,aclas, A_is_csc, A_is_hyper) ;
    B = GB_spec_random (n,s,density,100,aclas, B_is_csc, B_is_hyper) ;
    C = GB_spec_random (m,n,density,100,aclas, C_is_csc, C_is_hyper) ;

    C0 = GB_spec_mxm (C, [ ], accum, semiring, A, B, dtt);
    C1 = GB_mex_mxm  (C, [ ], accum, semiring, A, B, dtt);
    GB_spec_compare (C0, C1, identity) ;


    % A'*B', with mask
    % Mask = sprandn (m,n,0.2) ~= 0 ;
    Mask = GB_random_mask (m,n,0.2, M_is_csc, M_is_hyper) ;

    C0 = GB_spec_mxm (C, Mask, accum, semiring, A, B, dtt);
    C1 = GB_mex_mxm  (C, Mask, accum, semiring, A, B, dtt);
    GB_spec_compare (C0, C1, identity) ;

    ntrials = ntrials + 1 ;


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
end
end
end
end

fprintf ('semirings tested: %d\n', n_semirings) ;
fprintf ('\ntest20: all tests passed\n') ;

