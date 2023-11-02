function test77 (fulltest)
%TEST77 test GrB_kronecker

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% TODO: this test takes too long; cut it down

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
types = types.all ;

if (nargin < 1)
    fulltest = 0 ;
end

if (fulltest)
    fprintf ('--------------lengthy tests of GrB_kronecker\n') ;
    k1test = 1:length(types) ;
else
    fprintf ('--------------quick tests of GrB_kronecker\n') ;
    k1test = [4 5 10 11] ;
end

rng ('default') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

n_semirings = 0 ;
for k1 = k1test
    type = types {k1}  ;

    fprintf ('\n%s:\n', type) ;

    if (fulltest)
        k2test = 1:length(binops) ;
    else
        k2test = [45:52 45 4 7] ;
    end

    for k2 = k2test

    binop = binops {k2}  ;
    op.opname = binop ;
    op.optype = type ;
    if (GB_spec_is_positional (op.opname))
        if (~(isequal (type, 'int32') || isequal (type, 'int64')))
            continue
        end
    end

    try
        GB_spec_operator (op) ;
    catch
        continue
    end

    fprintf ('\n    binary op: [ %s %s ] ', binop, type) ;

    for k4 = [0 randi([0,length(binops)], 1, 3)] % 0:length(binops)

    clear accum
    if (k4 == 0)
        accum = ''  ;
        ntypes = 1 ;
        fprintf ('\n        accum: [ none ]') ;
    else
        if (GB_spec_is_positional (op.opname))
            continue ;
        end
        accum.opname = binops {k4}  ;
        if (GB_spec_is_positional (accum.opname))
            continue ;
        end
        ntypes = length (types) ;
        fprintf ('\n        accum: %s ', accum.opname) ;
    end

    for k5 = randi ([1 ntypes], 1, 3) % ntypes

    if (k4 > 0)
        accum.optype = types {k5}  ;
    end

    if (GB_spec_is_positional (accum))
        continue ;
    end

    try
        GB_spec_operator (accum) ;
    catch
        continue
    end

    if (~isempty (accum))
        fprintf ('%s ', accum.optype) ;
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

    % try some matrices
    for am = 5 %  % Was [1 5 10 ]
    for an = 3 % [1 10 ] % Was [ 1 5 10 ]
    for bm = 4 %  % Was [1 4 9 ]
    for bn = 2 %  % Was [1 4 9 ]
    fprintf ('.') ;

    Ax= sparse (100 * sprandn (am,an, 0.5));
    Bx= sparse (100 * sprandn (bm,bn, 0.5));
    cm = am * bm ;
    cn = an * bn ;
    Cx= sparse (100 * sprandn (cm,cn, 0.2));
    Mask = sprandn (cm,cn,0.2) ~= 0 ;
    AT = Ax' ;
    BT = Bx' ;

    for A_is_hyper = 0:1
    for A_is_csc   = 0:1
    for B_is_hyper = 0:1
    for B_is_csc   = 0:1
    for C_is_hyper = 0:1
    for C_is_csc   = 0:1

    clear A
    A.matrix = Ax ;
    A.is_hyper = A_is_hyper ;
    A.is_csc   = A_is_csc   ;

    clear B
    B.matrix = Bx ;
    B.is_hyper = B_is_hyper ;
    B.is_csc   = B_is_csc   ;

    clear C
    C.matrix = Cx ;
    C.is_hyper = C_is_hyper ;
    C.is_csc   = C_is_csc   ;

    %---------------------------------------
    % kron(A,B)
    %---------------------------------------

    % C = kron(A,B)
    C0 = GB_spec_kron (C, [ ], accum, op, A, B, dnn);
    C1 = GB_mex_kron  (C, [ ], accum, op, A, B, dnn);
    GB_spec_compare (C0, C1) ;

    % C = kron(A,B) with Mask
    C0 = GB_spec_kron (C, Mask, accum, op, A, B, dnn);
    C1 = GB_mex_kron  (C, Mask, accum, op, A, B, dnn);
    GB_spec_compare (C0, C1) ;

    %---------------------------------------
    % kron(A',B)
    %---------------------------------------

    % C = kron(A',B), no Mask
    C0 = GB_spec_kron (C, [ ], accum, op, AT, B, dtn);
    C1 = GB_mex_kron  (C, [ ], accum, op, AT, B, dtn);
    GB_spec_compare (C0, C1) ;

    % C = kron(A',B), with Mask
    C0 = GB_spec_kron (C, Mask, accum, op, AT, B, dtn);
    C1 = GB_mex_kron  (C, Mask, accum, op, AT, B, dtn);
    GB_spec_compare (C0, C1) ;

    %---------------------------------------
    % kron(A,B')
    %---------------------------------------

    % no mask
    C0 = GB_spec_kron (C, [ ], accum, op, A, BT, dnt);
    C1 = GB_mex_kron  (C, [ ], accum, op, A, BT, dnt);
    GB_spec_compare (C0, C1) ;

    % with mask
    C0 = GB_spec_kron (C, Mask, accum, op, A, BT, dnt);
    C1 = GB_mex_kron  (C, Mask, accum, op, A, BT, dnt);
    GB_spec_compare (C0, C1) ;

    %---------------------------------------
    % kron(A',B')
    %---------------------------------------

    % no Mask
    C0 = GB_spec_kron (C, [ ], accum, op, AT, BT, dtt);
    C1 = GB_mex_kron  (C, [ ], accum, op, AT, BT, dtt);
    GB_spec_compare (C0, C1) ;

    % with mask
    C0 = GB_spec_kron (C, Mask, accum, op, AT, BT, dtt);
    C1 = GB_mex_kron (C, Mask, accum, op, AT, BT, dtt);
    GB_spec_compare (C0, C1) ;

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

fprintf ('\ntest77: all tests passed\n') ;

