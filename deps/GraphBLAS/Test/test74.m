function test74
%TEST74 test GrB_mxm: all built-in semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

fprintf ('test74 -------- GrB_mxm on all semirings (all methods)\n') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

dnn_Gus  = struct ( 'axb', 'gustavson' ) ;
dnn_hash = struct ( 'axb', 'hash' ) ;

ntrials = 0 ;

GB_builtin_complex_set (true) ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = (~isempty (GraphBLAS_grbcov)) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
end

rng ('default') ;

m_list = [ 1  2    9 ] ;
n_list = [ 1  2   10 ] ;
k_list = [ 20 100 12 ] ;
d_list = [0.3 0.3 0.3] ;
n_semirings = 0 ;

for k0 = 1:size(m_list,2)
    jit_reset

    m = m_list (k0) ;
    n = n_list (k0) ;
    k = k_list (k0) ;
    density = d_list (k0) ;

    A = GB_spec_random (m,k,density,100,'none') ;
    B = GB_spec_random (k,n,density,100,'none') ;
    C = GB_spec_random (m,n,density,100,'none') ;
    M = spones (sprandn (m, n, 0.3)) ;
    F = GB_spec_random (m,n,inf,100,'none') ;
    F.sparsity = 8 ;

    A_bitmap = A ; A_bitmap.sparsity = 4 ;
    B_bitmap = B ; B_bitmap.sparsity = 4 ;

    Bfull = GB_spec_random (k,n,inf,2,'none') ;
    Bfull.sparsity = 8 ;

    Afull = GB_spec_random (m,k,inf,2,'none') ;
    Afull.sparsity = 8 ;

    clear AT
    AT = A ;
    AT.matrix  = A.matrix.' ;
    AT.pattern = A.pattern' ;
    fprintf ('\n\nm %d n %d k %d: \n', m, n, k) ;

    for k2 = 1:length(add_ops)
        addop = add_ops {k2} ;

        if (k0 == 1 || k0 == 3)
            % only use the min monoid for the 1st and 3rd matrices
            if (~isequal (addop, 'min'))
                continue
            end
        end

        for k4 = 1:length (types)
            monoid.opname = addop ;
            monoid.optype = types {k4} ;

            try
                identity = GB_spec_identity (monoid) ;
            catch
                continue ;
            end
            % jit_reset

            % skip these monoids; not needed for test coverage
            if (isequal (monoid.optype, 'logical'))
                if (isequal (addop, 'plus') || ... 
                    isequal (addop, 'times') || ... 
                    isequal (addop, 'or') || ... 
                    isequal (addop, 'and'))
                    continue
                end
            end

            fprintf ('\n%d %-14s: ', k0, [addop '.' monoid.optype]) ;
            if (track_coverage)
                fprintf ('\n') ;
            end

            for k1 = 1:length(mult_ops)
                mulop = mult_ops {k1} ;

                for k3 = 1:length (types)
                    semiring_type = types {k3} ;
                    semiring.multiply = mulop ;
                    semiring.add = monoid ;
                    semiring.class = semiring_type ;

                    %-----------------------------------------------------------
                    % skip these semirings; not needed for test coverage:
                    %-----------------------------------------------------------

                    if (k0 == 3)
                        if (isequal (addop, 'min') && ...
                            isequal (mulop, 'first') && ...
                            isequal (semiring_type, 'int32'))
                            % only do min.first.int32 with the 3rd matrix
                        else
                            % fprintf ('skip\n') ;
                            continue
                        end
                    end

                    if (k0 == 1)
                        if (isequal (addop, 'min') && ...
                            isequal (mulop, 'first'))
                            % only do min.first.* with the 1st matrix
                        else
                            % fprintf ('skip\n') ;
                            continue
                        end
                    end

                    if (isequal (mulop, 'oneb'))
                        continue
                    end

                    if ((isequal (addop, 'any') || isequal (addop, 'times')) ...
                        && ...
                        (isequal (mulop, 'and') || isequal (mulop, 'or') || ...
                         isequal (mulop, 'pair')|| isequal (mulop, 'xor')))
                        continue
                    end

                    if ((isequal (addop, 'min') || isequal (addop, 'max')) ...
                       && ...
                        (isequal (mulop, 'and') || isequal (mulop, 'xor') || ...
                         isequal (mulop, 'or')))
                        continue
                    end

                    if (isequal (mulop (1:2), 'is'))
                        if (isequal (semiring_type, 'logical'))
                            % *.is*.logical: do this but no other type
                        else
                            continue
                        end
                    end

                    %-----------------------------------------------------------
                    % create the semiring.  some are not valid because the
                    % or,and,xor monoids can only be used when z is boolean for
                    % z=mult(x,y).
                    try
                        [mult_op add_op id] = GB_spec_semiring (semiring) ;
                        [mult_opname mult_optype ztype xtype ytype] = ...
                            GB_spec_operator (mult_op);
                        [ add_opname  add_optype] = GB_spec_operator (add_op) ;
                        identity = GB_spec_identity (semiring.add) ;
                    catch me
                        continue
                    end

                    if (track_coverage)
                        fprintf ('[%s.%s.%s] ', addop, mulop, semiring_type) ;
                    end
                    n_semirings = n_semirings + 1 ;

                    AT.class = semiring_type ;
                    A.class = semiring_type ;
                    B.class = semiring_type ;
                    C.class = monoid.optype ;
                    A_bitmap.class = semiring_type ;
                    B_bitmap.class = semiring_type ;
                    F.class = monoid.optype ;
                    Afull.class = semiring_type ;
                    Bfull.class = semiring_type ;

                    % C<M> = A'*B, with Mask, no typecasting
                    C1 = GB_mex_mxm  (C, M, [ ], semiring, AT, B, dtn) ;
                    C0 = GB_spec_mxm (C, M, [ ], semiring, AT, B, dtn) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A'*B, no Mask, no typecasting
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, AT, B, dtn) ;
                    C0 = GB_spec_mxm (C, [ ], [ ], semiring, AT, B, dtn) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A*B, no Mask, no typecasting, Gustavson
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, dnn_Gus) ;
                    C0 = GB_spec_mxm (C, [ ], [ ], semiring, A, B, dnn_Gus) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A*B, no Mask, no typecasting, Hash
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, dnn_hash) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A_bitmap * B
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A_bitmap, B, [ ]) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A * B_bitmap
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B_bitmap, [ ]) ;
                    GB_spec_compare (C0, C1, identity) ;

                    % C = A_bitmap * B_bitmap
                    C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A_bitmap, B_bitmap,[]);
                    GB_spec_compare (C0, C1, identity) ;

                    % F += A * B_bitmap
                    C1 = GB_mex_mxm  (F, [ ], monoid, semiring, A, B_bitmap, [ ]) ;
                    C0 = GB_spec_mxm (F, [ ], monoid, semiring, A, B, [ ]) ;
                    GB_spec_compare (C0, C1, identity) ;

                    if (isequal (monoid.opname, 'times'))

                        F2 = F ;
                        F2.matrix = F2.matrix / max (F2.matrix, [ ], 'all') ;
                        A2 = A ;
                        A2.matrix = A2.matrix / max (A2.matrix, [ ], 'all') ;
                        B2 = B ;
                        B2.matrix = B2.matrix / max (B2.matrix, [ ], 'all') ;
                        A3 = Afull ;
                        A3.matrix = A3.matrix / max (A3.matrix, [ ], 'all') ;
                        B3 = Bfull ;
                        B3.matrix = B3.matrix / max (B3.matrix, [ ], 'all') ;
                        A4 = AT ;
                        A4.matrix = A4.matrix / max (A4.matrix, [ ], 'all') ;

                        % F2 += A * B
                        C1 = GB_mex_mxm_update  (F2, semiring, A2, B2, [ ]) ;
                        C0 = GB_spec_mxm (F2, [ ], monoid, semiring, A2, B2, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;
                        % F2 += A3 * B
                        C1 = GB_mex_mxm_update  (F2, semiring, A3, B2, [ ]) ;
                        C0 = GB_spec_mxm (F2, [ ], monoid, semiring, A3, B2, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;
                        % F2 += A2 * B3
                        C1 = GB_mex_mxm_update  (F2, semiring, A2, B3, [ ]) ;
                        C0 = GB_spec_mxm (F2, [ ], monoid, semiring, A2, B3, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;

                        % F2 += A4'*B3, no Mask, no typecasting
                        C1 = GB_mex_mxm_update (F2, semiring, A4, B3, dtn) ;
                        C0 = GB_spec_mxm (F2,[ ], monoid, semiring, A4, B3, dtn);
                        GB_spec_compare (C0, C1, identity) ;

                    else

                        % F += A * B
                        C1 = GB_mex_mxm_update  (F, semiring, A, B, [ ]) ;
                        C0 = GB_spec_mxm (F, [ ], monoid, semiring, A, B, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;
                        % F += Afull * B
                        C1 = GB_mex_mxm_update  (F, semiring, Afull, B, [ ]) ;
                        C0 = GB_spec_mxm (F, [ ], monoid, semiring, Afull, B, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;
                        % F += A * Bfull
                        C1 = GB_mex_mxm_update  (F, semiring, A, Bfull, [ ]) ;
                        C0 = GB_spec_mxm (F, [ ], monoid, semiring, A, Bfull, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;

                        % F += A'*Bfull, no Mask, no typecasting
                        C1 = GB_mex_mxm_update (F, semiring, AT, Bfull, dtn) ;
                        C0 = GB_spec_mxm (F, [ ], monoid, semiring, AT, Bfull, dtn);
                        GB_spec_compare (C0, C1, identity) ;

                        % F += A_iso_full * B
                        Afull.iso = true ;
                        C1 = GB_mex_mxm_update  (F, semiring, Afull, B, [ ]) ;
                        C0 = GB_spec_mxm (F, [ ], monoid, semiring, Afull, B, [ ]) ;
                        GB_spec_compare (C0, C1, identity) ;
                        Afull.iso = false ;

                        % F += A_iso_bitmap * B
                        % A_bitmap.iso = true ;
                        C1 = GB_mex_mxm_update  (F, semiring, A_bitmap, B, [ ]) ;
                        C0 = GB_spec_mxm (F, [], monoid, semiring, A_bitmap, B, []);
                        GB_spec_compare (C0, C1, identity) ;
                        % A_bitmap.iso = false ;

                    end

                    if (track_coverage)
                        c = sum (GraphBLAS_grbcov > 0) ;
                        d = c - clast ;
                        if (d > 0)
                            fprintf ('[%d %d] %d\n', d, c, k0) ;
                        else
                            fprintf ('                      [-] %d\n', k0) ;
                        end
                        clast = c ;
                    else
                        fprintf ('.') ;
                    end

                end
            end
        end
    end
end

fprintf ('\nsemirings tested: %d\n', n_semirings) ;
fprintf ('\ntest74: all tests passed\n') ;

