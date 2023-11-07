function gbtest5
%GBTEST5 test GrB.descriptorinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list_out  = { [ ], 'default', 'replace' } ;
list_in   = { [ ], 'default', 'transpose' } ;
list_mask = { [ ], 'default', 'complement', ...
                   'structural complement', 'structure' } ;
list_axb  = { [ ], 'default', 'gustavson', 'dot', 'hash', 'saxpy' } ;
list_kind = { [ ], 'sparse', 'full', 'grb', 'default' } ;

ntrials = 0;

d = struct %#ok<*NASGU,*NOPRT>

for k1 = 1:length (list_out)
    out = list_out {k1} ;
    for k2 = 1:length (list_in)
        in0 = list_in {k2} ;
        for k3 = 1:length (list_in)
            in1 = list_in {k3} ;
            for k4 = 1:length (list_mask)
                mask = list_mask {k4} ;
                for k5 = 1:length (list_axb)
                    axb = list_axb {k5} ;
                    for k6 = 1:length (list_kind)
                        kind = list_kind {k6} ;

%                       for nthreads = [0 2]
%                           for chunk = [0 10000]

                                clear d
                                d = struct ;

                                if (~isempty (out))
                                    d.out = out ;
                                end

                                if (~isempty (mask))
                                    d.mask = mask ;
                                end

                                if (~isempty (in0))
                                    d.in0 = in0 ;
                                end

                                if (~isempty (in1))
                                    d.in1 = in1 ;
                                end

                                if (~isempty (axb))
                                    d.axb = axb ;
                                end

                                if (~isempty (kind))
                                    d.kind = kind ;
                                end

%                               if (nthreads > 0)
%                                   d.nthreads = nthreads ;
%                               end

%                               if (chunk > 0)
%                                   d.chunk = chunk ;
%                               end

                                d
                                GrB.descriptorinfo (d) ;
                                ntrials = ntrials + 1 ;
%                           end
%                       end
                    end
                end
            end
        end
    end
end

fprintf ('testing error handling (errors expected):\n') ;

GrB.descriptorinfo

ntrials
fprintf ('gbtest5: all tests passed\n') ;

