function result = binops
%GRB.BINOPS list all binary ops
% Redundant binary ops are not listed.  For example '+.logical'
% exists, but it is identical to '|.logical'.
%
% Example:
%   GrB.binops ;            % prints a list, with descriptions
%   list = GrB.binops ;     % returns the list (nothing printed)
%
% See also GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = {
    'logical'
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    'single complex'
    'double complex'
    } ;

binops = gb_binops ;

skip_logical = {'min', 'max', '+', '-', 'rminus', '*', '/', '\', 'iseq', ...
    'isne', 'isgt', 'islt', 'isle', 'isge', 'pow', '~=', '==' } ;

nbinops = 0 ;
nops = size (binops, 1) ;

if (nargout > 0)
    result = { } ;
end

    for k2 = 1:nops
        mult = binops {k2,1} ;
        mult_description = binops {k2,2} ;
        first_mult = true ;
        for k3 = 1:length (types)
            type = types {k3} ;
            ok = false ;
            binop = [mult '.' type] ;

            % skip redundant logical binops
            if (isequal (type, 'logical') && ...
                ismember (mult, skip_logical))
                continue ;
            end

            try
                ok = gbbinopinfo (binop) ;
                nbinops = nbinops + 1 ;
                if (nargout > 0)
                    result = [result ; binop] ; %#ok<AGROW>
                end
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid binop.
            end
            if (ok && nargout == 0)
                if (first_mult)
                    fprintf ('binary op: %s.type', mult) ;
                    if (isempty (mult_description))
                        fprintf ('\n') ;
                    else
                        fprintf (', where %s\n', mult_description) ;
                    end
                    fprintf ('        types: %s', type) ;
                else
                    fprintf (', %s', type) ;
                end
                first_mult = false ;
            end
        end
        if (nargout == 0)
            fprintf ('\n\n') ;
        end
    end

if (nargout == 0)
    fprintf ('Total number of available binary ops: %d\n', nbinops) ;
end

