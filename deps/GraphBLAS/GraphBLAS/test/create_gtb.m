%CREATE_GTB create gtb_* test methods for gbtest

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

gtbs = {
'apply2'
'apply'
'argmax'
'argmin'
'argsort'
'assign'
'bfs'
'binopinfo'
'binops'
'build'
'burble'
'cell2mat'
'chunk'
'clear'
'compact'
'descriptorinfo'
'deserialize'
'dnn'
'eadd'
'empty'
'emult'
'entries'
'eunion'
'expand'
'extract'
'extracttuples'
'eye'
'false'
'finalize'
'format'
'GrB' % special case, needs editting afterwards
'incidence'
'init'
'isbycol'
'isbyrow'
'isfull'
'issigned'
'jit'
'kronecker'
'ktruss'
'laplacian'
'load'
'mis'
'monoidinfo'
'monoids'
'mxm'
'nmalloc'
'nonz'
'normdiff'
'nvals'
'offdiag'
'ones'
'optype'
'pagerank'
'print'
'prune'
'random'
'reduce'
'save'
'select'
'selectopinfo'
'selectops'
'semiringinfo'
'semirings'
'serialize'
'subassign'
'threads'
'speye'
'trans'
'tricount'
'true'
'type'
'unopinfo'
'unops'
'ver'
'version'
'vreduce'
'wait'
'zeros' } ;

for k = 1:length (gtbs)
    gtb_func = gtbs {k} ;
    fprintf ('%s ', gtb_func) ;
    f = fopen (['gtb_' gtb_func '.m'], 'w') ;
    fprintf (f, 'function [varargout] = gtb_%s (ghb, varargin)\n', ...
        gtb_func) ;
    fprintf (f, '%%GTB_%s wrapper for GrB.%s and GhB.%s\n', ...
        upper (gtb_func), gtb_func, gtb_func) ;
    fprintf (f, '\n') ;
    fprintf (f, '%% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.\n') ;
    fprintf (f, '%% SPDX-License-Identifier: Apache-2.0\n') ;
    fprintf (f, '\n') ;
    fprintf (f, 'if (~(ghb == 0 || ghb == 1))\n') ;
    fprintf (f, '    ghb = rand (1) > 0.5 ; %% choose ghb at random\n') ;
    fprintf (f, 'end\n') ;
    fprintf (f, 'if (ghb)\n') ;
    fprintf (f, '    [varargout{1:nargout}] = GhB.%s (varargin {:}) ;\n', ...
        gtb_func) ;
    fprintf (f, 'else\n') ;
    fprintf (f, '    [varargout{1:nargout}] = GrB.%s (varargin {:}) ;\n', ...
        gtb_func) ;
    fprintf (f, 'end\n\n') ;
    fclose (f) ;
end
fprintf ('\n') ;

