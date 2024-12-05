function codegen_type_enabled (f, fname)
% CODEGEN_TYPE_ENABLED create the GB_TYPE_ENABLED macro

fprintf (f, 'm4_define(`GB_type_enabled'', `#if defined (GxB_NO_%s)\n#define GB_TYPE_ENABLED 0\n#else\n#define GB_TYPE_ENABLED 1\n#endif\n'')\n', upper (fname)) ;
