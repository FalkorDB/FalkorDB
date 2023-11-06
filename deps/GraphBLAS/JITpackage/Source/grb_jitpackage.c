//------------------------------------------------------------------------------
// grb_jitpackage: package GraphBLAS source code for the JIT 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// zstd.h include file
//------------------------------------------------------------------------------

// ZSTD uses switch statements with no default case.
#pragma GCC diagnostic ignored "-Wswitch-default"

// disable ZSTD deprecation warnings and include all ZSTD definitions  

// GraphBLAS does not use deprecated functions, but the warnings pop up anyway
// when GraphBLAS is built, so silence them with this #define:
#define ZSTD_DISABLE_DEPRECATE_WARNINGS

// do not use multithreading in ZSTD itself.
#undef ZSTD_MULTITHREAD

// do not use asm
#define ZSTD_DISABLE_ASM

#include "zstd.h"

//------------------------------------------------------------------------------
// zstd source (subset used by GraphBLAS)
//------------------------------------------------------------------------------

// Include the unmodified zstd, version 1.5.3.  This ensures that any files
// compressed with grb_jitpackage can be uncompressed with the same zstd
// subset that resides within libgraphblas.so itself.

#include "zstd_subset/common/debug.c"
#include "zstd_subset/common/entropy_common.c"
#include "zstd_subset/common/error_private.c"
#include "zstd_subset/common/fse_decompress.c"
#include "zstd_subset/common/pool.c"
#include "zstd_subset/common/threading.c"
#include "zstd_subset/common/xxhash.c"
#include "zstd_subset/common/zstd_common.c"

#include "zstd_subset/compress/fse_compress.c"
#include "zstd_subset/compress/hist.c"
#include "zstd_subset/compress/huf_compress.c"
#include "zstd_subset/compress/zstd_compress.c"
#include "zstd_subset/compress/zstd_compress_literals.c"
#include "zstd_subset/compress/zstd_compress_sequences.c"
#include "zstd_subset/compress/zstd_compress_superblock.c"
#include "zstd_subset/compress/zstd_double_fast.c"
#include "zstd_subset/compress/zstd_fast.c"
#include "zstd_subset/compress/zstd_lazy.c"
#include "zstd_subset/compress/zstd_ldm.c"
#include "zstd_subset/compress/zstdmt_compress.c"
#include "zstd_subset/compress/zstd_opt.c"

/* no need for decompression here
#include "zstd_subset/decompress/huf_decompress.c"
#include "zstd_subset/decompress/zstd_ddict.c"
#include "zstd_subset/decompress/zstd_decompress_block.c"
#include "zstd_subset/decompress/zstd_decompress.c"
*/

//------------------------------------------------------------------------------
// grb_prepackage main program
//------------------------------------------------------------------------------

#define OK(x) if (!(x)) { printf ("Error line %d\n", __LINE__) ; abort ( ) ; }

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // start the GB_JITpackage.c file
    //--------------------------------------------------------------------------

    FILE *fp = fopen ("GB_JITpackage.c", "w") ;
    OK (fp != NULL) ;
    int nfiles = argc - 1 ;
    printf ("Processing %d input files ...\n", nfiles) ;

    fprintf (fp,
        "//------------------------------------------------------------------------------\n"
        "// GB_JITpackage.c: packaged GraphBLAS source code for the JIT\n"
        "//------------------------------------------------------------------------------\n"
        "\n"
        "// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "\n"
        "//------------------------------------------------------------------------------\n"
        "\n"
        "#include \"GB_JITpackage.h\"\n"
        "\n"
        "#ifdef NJIT\n"
        "// JIT is disabled at compile time\n"
        "int GB_JITpackage_nfiles = 0 ;\n"
        "GB_JITpackage_index_struct GB_JITpackage_index [1] "
        "= {{0, 0, NULL, NULL}} ;\n"
        "#else\n"
        "int GB_JITpackage_nfiles = %d ;\n\n", argc-1) ;

    //--------------------------------------------------------------------------
    // allocate the index
    //--------------------------------------------------------------------------

    size_t *Uncompressed_size = calloc (nfiles + 1, sizeof (size_t)) ;
    size_t *Compressed_size   = calloc (nfiles + 1, sizeof (size_t)) ;
    OK (Uncompressed_size != NULL) ;
    OK (Compressed_size != NULL) ;
    size_t total_compressed_size = 0 ;
    size_t total_uncompressed_size = 0 ;

    //--------------------------------------------------------------------------
    // compress each file
    //--------------------------------------------------------------------------

    for (int k = 1 ; k < argc ; k++)
    {

        //----------------------------------------------------------------------
        // read the input file
        //----------------------------------------------------------------------

        FILE *ff = fopen (argv [k], "r") ;
        OK (ff != NULL) ;
        fseek (ff, 0, SEEK_END) ;
        size_t inputsize = ftell (ff) ;
        OK (inputsize > 0) ;
        rewind (ff) ;
        char *input = malloc (inputsize+2) ;
        OK (input != NULL) ;
        size_t nread = fread (input, sizeof (char), inputsize, ff) ;
        OK (nread == inputsize) ;
        input [inputsize] = '\0' ; 
        fclose (ff) ;

        //----------------------------------------------------------------------
        // compress the file into dst (level 19)
        //----------------------------------------------------------------------

        size_t dbound = ZSTD_compressBound (inputsize) ;
        uint8_t *dst = malloc (dbound+2) ;
        OK (dst != NULL) ;
        size_t dsize = ZSTD_compress (dst, dbound+2, input, inputsize, 19) ;

        //----------------------------------------------------------------------
        // append the bytes to the output file 
        //----------------------------------------------------------------------

        fprintf (fp, "// %s:\n", argv [k]) ;
        fprintf (fp, "uint8_t GB_JITpackage_%d [%lu] = {\n", k-1, dsize) ;
        for (int64_t k = 0 ; k < dsize ; k++)
        {
            fprintf (fp, "%3d,", dst [k]) ;
            if (k % 20 == 19) fprintf (fp, "\n") ;
        }
        fprintf (fp, "\n} ;\n\n") ;
        free (dst) ;
        free (input) ;

        //----------------------------------------------------------------------
        // save the file info
        //----------------------------------------------------------------------

        Uncompressed_size [k] = inputsize ;
        Compressed_size   [k] = dsize ;
        total_uncompressed_size += inputsize ;
        total_compressed_size += dsize ;
    }

    //--------------------------------------------------------------------------
    // print the index
    //--------------------------------------------------------------------------

    printf ("Total uncompressed: %lu bytes\n", total_uncompressed_size) ;
    printf ("Total compressed:   %lu bytes\n", total_compressed_size) ;
    printf ("Compression:        %g\n", 
        (double) total_compressed_size / (double) total_uncompressed_size) ;

    fprintf (fp, "\nGB_JITpackage_index_struct GB_JITpackage_index [%d] =\n{\n",
        nfiles) ;
    for (int k = 1 ; k < argc ; k++)
    {
        // get the filename (without the path)
        char *name = argv [k] ;
        for (char *p = argv [k] ; *p != '\0' ; p++)
        {
            if (*p == '/')
            {
                name = p + 1 ;
            }
        }
        // append this file to the index
        fprintf (fp, "    { %8lu, %8lu, GB_JITpackage_%-3d, \"%s\" },\n",
            Uncompressed_size [k], Compressed_size [k], k-1, name) ;
    }
    fprintf (fp, "} ;\n#endif\n\n") ;
    fclose (fp) ;
    free (Uncompressed_size) ;
    free (Compressed_size) ;
}

