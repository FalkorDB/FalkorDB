//------------------------------------------------------------------------------
// grb_jitpackage: package GraphBLAS source code for the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

//------------------------------------------------------------------------------
// zstd.h include file
//------------------------------------------------------------------------------

// ZSTD uses switch statements with no default case.
#if defined (__GNUC__)
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

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
// safe_strlen:  compute the length of a string
//------------------------------------------------------------------------------

// This is the same as the POSIX strnlen, but just using my own method.
// Non-POSIX systems do not have strnlen.

// This method returns the # of bytes in the string s, excluding the null
// terminating byte.  If s has no null terminating byte, it returns maxlen.

#define BUF_LENGTH 4096

size_t safe_strlen (const char *s, size_t maxlen) ;

size_t safe_strlen (const char *s, size_t maxlen)
{
    if (s == NULL) return (0) ;
    for (size_t k = 0 ; k < maxlen ; k++)
    {
        if (s [k] == '\0') return (k) ;
    }
    return (maxlen) ;
}

//------------------------------------------------------------------------------
// match_prefix: return true if the input string matches the prefix
//------------------------------------------------------------------------------

bool match_prefix (char *string, char *prefix) ;

bool match_prefix (char *string, char *prefix)
{
    char *s = string ;
    char *p = prefix ;
    while (*s && *p)
    {
        if (*s != *p)
        {
            // both the string and prefix character are present, but
            // do not match
            return (false) ;
        }
        s++ ;
        p++ ;
        if (*p == '\0')
        {
            // the prefix is exhausted, so it has been found as the first part
            // of the string
            return (true) ;
        }
    }
    return (false) ;
}

//------------------------------------------------------------------------------
// grb_prepackage main program
//------------------------------------------------------------------------------

#define OK(x)                                                               \
{                                                                           \
    if (!(x))                                                               \
    {                                                                       \
        fprintf (stderr, "grb_jitpackage.c: error line %d\n", __LINE__) ;   \
        abort ( ) ;                                                         \
    }                                                                       \
}

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // get list of files to be processed
    //--------------------------------------------------------------------------

    char **file_list = NULL;
    size_t nfiles = 0;
    fprintf (stderr, "grb_jitpackage: building JITpackage\n") ;

    if (argc == 2 && argv[1][0] == '@')
    {
        // input argument is a "response file" containing the file list

        // open file
        FILE *fr = fopen (argv[1]+1, "r") ;
        OK (fr != NULL) ;

        // get number of lines in file
        int ch;
        do
        {
            ch = fgetc (fr);
            if (ch == '\n')
            {
                nfiles++;
            }
        } while (ch != EOF);

        // read file list from response file
        rewind (fr);
        file_list = malloc ( (nfiles+1) * sizeof (file_list) );
        OK (file_list != NULL) ;
        // prepend empty element for compatibility with argv
        file_list[0] = malloc (1);
        OK (file_list [0] != NULL) ;
        file_list[0][0] = '\0';
        // glibc defines MAX_PATH to 4096.
        // Use this as a buffer size on all platforms.
        char temp[BUF_LENGTH];
        size_t length;
        for (size_t i = 1 ; i < nfiles+1 ; i++)
        {
            OK ( fgets (temp, BUF_LENGTH, fr) != NULL );
            length = safe_strlen (temp, BUF_LENGTH);
            file_list[i] = malloc (length+1);
            OK (file_list [i] != NULL) ;
            strncpy (file_list[i], temp, length);
            file_list[i][length-1] = '\0';
        }
    }
    else
    {
//      // input argument list is the file list
//      nfiles = argc - 1 ;
//      file_list = argv;
        fprintf (stderr, "feature no longer supported\n") ;
        OK (false) ;
    }

    //--------------------------------------------------------------------------
    // start the GB_JITpackage.c file
    //--------------------------------------------------------------------------

    FILE *fp = fopen ("GB_JITpackage.c", "wb") ;
    OK (fp != NULL) ;
    fprintf (stderr, "Processing %zu input files ...\n", nfiles) ;

    fprintf (fp,
        "//------------------------------------------------------------------------------\n"
        "// GB_JITpackage.c: packaged GraphBLAS source code for the JIT\n"
        "//------------------------------------------------------------------------------\n"
        "\n"
        "// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "\n"
        "//------------------------------------------------------------------------------\n"
        "\n"
        "#include \"GB_JITpackage.h\"\n"
        "\n"
        "#ifdef NJIT\n"
        "// JIT is disabled at compile time\n"
        "int GB_JITpackage_nfiles_get (void) { return (0) ; }\n"
        "static GB_JITpackage_index_struct GB_JITpackage_index [1] =\n"
        "   {{0, 0, NULL, NULL}} ;\n"
        "#else\n"
        "int GB_JITpackage_nfiles_get (void) { return (%zu) ; }\n\n", nfiles) ;

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

    for (size_t k = 1 ; k < nfiles+1 ; k++)
    {

        //----------------------------------------------------------------------
        // read the input file
        //----------------------------------------------------------------------

//      fprintf (stderr, "k: %zu file: %s\n", k, file_list [k]) ;
        FILE *ff = fopen (file_list [k], "rb") ; // open as binary, for Windows
        OK (ff != NULL) ;
        fseek (ff, 0, SEEK_END) ;
        size_t inputsize = ftell (ff) ;
        OK (inputsize > 0) ;
        rewind (ff) ;
        char *input = malloc (inputsize+2) ;
        OK (input != NULL) ;
        size_t nread = fread (input, sizeof (char), inputsize, ff) ;
//      fprintf (stderr, "inputsize %zu nread %zu\n", inputsize, nread) ;
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

        fprintf (fp, "// %s:\n", file_list [k]) ;
        fprintf (fp, "uint8_t GB_JITpackage_%zu [%zu] = {\n", k-1, dsize) ;
        for (uint64_t k = 0 ; k < dsize ; k++)
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

    fprintf (stderr, "Total uncompressed: %zu bytes\n",
        total_uncompressed_size) ;
    fprintf (stderr, "Total compressed:   %zu bytes\n", total_compressed_size) ;
    fprintf (stderr, "Compression:        %g\n",
        (double) total_compressed_size / (double) total_uncompressed_size) ;

    fprintf (fp, "\nstatic GB_JITpackage_index_struct "
        "GB_JITpackage_index [%zu] =\n{\n", nfiles) ;
    for (int k = 1 ; k < ((int) nfiles) + 1 ; k++)
    {
        // get the filename
        char *fullname = file_list [k] ;
        char *filename = fullname ;
        int len = (int) safe_strlen (fullname, BUF_LENGTH) ;
        for (int i = 0 ; i < len ; i++)
        {
            if (fullname [i] == '/')
            {
                filename = fullname + i + 1 ;
                if (match_prefix (filename, "template") ||
                    match_prefix (filename, "include"))
                {
                    break ;
                }
            }
        }
        // append this file to the index
        fprintf (fp, "    { %8zu, %8zu, GB_JITpackage_%-3d, \"%s\" },\n",
            Uncompressed_size [k], Compressed_size [k], k-1, filename) ;
    }
    fprintf (fp,
        "} ;\n#endif\n\n"
        "void *GB_JITpackage_index_get (void)\n"
        "{\n"
        "    return ((void *) GB_JITpackage_index) ;\n"
        "}\n\n") ;
    fclose (fp) ;
    free (Uncompressed_size) ;
    free (Compressed_size) ;
    return (0) ;
}

