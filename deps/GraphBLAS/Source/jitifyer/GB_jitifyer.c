//------------------------------------------------------------------------------
// GB_jitifyer.c: CPU / CUDA jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"
#include "GB_config.h"
#include "zstd_wrapper/GB_zstd.h"
#include "JITpackage/GB_JITpackage.h"
#include "jitifyer/GB_file.h"

typedef GB_JIT_KERNEL_USER_OP_PROTO ((*GB_user_op_f)) ;
typedef GB_JIT_KERNEL_USER_TYPE_PROTO ((*GB_user_type_f)) ;

//------------------------------------------------------------------------------
// static objects:  hash table, strings, and status
//------------------------------------------------------------------------------

// The hash table is static and shared by all threads of the user application.
// It is only visible inside this file.  It starts out empty (NULL).  Its size
// is either zero (at the beginning), or a power of two (of size
// GB_JITIFIER_INITIAL_SIZE or more).

// The strings are used to create filenames and JIT compilation commands.

#ifdef GBCOVER
// use a smaller JIT table size during test coverage
#define GB_JITIFIER_INITIAL_SIZE (1024)
#else
#define GB_JITIFIER_INITIAL_SIZE (32*1024)
#endif

static GB_jit_entry *GB_jit_table = NULL ;
static int64_t  GB_jit_table_size = 0 ;  // always a power of 2
static uint64_t GB_jit_table_bits = 0 ;  // hash mask (0xFFFF if size is 2^16)
static int64_t  GB_jit_table_populated = 0 ;
static size_t   GB_jit_table_allocated = 0 ;

static bool GB_jit_use_cmake =
    #if defined (_MSC_VER)
    true ;      // MSVC requires cmake
    #else
    false ;     // otherwise, default is to skip cmake and compile directly
    #endif

// path to user cache folder:
static char    *GB_jit_cache_path = NULL ;
static size_t   GB_jit_cache_path_allocated = 0 ;

// path to error log file:
static char    *GB_jit_error_log = NULL ;
static size_t   GB_jit_error_log_allocated = 0 ;

// name of the C compiler:
static char    *GB_jit_C_compiler = NULL ;
static size_t   GB_jit_C_compiler_allocated = 0 ;

// flags for the C compiler:
static char    *GB_jit_C_flags = NULL ;
static size_t   GB_jit_C_flags_allocated = 0 ;

// link flags for the C compiler:
static char    *GB_jit_C_link_flags = NULL ;
static size_t   GB_jit_C_link_flags_allocated = 0 ;

// libraries to link against when using the direct compile/link:
static char    *GB_jit_C_libraries = NULL ;
static size_t   GB_jit_C_libraries_allocated = 0 ;

// libraries to link against when using cmake:
static char    *GB_jit_C_cmake_libs = NULL ;
static size_t   GB_jit_C_cmake_libs_allocated = 0 ;

// preface to add to each CPU JIT kernel:
static char    *GB_jit_C_preface = NULL ;
static size_t   GB_jit_C_preface_allocated = 0 ;

// preface to add to each CUDA JIT kernel:
static char    *GB_jit_CUDA_preface = NULL ;
static size_t   GB_jit_CUDA_preface_allocated = 0 ;

// temporary workspace for filenames and system commands:
static char    *GB_jit_temp = NULL ;
static size_t   GB_jit_temp_allocated = 0 ;

// compile with -DJITINIT=4 (for example) to set the initial JIT C control
#ifdef JITINIT
#define GB_JIT_C_CONTROL_INIT JITINIT
#else
// default initial state
#define GB_JIT_C_CONTROL_INIT GxB_JIT_ON
#endif

static int GB_jit_control = GB_JIT_C_CONTROL_INIT ;

//------------------------------------------------------------------------------
// check_table: check if the hash table is OK
//------------------------------------------------------------------------------

#ifdef GB_DEBUG
static void check_table (void)
{
    int64_t populated = 0 ;
    if (GB_jit_table != NULL)
    {
        for (uint64_t k = 0 ; k < GB_jit_table_size ; k++)
        {
            GB_jit_entry *e = &(GB_jit_table [k]) ;
            if (e->dl_function != NULL)
            {
                populated++ ;
            }
        }
    }
    ASSERT (populated == GB_jit_table_populated) ;
}
#define ASSERT_TABLE_OK check_table ( ) ;
#else
#define ASSERT_TABLE_OK
#endif

//------------------------------------------------------------------------------
// malloc/free macros
//------------------------------------------------------------------------------

// The JIT must use persistent malloc/free methods when GraphBLAS is used in
// MATLAB.  Outside of MATLAB, these are the same as malloc/free passed to
// GxB_init (or ANSI C malloc/free if using GrB_init).  Inside MATLAB,
// GB_Global_persistent_malloc uses the same malloc/free given to GxB_init, but
// then calls mexMakeMemoryPersistent to ensure the memory is not freed when a
// mexFunction returns to the MATLAB m-file caller.

#define OK(method)                      \
{                                       \
    GrB_Info myinfo = (method) ;        \
    if (myinfo != GrB_SUCCESS)          \
    {                                   \
        return (myinfo) ;               \
    }                                   \
}

#ifdef GB_MEMDUMP

    #define GB_MALLOC_PERSISTENT(X,siz)                     \
    {                                                       \
        X = GB_Global_persistent_malloc (siz) ;             \
        GBMDUMP ("persistent malloc (%4d): %p size %g\n",   \
            __LINE__, (void *) X, (double) siz) ;           \
    }

    #define GB_FREE_PERSISTENT(X)                           \
    {                                                       \
        if (X != NULL)                                      \
        {                                                   \
            GBMDUMP ("persistent free   (%4d): %p\n",       \
            __LINE__, (void *) X) ;                         \
        }                                                   \
        GB_Global_persistent_free ((void **) &(X)) ;        \
    }

#else

    #define GB_MALLOC_PERSISTENT(X,siz)                     \
    {                                                       \
        X = GB_Global_persistent_malloc (siz) ;             \
    }

    #define GB_FREE_PERSISTENT(X)                           \
    {                                                       \
        GB_Global_persistent_free ((void **) &(X)) ;        \
    }

#endif

#define GB_FREE_STUFF(X)                                \
{                                                       \
    GB_FREE_PERSISTENT (X) ;                            \
    X ## _allocated = 0 ;                               \
}

#define GB_MALLOC_STUFF(X,len)                          \
{                                                       \
    GB_MALLOC_PERSISTENT (X, (len) + 2) ;               \
    if (X == NULL)                                      \
    {                                                   \
        return (GrB_OUT_OF_MEMORY) ;                    \
    }                                                   \
    X ## _allocated = (len) + 2 ;                       \
}

#define GB_COPY_STUFF(X,src)                            \
{                                                       \
    ASSERT (src != NULL) ;                              \
    size_t len = strlen (src) ;                         \
    GB_MALLOC_STUFF (X, len) ;                          \
    strncpy (X, src, X ## _allocated) ;                 \
}

//------------------------------------------------------------------------------
// GB_jitifyer_finalize: free the JIT table and all the strings
//------------------------------------------------------------------------------

void GB_jitifyer_finalize (void)
{ 
    GB_jitifyer_table_free (true) ;
    GB_FREE_STUFF (GB_jit_cache_path) ;
    GB_FREE_STUFF (GB_jit_error_log) ;
    GB_FREE_STUFF (GB_jit_C_compiler) ;
    GB_FREE_STUFF (GB_jit_C_flags) ;
    GB_FREE_STUFF (GB_jit_C_link_flags) ;
    GB_FREE_STUFF (GB_jit_C_libraries) ;
    GB_FREE_STUFF (GB_jit_C_cmake_libs) ;
    GB_FREE_STUFF (GB_jit_C_preface) ;
    GB_FREE_STUFF (GB_jit_CUDA_preface) ;
    GB_FREE_STUFF (GB_jit_temp) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_sanitize
//------------------------------------------------------------------------------

// Replace invalid characters in a string with underscore.

// Valid characters: letters, numbers, space, and four others (dot, dash,
// underscore, and slash).  Backslash is valid but replaced with slash.
// All other invalid characters are replaced with underscore.

void GB_jitifyer_sanitize (char *string, size_t len)
{
    for (int k = 0 ; k < len ; k++)
    {
        // check for the end of the string
        if (string [k] == '\0') break ;
        #ifdef _WIN32
        // check the colon for "C:...", only in the second character
        if (k == 1 && string [k] == ':') continue ;
        #endif
        // replace backslash with forward slash
        if (string [k] == '\\')
        { 
            string [k] = '/' ;
            continue ;
        }
        // replace other invalid characters with "_"
        static char valid_character_set [ ] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 .-_/" ;
        bool ok = false ;
        for (char *s = valid_character_set ; *s != '\0' ; s++)
        {
            if (string [k] == *s)
            { 
                ok = true ;
                break ;
            }
        }
        if (!ok)
        { 
            // replace a bad character with an underscore
            string [k] = '_' ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_init: initialize the JIT folders, flags, etc
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS or GrB_OUT_OF_MEMORY.  If any other error occurs (such
// as being unable to access the cache folder), the JIT is disabled, but
// GrB_SUCCESS is returned.  This is because GrB_init calls this method, and
// GraphBLAS can continue without the JIT.

GrB_Info GB_jitifyer_init (void)
{ 
    #if defined ( GRAPHBLAS_HAS_CUDA )
    int device = -1 ;
    GB_cuda_get_device (&device) ;
    printf ("JIT init, device %d\n", device) ;  // for CUDA only
    #endif

    //--------------------------------------------------------------------------
    // initialize the JIT control
    //--------------------------------------------------------------------------

    int control = (int) GB_JIT_C_CONTROL_INIT ;
    control = GB_IMAX (control, (int) GxB_JIT_OFF) ;
    #ifndef NJIT
    // The full JIT is available.
    control = GB_IMIN (control, (int) GxB_JIT_ON) ;
    #else
    // The JIT is restricted; only OFF, PAUSE, and RUN settings can be
    // used.  No JIT kernels can be loaded or compiled.  Only PreJIT kernels
    // can be used.
    control = GB_IMIN (control, (int) GxB_JIT_RUN) ;
    #endif
    GB_jit_control = control ;

    GB_jitifyer_finalize ( ) ;

    //--------------------------------------------------------------------------
    // find the GB_jit_cache_path
    //--------------------------------------------------------------------------

    char *cache_path = getenv ("GRAPHBLAS_CACHE_PATH") ;
    if (cache_path != NULL)
    { 
        // use the environment variable GRAPHBLAS_CACHE_PATH as-is
        GB_COPY_STUFF (GB_jit_cache_path, cache_path) ;
    }
    else
    { 
        // Linux, Mac, Unix: look for HOME
        char *home = getenv ("HOME") ;
        char *dot = "." ;
        if (home == NULL)
        {
            // Windows: look for LOCALAPPDATA
            home = getenv ("LOCALAPPDATA") ;
            dot = "" ;
        }
        if (home != NULL)
        { 
            // found home; create the cache path
            size_t len = strlen (home) + 60 ;
            GB_MALLOC_STUFF (GB_jit_cache_path, len) ;
            snprintf (GB_jit_cache_path, GB_jit_cache_path_allocated,
                "%s/%sSuiteSparse/GrB%d.%d.%d"
                #if defined ( GBMATLAB ) && defined ( __APPLE__ )
                "_matlab"
                #endif
                , home, dot,
                GxB_IMPLEMENTATION_MAJOR,
                GxB_IMPLEMENTATION_MINOR,
                GxB_IMPLEMENTATION_SUB) ;
        }
    }

    if (GB_jit_cache_path == NULL)
    {
        // cannot determine the JIT cache.  Disable loading and compiling, but
        // continue with the rest of the initializations.  The PreJIT could
        // still be used.
        GBURBLE ("(jit init: unable to access cache path, jit disabled) ") ;
        GB_jit_control = GxB_JIT_RUN ;
        GB_FREE_STUFF (GB_jit_cache_path) ;
        GB_COPY_STUFF (GB_jit_cache_path, "") ;
    }

    // sanitize the cache path
    GB_jitifyer_sanitize (GB_jit_cache_path, GB_jit_cache_path_allocated) ;

    //--------------------------------------------------------------------------
    // initialize the remaining strings
    //--------------------------------------------------------------------------

    GB_COPY_STUFF (GB_jit_error_log,    "") ;
    GB_COPY_STUFF (GB_jit_C_compiler,   GB_C_COMPILER) ;
    GB_COPY_STUFF (GB_jit_C_flags,      GB_C_FLAGS) ;
    GB_COPY_STUFF (GB_jit_C_link_flags, GB_C_LINK_FLAGS) ;
    GB_COPY_STUFF (GB_jit_C_libraries,  GB_C_LIBRARIES) ;
    GB_COPY_STUFF (GB_jit_C_cmake_libs, GB_CMAKE_LIBRARIES) ;
    GB_COPY_STUFF (GB_jit_C_preface,    "") ;
    GB_COPY_STUFF (GB_jit_CUDA_preface, "") ;
    OK (GB_jitifyer_alloc_space ( )) ;

    //--------------------------------------------------------------------------
    // establish the cache path and src path, and make sure they exist
    //--------------------------------------------------------------------------

    OK (GB_jitifyer_establish_paths (GrB_SUCCESS)) ;

    //--------------------------------------------------------------------------
    // remove "-arch arm64" if compiling JIT kernels for MATLAB
    //--------------------------------------------------------------------------

    // When the x86-based version of gcc-12 is configured to compile the MATLAB
    // GraphBLAS library on an Apple-Silicon-based Mac, cmake gives it the flag
    // "-arch arm64".  MATLAB does not support that architecture directly,
    // using Rosetta 2 instead.  gcc-12 also does not support "-arch arm64", so
    // it ignores it (which is the right thing to do), but it generates a
    // warning.  This spurious warning message appears every time a JIT kernel
    // is compiled while inside MATLAB.  As a result, "-arch arm64" is removed
    // from the initial C flags, if compiling for MATLAB.

    #ifdef GBMATLAB
    {
        #define ARCH_ARM64 "-arch arm64"
        char *dst = strstr (GB_jit_C_flags, ARCH_ARM64) ;
        if (dst != NULL)
        {
            // found it; now remove it from the C flags
            char *src = dst + strlen (ARCH_ARM64) ;
            while (((*dst++) = (*src++))) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // hash all PreJIT kernels
    //--------------------------------------------------------------------------

    void **Kernels = NULL ;
    void **Queries = NULL ;
    char **Names = NULL ;
    int32_t nkernels = 0 ;
    GB_prejit (&nkernels, &Kernels, &Queries, &Names) ;

    for (int k = 0 ; k < nkernels ; k++)
    {

        //----------------------------------------------------------------------
        // get the name and function pointer of the PreJIT kernel
        //----------------------------------------------------------------------

        void *dl_function = Kernels [k] ;

//      GB_jit_query_func dl_query = (GB_jit_query_func) Queries [k] ;
        GB_jit_query_func dl_query = GB_jitifyer_get_query (Queries [k]) ;
        ASSERT (dl_function != NULL && dl_query != NULL && Names [k] != NULL) ;
        char kernel_name [GB_KLEN+1] ;
        strncpy (kernel_name, Names [k], GB_KLEN) ;
        kernel_name [GB_KLEN] = '\0' ;

        //----------------------------------------------------------------------
        // parse the kernel name
        //----------------------------------------------------------------------

        char *name_space = NULL ;
        char *kname = NULL ;
        uint64_t method_code = 0 ;
        char *suffix = NULL ;
        GrB_Info info = GB_demacrofy_name (kernel_name, &name_space, &kname,
            &method_code, &suffix) ;

        if (info != GrB_SUCCESS || !GB_STRING_MATCH (name_space, "GB_jit"))
        {
            // PreJIT error: kernel_name is invalid; ignore this kernel
            continue ;
        }

        //----------------------------------------------------------------------
        // find the kcode of the kname
        //----------------------------------------------------------------------

        GB_jit_encoding encoding_struct ;
        GB_jit_encoding *encoding = &encoding_struct ;
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;

        #define IS(kernel) GB_STRING_MATCH (kname, kernel)

        GB_jit_kcode c = 0 ;
        if      (IS ("add"          )) c = GB_JIT_KERNEL_ADD ;
        else if (IS ("apply_bind1st")) c = GB_JIT_KERNEL_APPLYBIND1 ;
        else if (IS ("apply_bind2nd")) c = GB_JIT_KERNEL_APPLYBIND2 ;
        else if (IS ("apply_unop"   )) c = GB_JIT_KERNEL_APPLYUNOP ;
        else if (IS ("AxB_dot2"     )) c = GB_JIT_KERNEL_AXB_DOT2 ;
        else if (IS ("AxB_dot2n"    )) c = GB_JIT_KERNEL_AXB_DOT2N ;
        else if (IS ("AxB_dot3"     )) c = GB_JIT_KERNEL_AXB_DOT3 ;
        else if (IS ("AxB_dot4"     )) c = GB_JIT_KERNEL_AXB_DOT4 ;
        else if (IS ("AxB_saxbit"   )) c = GB_JIT_KERNEL_AXB_SAXBIT ;
        else if (IS ("AxB_saxpy3"   )) c = GB_JIT_KERNEL_AXB_SAXPY3 ;
        else if (IS ("AxB_saxpy4"   )) c = GB_JIT_KERNEL_AXB_SAXPY4 ;
        else if (IS ("AxB_saxpy5"   )) c = GB_JIT_KERNEL_AXB_SAXPY5 ;
        else if (IS ("build"        )) c = GB_JIT_KERNEL_BUILD ;
        else if (IS ("colscale"     )) c = GB_JIT_KERNEL_COLSCALE ;
        else if (IS ("concat_bitmap")) c = GB_JIT_KERNEL_CONCAT_BITMAP ;
        else if (IS ("concat_full"  )) c = GB_JIT_KERNEL_CONCAT_FULL ;
        else if (IS ("concat_sparse")) c = GB_JIT_KERNEL_CONCAT_SPARSE ;
        else if (IS ("convert_s2b"  )) c = GB_JIT_KERNEL_CONVERT_S2B ;
        else if (IS ("emult_02"     )) c = GB_JIT_KERNEL_EMULT2 ;
        else if (IS ("emult_03"     )) c = GB_JIT_KERNEL_EMULT3 ;
        else if (IS ("emult_04"     )) c = GB_JIT_KERNEL_EMULT4 ;
        else if (IS ("emult_08"     )) c = GB_JIT_KERNEL_EMULT8 ;
        else if (IS ("emult_bitmap" )) c = GB_JIT_KERNEL_EMULT_BITMAP ;
        else if (IS ("ewise_fulla"  )) c = GB_JIT_KERNEL_EWISEFA ;
        else if (IS ("ewise_fulln"  )) c = GB_JIT_KERNEL_EWISEFN ;
        else if (IS ("reduce"       )) c = GB_JIT_KERNEL_REDUCE ;
        else if (IS ("rowscale"     )) c = GB_JIT_KERNEL_ROWSCALE ;
        else if (IS ("select_bitmap")) c = GB_JIT_KERNEL_SELECT_BITMAP ;
        else if (IS ("select_phase1")) c = GB_JIT_KERNEL_SELECT1 ;
        else if (IS ("select_phase2")) c = GB_JIT_KERNEL_SELECT2 ;
        else if (IS ("split_bitmap" )) c = GB_JIT_KERNEL_SPLIT_BITMAP ;
        else if (IS ("split_full"   )) c = GB_JIT_KERNEL_SPLIT_FULL ;
        else if (IS ("split_sparse" )) c = GB_JIT_KERNEL_SPLIT_SPARSE ;

        else if (IS ("subassign_05d")) c = GB_JIT_KERNEL_SUBASSIGN_05d ;
        else if (IS ("subassign_06d")) c = GB_JIT_KERNEL_SUBASSIGN_06d ;
        else if (IS ("subassign_22" )) c = GB_JIT_KERNEL_SUBASSIGN_22 ;
        else if (IS ("subassign_23" )) c = GB_JIT_KERNEL_SUBASSIGN_23 ;
        else if (IS ("subassign_25" )) c = GB_JIT_KERNEL_SUBASSIGN_25 ;

        else if (IS ("trans_bind1st")) c = GB_JIT_KERNEL_TRANSBIND1 ;
        else if (IS ("trans_bind2nd")) c = GB_JIT_KERNEL_TRANSBIND2 ;
        else if (IS ("trans_unop"   )) c = GB_JIT_KERNEL_TRANSUNOP ;
        else if (IS ("union"        )) c = GB_JIT_KERNEL_UNION ;
        else if (IS ("user_op"      )) c = GB_JIT_KERNEL_USEROP ;
        else if (IS ("user_type"    )) c = GB_JIT_KERNEL_USERTYPE ;

        // added for v9.4.1:
        else if (IS ("subassign_01" )) c = GB_JIT_KERNEL_SUBASSIGN_01 ;
        else if (IS ("subassign_02" )) c = GB_JIT_KERNEL_SUBASSIGN_02 ;
        else if (IS ("subassign_03" )) c = GB_JIT_KERNEL_SUBASSIGN_03 ;
        else if (IS ("subassign_04" )) c = GB_JIT_KERNEL_SUBASSIGN_04 ;
        else if (IS ("subassign_05" )) c = GB_JIT_KERNEL_SUBASSIGN_05 ;
        else if (IS ("subassign_06n")) c = GB_JIT_KERNEL_SUBASSIGN_06n ;
        else if (IS ("subassign_06s")) c = GB_JIT_KERNEL_SUBASSIGN_06s ;
        else if (IS ("subassign_07" )) c = GB_JIT_KERNEL_SUBASSIGN_07 ;
        else if (IS ("subassign_08n")) c = GB_JIT_KERNEL_SUBASSIGN_08n ;
        else if (IS ("subassign_08s")) c = GB_JIT_KERNEL_SUBASSIGN_08s ;
        else if (IS ("subassign_09" )) c = GB_JIT_KERNEL_SUBASSIGN_09 ;
        else if (IS ("subassign_10" )) c = GB_JIT_KERNEL_SUBASSIGN_10 ;
        else if (IS ("subassign_11" )) c = GB_JIT_KERNEL_SUBASSIGN_11 ;
        else if (IS ("subassign_12" )) c = GB_JIT_KERNEL_SUBASSIGN_12 ;
        else if (IS ("subassign_13" )) c = GB_JIT_KERNEL_SUBASSIGN_13 ;
        else if (IS ("subassign_15" )) c = GB_JIT_KERNEL_SUBASSIGN_15 ;
        else if (IS ("subassign_17" )) c = GB_JIT_KERNEL_SUBASSIGN_17 ;
        else if (IS ("subassign_19" )) c = GB_JIT_KERNEL_SUBASSIGN_19 ;

        else if (IS ("bitmap_assign_1"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_1 ;
        else if (IS ("bitmap_assign_1_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_1_WHOLE ;
        else if (IS ("bitmap_assign_2"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_2 ;
        else if (IS ("bitmap_assign_2_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_2_WHOLE ;
        else if (IS ("bitmap_assign_3"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_3 ;
        else if (IS ("bitmap_assign_3_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_3_WHOLE ;
        else if (IS ("bitmap_assign_4"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_4 ;
        else if (IS ("bitmap_assign_4_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_4_WHOLE ;
        else if (IS ("bitmap_assign_5"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_5 ;
        else if (IS ("bitmap_assign_5_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_5_WHOLE ;
        else if (IS ("bitmap_assign_6"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_6 ;
        else if (IS ("bitmap_assign_6b_whole")) c = GB_JIT_KERNEL_BITMAP_ASSIGN_6b_WHOLE ;
        else if (IS ("bitmap_assign_7"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_7 ;
        else if (IS ("bitmap_assign_7_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_7_WHOLE ;
        else if (IS ("bitmap_assign_8"       )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_8 ;
        else if (IS ("bitmap_assign_8_whole" )) c = GB_JIT_KERNEL_BITMAP_ASSIGN_8_WHOLE  ;

        else if (IS ("masker_phase1")) c = GB_JIT_KERNEL_MASKER_PHASE1 ;
        else if (IS ("masker_phase2")) c = GB_JIT_KERNEL_MASKER_PHASE2 ;

        else if (IS ("subref_sparse")) c = GB_JIT_KERNEL_SUBREF_SPARSE ;
        else if (IS ("subref_bitmap")) c = GB_JIT_KERNEL_BITMAP_SUBREF ;

        else if (IS ("iso_expand"   )) c = GB_JIT_KERNEL_ISO_EXPAND ;
        else if (IS ("unjumble"     )) c = GB_JIT_KERNEL_UNJUMBLE ;
        else if (IS ("convert_b2s"  )) c = GB_JIT_KERNEL_CONVERT_B2S ;
        else if (IS ("kroner"       )) c = GB_JIT_KERNEL_KRONER ;
        else if (IS ("sort"         )) c = GB_JIT_KERNEL_SORT ;

        // add CUDA PreJIT kernels here (future):
//      else if (IS ("cuda_reduce"  )) c = GB_JIT_CUDA_KERNEL_REDUCE ;
        else
        {
            // PreJIT error: kernel_name is invalid; ignore this kernel
            continue ;
        }

        #undef IS
        encoding->code = method_code ;
        encoding->major = 0 ;       // CUDA PreJIT kernels not yet supported
        encoding->minor = 0 ;
        encoding->kcode = c ;
        encoding->suffix_len = (int32_t) GB_STRLEN (suffix) ;

        //----------------------------------------------------------------------
        // get the hash of this PreJIT kernel
        //----------------------------------------------------------------------

        // Query the kernel for its hash and version number.  The hash is
        // needed now so the PreJIT kernel can be added to the hash table.

        // The type/op definitions and monoid id/term values for user-defined
        // types/ops/ monoids are ignored, because the user-defined objects
        // have not yet been created during this use of GraphBLAS (this method
        // is called by GrB_init).  These definitions are checked the first
        // time the kernel is run.

        uint64_t hash = 0 ;
        const char *ignored [5] ;
        int version [3] ;
        (void) dl_query (&hash, version, ignored, NULL, NULL, 0, 0) ;

        if (hash == 0 || hash == UINT64_MAX ||
            (version [0] != GxB_IMPLEMENTATION_MAJOR) ||
            (version [1] != GxB_IMPLEMENTATION_MINOR) ||
            (version [2] != GxB_IMPLEMENTATION_SUB))
        {
            // PreJIT error: the kernel is stale; ignore it
            continue ;
        }

        //----------------------------------------------------------------------
        // make sure this kernel is not a duplicate
        //----------------------------------------------------------------------

        int64_t k1 = -1, kk = -1 ;
        if (GB_jitifyer_lookup (hash, encoding, suffix, &k1, &kk) != NULL)
        {
            // PreJIT error: the kernel is a duplicate; ignore it
            continue ;
        }

        //----------------------------------------------------------------------
        // insert the PreJIT kernel in the hash table
        //----------------------------------------------------------------------

        if (!GB_jitifyer_insert (hash, encoding, suffix, NULL, dl_function, k))
        {
            // PreJIT error: out of memory
            GB_jit_control = GxB_JIT_PAUSE ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // uncompress all the source files into the user source folder
    //--------------------------------------------------------------------------

    return (GB_jitifyer_extract_JITpackage (GrB_SUCCESS)) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_path_256: establish a folder and its 256 subfolders
//------------------------------------------------------------------------------

bool GB_jitifyer_path_256 (char *folder)
{
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/%s",
        GB_jit_cache_path, folder) ;
    bool ok = GB_file_mkdir (GB_jit_temp) ;
    for (uint32_t bucket = 0 ; bucket <= 0xFF ; bucket++)
    { 
        snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/%s/%02x",
            GB_jit_cache_path, folder, bucket) ;
        ok = ok && GB_file_mkdir (GB_jit_temp) ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_establish_paths: make sure cache and its folders exist
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS if succesful, or GrB_OUT_OF_MEMORY if out of memory.  If
// the paths cannot be established, the JIT is disabled, and the
// error_condition is returned.  GrB_init uses this to return GrB_SUCCESS,
// since GraphBLAS can continue without the JIT.  GxB_set returns
// GrB_INVALID_VALUE to indicate that the cache path is not valid.
// If the JIT is disabled at compile time, the directories are not created and
// GrB_SUCCESS is returned (except if an out of memory condition occurs).

GrB_Info GB_jitifyer_establish_paths (GrB_Info error_condition)
{ 

    //--------------------------------------------------------------------------
    // construct the src folders
    //--------------------------------------------------------------------------

    bool ok = GB_file_mkdir (GB_jit_cache_path) ;

    // construct the c and lib and their 256 subfolders
    ok = ok && GB_jitifyer_path_256 ("c") ;
    ok = ok && GB_jitifyer_path_256 ("lib") ;

    // construct the src path and its subfolders
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/src", GB_jit_cache_path) ;
    ok = ok && GB_file_mkdir (GB_jit_temp) ;
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/src/template",
        GB_jit_cache_path) ;
    ok = ok && GB_file_mkdir (GB_jit_temp) ;
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/src/include",
        GB_jit_cache_path) ;
    ok = ok && GB_file_mkdir (GB_jit_temp) ;

    // construct the tmp path
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/tmp", GB_jit_cache_path) ;
    ok = ok && GB_file_mkdir (GB_jit_temp) ;

    //--------------------------------------------------------------------------
    // make sure the cache and source paths exist
    //--------------------------------------------------------------------------

    if (!ok)
    { 
        // JIT is disabled, or cannot determine the JIT cache path.
        // Disable loading and compiling, but continue with the rest of the
        // initializations.  The PreJIT could still be used.
        GBURBLE ("(jit: unable to access cache path, jit disabled) ") ;
        GB_jit_control = GxB_JIT_RUN ;
        GB_FREE_STUFF (GB_jit_cache_path) ;
        GB_COPY_STUFF (GB_jit_cache_path, "") ;
    }

    return (ok ? GrB_SUCCESS : error_condition) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_extract_JITpackage: extract the GraphBLAS source
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS if successful, GrB_OUT_OF_MEMORY if out of memory, or
// error_condition if the files cannot be written to the cache folder for any
// reason.  If the JIT is disabled at compile time, this method does nothing.

GrB_Info GB_jitifyer_extract_JITpackage (GrB_Info error_condition)
{ 

    #ifndef NJIT

    //--------------------------------------------------------------------------
    // check the version number in src/GraphBLAS.h
    //--------------------------------------------------------------------------

    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/src/GraphBLAS.h",
        GB_jit_cache_path) ;
    FILE *fp_graphblas = fopen (GB_jit_temp, "r") ;
    if (fp_graphblas != NULL)
    { 
        int v1 = -1, v2 = -1, v3 = -1 ;
        int r = fscanf (fp_graphblas, "// SuiteSparse:GraphBLAS %d.%d.%d",
            &v1, &v2, &v3) ;
        fclose (fp_graphblas) ;
        if (r == 3 &&
            v1 == GxB_IMPLEMENTATION_MAJOR &&
            v2 == GxB_IMPLEMENTATION_MINOR &&
            v3 == GxB_IMPLEMENTATION_SUB)
        { 
            // looks fine; assume the rest of the source is fine
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // get the JITpackage
    //--------------------------------------------------------------------------

    int GB_JITpackage_nfiles = GB_JITpackage_nfiles_get ( ) ;
    GB_JITpackage_index_struct *GB_JITpackage_index =
        GB_JITpackage_index_get ( ) ;

    //--------------------------------------------------------------------------
    // allocate workspace for the largest uncompressed file
    //--------------------------------------------------------------------------

    size_t dst_size = 0 ;
    for (int k = 0 ; k < GB_JITpackage_nfiles ; k++)
    { 
        size_t uncompressed_size = GB_JITpackage_index [k].uncompressed_size ;
        dst_size = GB_IMAX (dst_size, uncompressed_size) ;
    }

    uint8_t *dst ;
    GB_MALLOC_PERSISTENT (dst, (dst_size+2) * sizeof(uint8_t)) ;
    if (dst == NULL)
    {
        // JITPackage error: out of memory; disable the JIT
        GB_jit_control = GxB_JIT_PAUSE ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // uncompress each file into the src folder
    //--------------------------------------------------------------------------

    bool ok = true ;
    for (int k = 0 ; k < GB_JITpackage_nfiles ; k++)
    { 
        // uncompress the blob
        uint8_t *src = GB_JITpackage_index [k].blob ;
        size_t src_size = GB_JITpackage_index [k].compressed_size ;
        size_t u = ZSTD_decompress (dst, dst_size, src, src_size) ;
        if (u != GB_JITpackage_index [k].uncompressed_size)
        {
            // JITPackage error: blob is invalid
            ok = false ;
            break ;
        }
        // construct the filename
        snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/src/%s",
            GB_jit_cache_path, GB_JITpackage_index [k].filename) ;
        // open the file
        FILE *fp_src = fopen (GB_jit_temp, "w") ;
        if (fp_src == NULL)
        {
            // JITPackage error: file cannot be created
            ok = false ;
            break ;
        }
        // write the uncompressed blob to the file
        size_t nwritten = fwrite (dst, sizeof (uint8_t), u, fp_src) ;
        fclose (fp_src) ;
        if (nwritten != u)
        {
            // JITPackage error: file is invalid
            ok = false ;
            break ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    GB_FREE_PERSISTENT (dst) ;
    if (!ok)
    {
        // JITPackage error: disable the JIT
        GBURBLE ("(jit: unable to write to source cache, jit disabled) ") ;
        GB_jit_control = GxB_JIT_RUN ;
        return (error_condition) ;
    }
    #endif

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_control: get the JIT control
//------------------------------------------------------------------------------

int GB_jitifyer_get_control (void)
{
    int control ;
    GB_OPENMP_LOCK_SET (1)
    { 
        control = GB_jit_control ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (control) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_control: set the JIT control
//------------------------------------------------------------------------------

void GB_jitifyer_set_control (int control)
{ 
    GB_OPENMP_LOCK_SET (1)
    {
        control = GB_IMAX (control, (int) GxB_JIT_OFF) ;
        #ifndef NJIT
        // The full JIT is available.
        control = GB_IMIN (control, (int) GxB_JIT_ON) ;
        #else
        // The JIT is restricted; only OFF, PAUSE, and RUN settings can be
        // used.  No JIT kernels can be loaded or compiled.
        control = GB_IMIN (control, (int) GxB_JIT_RUN) ;
        #endif
        GB_jit_control = control ;
        if (GB_jit_control == GxB_JIT_OFF)
        { 
            // free all loaded JIT kernels but do not free the JIT hash table,
            // and do not free the PreJIT kernels
            GB_jitifyer_table_free (false) ;
        }
    }
    GB_OPENMP_LOCK_UNSET (1)
}

//------------------------------------------------------------------------------
// GB_jitifyer_alloc_space: allocate temporary workspace for the JIT
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS or GrB_OUT_OF_MEMORY.

GrB_Info GB_jitifyer_alloc_space (void)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (GB_jit_C_flags == NULL ||
        GB_jit_C_link_flags == NULL ||
        GB_jit_C_libraries == NULL ||
        GB_jit_C_cmake_libs == NULL ||
        GB_jit_C_compiler == NULL ||
        GB_jit_cache_path == NULL)
    {
        // JIT error: out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // free the old GB_jit_temp and allocate it at the proper size
    //--------------------------------------------------------------------------

    GB_FREE_STUFF (GB_jit_temp) ;
    size_t len =
        2 * GB_jit_C_compiler_allocated +
        2 * GB_jit_C_flags_allocated +
        GB_jit_C_link_flags_allocated +
        strlen (GB_OMP_INC) +
        5 * GB_jit_cache_path_allocated + 7 * GB_KLEN +
        GB_jit_C_libraries_allocated +
        GB_jit_C_cmake_libs_allocated +
        GB_jit_error_log_allocated +
        300 ;
    GB_MALLOC_STUFF (GB_jit_temp, len) ;

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_cache_path: return the current cache path
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_cache_path (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_cache_path ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_cache_path: set a new cache path
//------------------------------------------------------------------------------

// This method is only used by GxB_set.  It returns GrB_SUCCESS if successful,
// GrB_OUT_OF_MEMORY if out of memory, GrB_NULL_POINTER if the requested path
// is a NULL string, or GrB_INVALID_VALUE if any file I/O error occurs.  The
// latter indicates that the requested path is not valid.

GrB_Info GB_jitifyer_set_cache_path (const char *new_cache_path)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_cache_path == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the cache path in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_cache_path_worker (new_cache_path) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_cache_path_worker: set cache path in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_cache_path_worker (const char *new_cache_path)
{ 
    // free the old the cache path
    GB_FREE_STUFF (GB_jit_cache_path) ;
    // allocate the new GB_jit_cache_path
    GB_COPY_STUFF (GB_jit_cache_path, new_cache_path) ;
    // sanitize the cache path
    GB_jitifyer_sanitize (GB_jit_cache_path, GB_jit_cache_path_allocated) ;
    // allocate workspace
    OK (GB_jitifyer_alloc_space ( )) ;
    // set the src path and make sure cache and src paths are accessible
    OK (GB_jitifyer_establish_paths (GrB_INVALID_VALUE)) ;
    // uncompress all the source files into the user source folder
    return (GB_jitifyer_extract_JITpackage (GrB_INVALID_VALUE)) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_error_log: return the current log file
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_error_log (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_error_log ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_error_log: set a new log file
//------------------------------------------------------------------------------

// If the new_error_log is NULL or the empty string, stderr is not redirected to
// a log file.

GrB_Info GB_jitifyer_set_error_log (const char *new_error_log)
{ 

    //--------------------------------------------------------------------------
    // set the log file in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_error_log_worker
            ((new_error_log == NULL) ? "" : new_error_log) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_error_log_worker: set log file in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_error_log_worker (const char *new_error_log)
{ 
    // free the old log file
    GB_FREE_STUFF (GB_jit_error_log) ;
    // allocate the new GB_jit_error_log
    GB_COPY_STUFF (GB_jit_error_log, new_error_log) ;
    // sanitize the error log
    GB_jitifyer_sanitize (GB_jit_error_log, GB_jit_error_log_allocated) ;
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_compiler: return the current C compiler
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_compiler (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_compiler ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_compiler: set a new C compiler
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_compiler (const char *new_C_compiler)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_compiler == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C compiler in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_compiler_worker (new_C_compiler) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_compiler_worker: set C compiler in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_compiler_worker (const char *new_C_compiler)
{ 
    // free the old C compiler string
    GB_FREE_STUFF (GB_jit_C_compiler) ;
    // allocate the new GB_jit_C_compiler
    GB_COPY_STUFF (GB_jit_C_compiler, new_C_compiler) ;
    // allocate workspace
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_flags: return the current C flags
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_flags (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_flags ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_flags: set new C flags
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_flags (const char *new_C_flags)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_flags == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C flags in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_flags_worker (new_C_flags) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_flags_worker: set C flags in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_flags_worker (const char *new_C_flags)
{ 
    // free the old C flag string
    GB_FREE_STUFF (GB_jit_C_flags) ;
    // allocate the new GB_jit_C_flags
    GB_COPY_STUFF (GB_jit_C_flags, new_C_flags) ;
    // allocate workspace
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_link_flags: return the current C link flags
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_link_flags (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_link_flags ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_link_flags: set new C link flags
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_link_flags (const char *new_C_link_flags)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_link_flags == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C link flags in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_link_flags_worker (new_C_link_flags) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_link_flags_worker: set C link flags in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_link_flags_worker (const char *new_C_link_flags)
{ 
    // free the old C link flags string
    GB_FREE_STUFF (GB_jit_C_link_flags) ;
    // allocate the new GB_jit_C_link_flags
    GB_COPY_STUFF (GB_jit_C_link_flags, new_C_link_flags) ;
    // allocate workspace
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_libraries: return the current C libraries
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_libraries (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_libraries ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_libraries: set new C libraries
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_libraries (const char *new_C_libraries)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_libraries == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C libraries in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_libraries_worker (new_C_libraries) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_libraries_worker: set C libraries in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_libraries_worker (const char *new_C_libraries)
{ 
    // free the old C libraries string
    GB_FREE_STUFF (GB_jit_C_libraries) ;
    // allocate the new GB_jit_C_libraries
    GB_COPY_STUFF (GB_jit_C_libraries, new_C_libraries) ;
    // allocate workspace
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_use_cmake: return true/false if cmake is in use
//------------------------------------------------------------------------------

bool GB_jitifyer_get_use_cmake (void)
{ 
    bool use_cmake ;
    GB_OPENMP_LOCK_SET (1)
    {
        use_cmake = GB_jit_use_cmake ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (use_cmake) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_use_cmake: set controls true/false to use cmake
//------------------------------------------------------------------------------

void GB_jitifyer_set_use_cmake (bool use_cmake)
{ 
    GB_OPENMP_LOCK_SET (1)
    {
        #if defined (_MSC_VER)
        // Windows requires cmake
        GB_jit_use_cmake = true ;
        #elif defined (__MINGW32__)
        // MINGW requires direct compile
        GB_jit_use_cmake = false ;
        #else
        // all other platforms have the option to use cmake or a direct compile
        GB_jit_use_cmake = use_cmake ;
        #endif
    }
    GB_OPENMP_LOCK_UNSET (1)
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_cmake_libs: return the current cmake libs
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_cmake_libs (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_cmake_libs ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_cmake_libs: set new cmake libs
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_cmake_libs (const char *new_cmake_libs)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_cmake_libs == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the cmake libs in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_cmake_libs_worker (new_cmake_libs) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_cmake_libs_worker: set cmake libs in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_cmake_libs_worker (const char *new_cmake_libs)
{ 
    // free the old C_cmake_libs string
    GB_FREE_STUFF (GB_jit_C_cmake_libs) ;
    // allocate the new GB_jit_C_cmake_libs
    GB_COPY_STUFF (GB_jit_C_cmake_libs, new_cmake_libs) ;
    // allocate workspace
    return (GB_jitifyer_alloc_space ( )) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_C_preface: return the current C preface
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_C_preface (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_C_preface ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_preface: set new C preface
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_preface (const char *new_C_preface)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_C_preface == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C preface in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_C_preface_worker (new_C_preface) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_C_preface_worker: set C preface in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_C_preface_worker (const char *new_C_preface)
{ 
    // free the old strings that depend on the C preface
    GB_FREE_STUFF (GB_jit_C_preface) ;
    // allocate the new GB_jit_C_preface
    GB_COPY_STUFF (GB_jit_C_preface, new_C_preface) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_get_CUDA_preface: return the current C preface
//------------------------------------------------------------------------------

const char *GB_jitifyer_get_CUDA_preface (void)
{ 
    const char *s ;
    GB_OPENMP_LOCK_SET (1)
    {
        s = GB_jit_CUDA_preface ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (s) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_CUDA_preface: set new C preface
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_CUDA_preface (const char *new_CUDA_preface)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (new_CUDA_preface == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // set the C preface in a critical section
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_OPENMP_LOCK_SET (1)
    {
        info = GB_jitifyer_set_CUDA_preface_worker (new_CUDA_preface) ;
    }
    GB_OPENMP_LOCK_UNSET (1)
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_set_CUDA_preface_worker: set C preface in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_set_CUDA_preface_worker (const char *new_CUDA_preface)
{ 
    // free the old strings that depend on the C preface
    GB_FREE_STUFF (GB_jit_CUDA_preface) ;
    // allocate the new GB_jit_CUDA_preface
    GB_COPY_STUFF (GB_jit_CUDA_preface, new_CUDA_preface) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_query: check if the type/op/monoid definitions match
//------------------------------------------------------------------------------

// Returns true if type/op/monoid/etc definitions match, false otherwise.

bool GB_jitifyer_query
(
    GB_jit_query_func dl_query,
    const bool builtin,         // true if method is all builtin
    uint64_t hash,              // hash code for the kernel
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{ 

    //--------------------------------------------------------------------------
    // get the terms to query
    //--------------------------------------------------------------------------

    int version [3] ;
    const char *ldef [5] ;
    size_t zsize = 0 ;
    size_t tsize = 0 ;
    void *id = NULL ;
    void *term = NULL ;

    GB_Operator op1 = NULL, op2 = NULL ;
    if (semiring != NULL)
    { 
        monoid = semiring->add ;
        op1 = (GB_Operator) monoid->op ;
        op2 = (GB_Operator) semiring->multiply ;
    }
    else if (monoid != NULL)
    { 
        op1 = (GB_Operator) monoid->op ;
    }
    else
    { 
        // op may be NULL, if this is a user_type kernel
        op1 = op ;
    }

    if (monoid != NULL && monoid->hash != 0)
    { 
        // compare the user-defined identity and terminal values
        zsize = monoid->op->ztype->size ;
        tsize = (monoid->terminal == NULL) ? 0 : zsize ;
        id = monoid->identity ;
        term = monoid->terminal ;
    }

    //--------------------------------------------------------------------------
    // query the JIT kernel for its definitions
    //--------------------------------------------------------------------------

    uint64_t hash2 = 0 ;
    bool ok = dl_query (&hash2, version, ldef, id, term, zsize, tsize) ;
    ok = ok && (version [0] == GxB_IMPLEMENTATION_MAJOR) &&
               (version [1] == GxB_IMPLEMENTATION_MINOR) &&
               (version [2] == GxB_IMPLEMENTATION_SUB) &&
               (hash == hash2) ;

    //--------------------------------------------------------------------------
    // compare current definitions with the ones in the JIT kernel
    //--------------------------------------------------------------------------

    char *defn [5] ;
    defn [0] = (builtin || op1   == NULL) ? NULL : op1->defn ;
    defn [1] = (builtin || op2   == NULL) ? NULL : op2->defn ;
    defn [2] = (builtin || type1 == NULL) ? NULL : type1->defn ;
    defn [3] = (builtin || type2 == NULL) ? NULL : type2->defn ;
    defn [4] = (builtin || type3 == NULL) ? NULL : type3->defn ;

    for (int k = 0 ; k < 5 ; k++)
    { 
        // ensure the definition hasn't changed
        ok = ok && (strcmp (
            ((defn [k] == NULL) ? "" : defn [k]),
            ((ldef [k] == NULL) ? "" : ldef [k])) == 0) ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_load: load a JIT kernel, compiling it if needed
//------------------------------------------------------------------------------

// Returns GrB_SUCCESS if kernel is found (already loaded, or just now loaded,
// or just now compiled and loaded).

// Returns GrB_NO_VALUE only if the kernel intentionally cannot be loaded, run,
// or compiled.  This tells the caller that a generic method must be used.

// Returns GxB_JIT_ERROR if the JIT should succeed, but fails.

GrB_Info GB_jitifyer_load
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    #ifdef GBMATLAB
    if (GB_Global_hack_get (3) != 0)
    {
        // the JIT can be disabled for testing, to test error handling
        GBURBLE ("(jit: test error handling) ") ;
        return (GrB_NOT_IMPLEMENTED) ; // only to test error handling in MATLAB
    }
    #endif

    GrB_Info info ;
    if (hash == UINT64_MAX)
    { 
        // The kernel may not be compiled; it does not have a valid definition.
        // This is not a JIT failure.  It is an expected error if the strings
        // (name & defn) are NULL, so always fallback to the generic case.
        GBURBLE ("(jit: undefined) ") ;
        return (GrB_NO_VALUE) ;     // no hash code (no strings given)
    }

    if ((GB_jit_control == GxB_JIT_OFF) || (GB_jit_control == GxB_JIT_PAUSE))
    { 
        // The JIT control has disabled all JIT kernels.  Punt to generic.
        // This is not a JIT failure.
        return (GrB_NO_VALUE) ;     // JIT is off or paused
    }

    //--------------------------------------------------------------------------
    // handle the GxB_JIT_RUN case: critical section not required
    //--------------------------------------------------------------------------

    if ((GB_jit_control == GxB_JIT_RUN) &&
        (family != GB_jit_user_op_family) &&
        (family != GB_jit_user_type_family))
    {

        //----------------------------------------------------------------------
        // look up the kernel in the hash table
        //----------------------------------------------------------------------

        int64_t k1 = -1, kk = -1 ;
        (*dl_function) = GB_jitifyer_lookup (hash, encoding, suffix, &k1, &kk) ;
        if (k1 >= 0)
        { 
            // an unchecked PreJIT kernel; check it inside critical section
        }
        else if ((*dl_function) != NULL)
        { 
            // found the kernel in the hash table
            return (GrB_SUCCESS) ;
        }
        else
        { 
            // No kernels may be loaded or compiled, but existing kernels
            // already loaded may be run (handled above if dl_function was
            // found).  This kernel was not loaded, so punt to generic.
            // This is not a JIT failure since the JIT control is already
            // set to 'run', and the kernel is not already loaded.  So always
            // fallback to the generic kernel.
            return (GrB_NO_VALUE) ; // JIT set to 'run'; but kernel not loaded
        }
    }

    //--------------------------------------------------------------------------
    // do the rest inside a critical section
    //--------------------------------------------------------------------------

    GB_OPENMP_LOCK_SET (1)
    { 
        info = GB_jitifyer_load2_worker (dl_function, family, kname, hash,
            encoding, suffix, semiring, monoid, op, type1, type2, type3) ;
    }
    GB_OPENMP_LOCK_UNSET (1)

    return (info) ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_load2_worker: do the work for GB_jitifyer_load in a critical section
//------------------------------------------------------------------------------

GrB_Info GB_jitifyer_load2_worker
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{

    //--------------------------------------------------------------------------
    // look up the kernel in the hash table
    //--------------------------------------------------------------------------

    // e->prejit_index >= 0 denotes an unchecked prejit kernel:
    #define GB_PREJIT_CHECKED(i) (-(i)-2)
    // ensure that e->prejit_index is >= 0, to denote that it's unchecked:
    #define GB_PREJIT_UNCHECKED(i) (((i) < 0) ? GB_PREJIT_CHECKED(i) : (i))

    int64_t k1 = -1, kk = -1 ;
    (*dl_function) = GB_jitifyer_lookup (hash, encoding, suffix, &k1, &kk) ;
    if ((*dl_function) != NULL)
    { 
        // found the kernel in the hash table
        GB_jit_entry *e = &(GB_jit_table [kk]) ;
        if (k1 >= 0)
        {
            // unchecked PreJIT kernel; check it now
            void **Kernels = NULL ;
            void **Queries = NULL ;
            char **Names = NULL ;
            int32_t nkernels = 0 ;
            GB_prejit (&nkernels, &Kernels, &Queries, &Names) ;
//          GB_jit_query_func dl_query = (GB_jit_query_func) Queries [k1] ;
            GB_jit_query_func dl_query = GB_jitifyer_get_query (Queries [k1]) ;
            bool builtin = (encoding->suffix_len == 0) ;
            bool ok = GB_jitifyer_query (dl_query, builtin, hash, semiring,
                monoid, op, type1, type2, type3) ;
            if (ok)
            { 
                // PreJIT kernel is fine; flag it as checked by marking
                // its prejit_index as negative.
                GBURBLE ("(prejit: ok) ") ;
                e->prejit_index = GB_PREJIT_CHECKED (k1) ;
                return (GrB_SUCCESS) ;
            }
            else
            { 
                // remove the PreJIT kernel from the hash table; do not return.
                // Instead, keep going and compile a JIT kernel.
                GBURBLE ("(prejit: disabled) ") ;
                GB_jitifyer_entry_free (e) ;
            }
        }
        else if (family == GB_jit_user_op_family)
        {
            // user-defined operator; check it now
//          GB_user_op_f GB_user_op = (GB_user_op_f) (*dl_function) ;
            GB_user_op_f GB_user_op = GB_jitifyer_get_user_op (*dl_function) ;

            void *ignore ;
            char *defn ;
            GB_user_op (&ignore, &defn) ;
            if (strcmp (defn, op->defn) == 0)
            { 
                return (GrB_SUCCESS) ;
            }
            else
            { 
                // the op has changed; need to re-JIT the kernel; do not return.
                // Instead, keep going and compile a JIT kernel.
                GBURBLE ("(jit: op changed) ") ;
                GB_jitifyer_entry_free (e) ;
            }
        }
        else if (family == GB_jit_user_type_family)
        {
            // user-defined type; check it now
//          GB_user_type_f GB_user_type = (GB_user_type_f) (*dl_function) ;
            GB_user_type_f GB_user_type =
                GB_jitifyer_get_user_type (*dl_function) ;
    
            size_t ignore ;
            char *defn ;
            GB_user_type (&ignore, &defn) ;
            if (strcmp (defn, type1->defn) == 0)
            { 
                return (GrB_SUCCESS) ;
            }
            else
            { 
                // type has changed; need to re-JIT the kernel; do not return.
                // Instead, keep going and compile a JIT kernel.
                GBURBLE ("(jit: type changed) ") ;
                GB_jitifyer_entry_free (e) ;
            }
        }
        else
        { 
            // JIT kernel, or checked PreJIT kernel
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // quick return if not in the hash table and load/compile is disabled
    //--------------------------------------------------------------------------

    #ifndef NJIT
    if (GB_jit_control <= GxB_JIT_RUN)
    #endif
    { 
        // No kernels may be loaded or compiled, but existing kernels already
        // loaded may be run (handled above if dl_function was found).  This
        // kernel was not loaded, so punt to generic.
        // This is not a JIT failure since the JIT control is already
        // set to 'run', and the kernel is not already loaded.  So always
        // fallback to the generic kernel.
        return (GrB_NO_VALUE) ;     // JIT set to 'run'; but kernel not loaded
    }

    //--------------------------------------------------------------------------
    // construct the kernel name
    //--------------------------------------------------------------------------

    #ifndef NJIT
    GB_Operator op1 = NULL ;
    GB_Operator op2 = NULL ;
    int method_code_digits = 0 ;

    switch (family)
    {
        case GB_jit_apply_family  : 
            op1 = op ;
            method_code_digits = 12 ;
            break ;

        case GB_jit_assign_family : 
            op1 = op ;
            method_code_digits = 16 ;
            break ;

        case GB_jit_build_family  : 
            op1 = op ;
            method_code_digits = 8 ;
            break ;

        case GB_jit_ewise_family  : 
            op1 = op ;
            method_code_digits = 15 ;
            break ;

        case GB_jit_mxm_family    : 
            monoid = semiring->add ;
            op1 = (GB_Operator) semiring->add->op ;
            op2 = (GB_Operator) semiring->multiply ;
            method_code_digits = 16 ;
            break ;

        case GB_jit_reduce_family : 
            op1 = (GB_Operator) monoid->op ;
            method_code_digits = 5 ;
            break ;

        case GB_jit_select_family : 
            op1 = op ;
            method_code_digits = 9 ;
            break ;

        case GB_jit_user_type_family : 
            method_code_digits = 1 ;
            break ;

        case GB_jit_user_op_family : 
            method_code_digits = 1 ;
            op1 = op ;
            break ;

        case GB_jit_masker_family  : 
            method_code_digits = 8 ;
            break ;

        case GB_jit_subref_family  : 
            method_code_digits = 7 ;
            break ;

        case GB_jit_sort_family  : 
            method_code_digits = 5 ;
            break ;

        default: ;
    }

    char kernel_name [GB_KLEN] ;
    GB_macrofy_name (kernel_name, "GB_jit", kname, method_code_digits,
        encoding, suffix) ;

    //--------------------------------------------------------------------------
    // load the kernel, compiling it if needed
    //--------------------------------------------------------------------------

    GrB_Info info = GB_jitifyer_load_worker (dl_function, kernel_name, family,
        kname, hash, encoding, suffix, semiring, monoid, op, op1, op2,
        type1, type2, type3) ;

    return (info) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_load_worker: load/compile a kernel
//------------------------------------------------------------------------------

// This work is done inside a critical section for this process.

GrB_Info GB_jitifyer_load_worker
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    char *kernel_name,          // kernel file name (excluding the path)
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GB_Operator op1,
    GB_Operator op2,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
)
{

    #ifndef NJIT

    //--------------------------------------------------------------------------
    // try to load the lib*.so from the user's library folder
    //--------------------------------------------------------------------------

    uint32_t bucket = hash & 0xFF ;
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/lib/%02x/%s%s%s",
        GB_jit_cache_path, bucket, GB_LIB_PREFIX, kernel_name, GB_LIB_SUFFIX) ;
    void *dl_handle = GB_file_dlopen (GB_jit_temp) ;
    GB_jit_kcode kcode = encoding->kcode ;

    //--------------------------------------------------------------------------
    // check if the kernel was found, but needs to be compiled anyway
    //--------------------------------------------------------------------------

    if (dl_handle != NULL)
    { 
        // library is loaded but make sure the defn match
//      GB_jit_query_func dl_query = (GB_jit_query_func)
//          GB_file_dlsym (dl_handle, "GB_jit_query") ;
        GB_jit_query_func dl_query = GB_jitifyer_get_query (
            GB_file_dlsym (dl_handle, "GB_jit_query")) ;
        bool ok = (dl_query != NULL) ;
        if (ok)
        { 
            bool builtin = (encoding->suffix_len == 0) ;
            ok = GB_jitifyer_query (dl_query, builtin, hash, semiring,
                monoid, op, type1, type2, type3) ;
        }
        if (!ok)
        { 
            // library is loaded but needs to change, so close it
            GB_file_dlclose (dl_handle) ; dl_handle = NULL ;
            // remove the library itself so it doesn't cause the error again
            remove (GB_jit_temp) ;
            GBURBLE ("(jit: loaded but must recompile) ") ;
        }
    }

    //--------------------------------------------------------------------------
    // create and compile source file, if needed
    //--------------------------------------------------------------------------

    if (dl_handle == NULL)
    { 

        //----------------------------------------------------------------------
        // quick return if the JIT is not permitted to compile new kernels
        //----------------------------------------------------------------------

        if (GB_jit_control < GxB_JIT_ON)
        { 
            // No new kernels may be compiled, so punt to generic.  This is not
            // a JIT failure.  It is an expected condition because of the JIT
            // control, so always allow a fallback to the generic kernel.
            GBURBLE ("(jit: not compiled) ") ;
            return (GrB_NO_VALUE) ; // JIT not on; compiler disabled
        }

        //----------------------------------------------------------------------
        // create the source, compile it, and load it
        //----------------------------------------------------------------------

        GBURBLE ("(jit: compile and load) ") ;
        const char *kernel_filetype =
            (kcode < GB_JIT_CUDA_KERNEL) ? "c" : "cu" ;

        // create (or recreate) the kernel source, compile it, and load it
        snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/c/%02x/%s.%s",
            GB_jit_cache_path, bucket, kernel_name, kernel_filetype) ;
        FILE *fp = fopen (GB_jit_temp, "w") ;
        if (fp != NULL)
        { 
            // create the preface
            GB_macrofy_preface (fp, kernel_name,
                GB_jit_C_preface, GB_jit_CUDA_preface, kcode,
                encoding->major, encoding->minor) ;
            // macrofy the kernel operators, types, and matrix formats
            GB_macrofy_family (fp, family, encoding->code, encoding->kcode,
                semiring, monoid, op, type1, type2, type3) ;
            // #include the kernel, renaming it for the PreJIT
            fprintf (fp, "#ifndef GB_JIT_RUNTIME\n"
                         "#define GB_jit_kernel %s\n"
                         "#define GB_jit_query  %s_query\n"
                         "#endif\n"
                         "#include \"template/GB_jit_kernel_%s.%s\"\n",
                         kernel_name, kernel_name, kname,
                         kernel_filetype) ;

            // macrofy the query function
            bool builtin = (encoding->suffix_len == 0) ;
            GB_macrofy_query (fp, builtin, monoid, op1, op2, type1, type2,
                type3, hash, kcode) ;
            fclose (fp) ;
        }

        // if the source file was not created above, the compilation will
        // gracefully fail.

        // compile the kernel to get the lib*.so file
        if (kcode >= GB_JIT_CUDA_KERNEL)
        {
            // use NVCC to directly compile the CUDA kernel
            GB_jitifyer_nvcc_compile (kernel_name, bucket,
                encoding->major, encoding->minor) ;
        }
        else if (GB_jit_use_cmake)
        { 
            // use cmake to compile the CPU kernel
            GB_jitifyer_cmake_compile (kernel_name, hash) ;
        }
        else
        { 
            // use the compiler to directly compile the CPU kernel
            GB_jitifyer_direct_compile (kernel_name, bucket) ;
        }

        // load the kernel from the lib*.so file
        snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/lib/%02x/%s%s%s",
            GB_jit_cache_path, bucket, GB_LIB_PREFIX, kernel_name,
            GB_LIB_SUFFIX) ;
        dl_handle = GB_file_dlopen (GB_jit_temp) ;

        //----------------------------------------------------------------------
        // handle any error conditions
        //----------------------------------------------------------------------

        if (dl_handle == NULL)
        { 
            // unable to create the kernel source or open lib*.so file
            // disable the JIT to avoid repeated compilation errors
            GB_jit_control = GxB_JIT_LOAD ;
            // remove the compiled library
            remove (GB_jit_temp) ;
            GBURBLE ("\n(jit failure: compiler error; compilation disabled)\n");
            return (GxB_JIT_ERROR) ;
        }

    }
    else
    { 
        if (kcode >= GB_JIT_CUDA_KERNEL)
        {
            GBURBLE ("(jit: cuda load) ") ;
        }
        else
        {
            GBURBLE ("(jit: cpu load) ") ;
        }
    }

    //--------------------------------------------------------------------------
    // get the GB_jit_kernel function pointer
    //--------------------------------------------------------------------------

    (*dl_function) = GB_file_dlsym (dl_handle, "GB_jit_kernel") ;
    if ((*dl_function) == NULL)
    {
        // JIT error: dlsym unable to find GB_jit_kernel: punt to generic
        GB_file_dlclose (dl_handle) ; dl_handle = NULL ;
        // disable the JIT to avoid repeated loading errors
        GB_jit_control = GxB_JIT_RUN ;
        // remove the compiled library
        remove (GB_jit_temp) ;
        GBURBLE ("\n(jit failure: load error; compilation disabled)\n") ;
        return (GxB_JIT_ERROR) ;
    }

    // insert the new kernel into the hash table
    if (!GB_jitifyer_insert (hash, encoding, suffix, dl_handle, (*dl_function),
        -1))
    {
        // JIT error: unable to add kernel to hash table
        GB_file_dlclose (dl_handle) ; dl_handle = NULL ;
        // disable the JIT to avoid repeated errors
        GB_jit_control = GxB_JIT_PAUSE ;
        // remove the compiled library
        remove (GB_jit_temp) ;
        // report the error: punt to generic or panic
        return (GrB_OUT_OF_MEMORY) ;
    }

    return (GrB_SUCCESS) ;
    #else
    (*dl_function) = NULL ;
    return (GrB_INVALID_VALUE) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_lookup:  find a jit entry in the hash table
//------------------------------------------------------------------------------

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_encoding (encoding) ;
    GB_jit_encoding *encoding,
    const char *suffix,
    // output
    int64_t *k1,            // location of unchecked kernel in PreJIT table
    int64_t *kk             // location of hash entry in hash table
)
{

    (*k1) = -1 ;

    if (GB_jit_table == NULL)
    { 
        // no table yet so it isn't present
        return (NULL) ;
    }

    uint32_t suffix_len = encoding->suffix_len ;
    bool builtin = (bool) (suffix_len == 0) ;

    // look up the entry in the hash table
    for (uint64_t k = hash ; ; k++)
    {
        k = k & GB_jit_table_bits ;
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_function == NULL)
        { 
            // found an empty entry, so the entry is not in the table
            return (NULL) ;
        }
        else if (e->hash == hash &&
            e->encoding.code == encoding->code &&
            e->encoding.kcode == encoding->kcode &&
            e->encoding.suffix_len == suffix_len &&
            (builtin || (memcmp (e->suffix, suffix, suffix_len) == 0)))
        { 
            // found the right entry: return the corresponding dl_function
            int64_t my_k1 ;
            GB_ATOMIC_READ
            my_k1 = e->prejit_index ;   // >= 0: unchecked JIT kernel
            (*k1) = my_k1 ;
            (*kk) = k ;
            return (e->dl_function) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_insert:  insert a jit entry in the hash table
//------------------------------------------------------------------------------

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    const char *suffix,         // suffix for user-defined types/operators
    void *dl_handle,            // library handle from GB_file_dlopen;
                                // NULL for PreJIT
    void *dl_function,          // function handle from GB_file_dlsym
    int32_t prejit_index        // index into PreJIT table; =>0 if unchecked.
)
{

    size_t siz = 0 ;
    ASSERT_TABLE_OK ;

    //--------------------------------------------------------------------------
    // ensure the hash table is large enough
    //--------------------------------------------------------------------------

    if (GB_jit_table == NULL)
    { 

        //----------------------------------------------------------------------
        // allocate the initial hash table
        //----------------------------------------------------------------------

        siz = GB_JITIFIER_INITIAL_SIZE * sizeof (struct GB_jit_entry_struct) ;
        GB_MALLOC_PERSISTENT (GB_jit_table, siz) ;
        if (GB_jit_table == NULL)
        {
            // JIT error: out of memory
            return (false) ;
        }
        memset (GB_jit_table, 0, siz) ;
        GB_jit_table_size = GB_JITIFIER_INITIAL_SIZE ;
        GB_jit_table_bits = GB_JITIFIER_INITIAL_SIZE - 1 ;
        GB_jit_table_allocated = siz ;

    }
    else if (4 * GB_jit_table_populated >= GB_jit_table_size)
    {

        //----------------------------------------------------------------------
        // expand the existing hash table by a factor of 4 and rehash
        //----------------------------------------------------------------------

        ASSERT_TABLE_OK ;
        // create a new table that is four times the size
        int64_t new_size = 4 * GB_jit_table_size ;
        int64_t new_bits = new_size - 1 ;
        siz = new_size * sizeof (struct GB_jit_entry_struct) ;
        GB_jit_entry *new_table ;
        GB_MALLOC_PERSISTENT (new_table, siz) ;
        if (new_table == NULL)
        {
            // JIT error: out of memory; leave the existing table as-is
            return (false) ;
        }

        // rehash into the new table
        memset (new_table, 0, siz) ;
        for (uint64_t k = 0 ; k < GB_jit_table_size ; k++)
        {
            if (GB_jit_table [k].dl_function != NULL)
            { 
                // rehash the entry to the larger hash table
                uint64_t hash = GB_jit_table [k].hash ;
                for (uint64_t knew = hash ; ; knew++)
                {
                    knew = knew & new_bits ;
                    GB_jit_entry *e = &(new_table [knew]) ;
                    if (e->dl_function == NULL)
                    { 
                        // found an empty slot in the new table
                        new_table [knew] = GB_jit_table [k] ;
                        break ;
                    }
                }
            }
        }

        // free the old table
        GB_FREE_STUFF (GB_jit_table) ;

        // use the new table
        GB_jit_table = new_table ;
        GB_jit_table_size = new_size ;
        GB_jit_table_bits = new_bits ;
        GB_jit_table_allocated = siz ;
        ASSERT_TABLE_OK ;
    }

    //--------------------------------------------------------------------------
    // insert the jit entry in the hash table
    //--------------------------------------------------------------------------

    uint64_t suffix_len = (uint64_t) (encoding->suffix_len) ;
    bool builtin = (bool) (suffix_len == 0) ;
    ASSERT_TABLE_OK ;

    for (uint64_t k = hash ; ; k++)
    {
        k = k & GB_jit_table_bits ;
        GB_jit_entry *e = &(GB_jit_table [k]) ;
        if (e->dl_function == NULL)
        { 
            // found an empty slot
            e->suffix = NULL ;
            if (!builtin)
            { 
                // allocate the suffix if the kernel is not builtin
                GB_MALLOC_PERSISTENT (e->suffix, suffix_len+2) ;
                if (e->suffix == NULL)
                {
                    // JIT error: out of memory
                    return (false) ;
                }
                strncpy (e->suffix, suffix, suffix_len+1) ;
            }
            e->hash = hash ;
            memcpy (&(e->encoding), encoding, sizeof (GB_jit_encoding)) ;
            e->dl_handle = dl_handle ;              // NULL for PreJIT
            e->dl_function = dl_function ;
            GB_jit_table_populated++ ;
            e->prejit_index = prejit_index ;        // -1 for JIT kernels
            ASSERT_TABLE_OK ;
            return (true) ;
        }
        // otherwise, keep looking
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_entry_free: free a single JIT hash table entry
//------------------------------------------------------------------------------

void GB_jitifyer_entry_free (GB_jit_entry *e)
{
    e->dl_function = NULL ;
    GB_jit_table_populated-- ;
    GB_FREE_PERSISTENT (e->suffix) ;
    // unload the dl library
    if (e->dl_handle != NULL)
    { 
        GB_file_dlclose (e->dl_handle) ; e->dl_handle = NULL ;
    }
    ASSERT_TABLE_OK ;
}

//------------------------------------------------------------------------------
// GB_jitifyer_table_free:  free the hash and clear all loaded kernels
//------------------------------------------------------------------------------

// Clears all runtime JIT kernels from the hash table.  PreJIT kernels and JIT
// kernels containing user-defined operators are not freed if freall is true
// (only done by GrB_finalize), but they are flagged as unchecked.  This allows
// the application to call GxB_set to set the JIT control to OFF then ON again,
// to indicate that a user-defined type or operator has been changed, and that
// all JIT kernels must cleared and all PreJIT kernels checked again before
// using them.

// After calling this function, the JIT is still enabled.  GB_jitifyer_insert
// will reallocate the table if it is NULL.

void GB_jitifyer_table_free (bool freeall)
{ 
    if (GB_jit_table != NULL)
    {
        for (uint64_t k = 0 ; k < GB_jit_table_size ; k++)
        {
            GB_jit_entry *e = &(GB_jit_table [k]) ;
            if (e->dl_function != NULL)
            {
                // found an entry
                if (e->dl_handle == NULL)
                { 
                    // flag the PreJIT kernel as unchecked
                    e->prejit_index = GB_PREJIT_UNCHECKED (e->prejit_index) ;
                }
                // free it if permitted
                if (freeall || (e->dl_handle != NULL &&
                      e->encoding.kcode != GB_JIT_KERNEL_USEROP))
                { 
                    // free the entry
                    GB_jitifyer_entry_free (e) ;
                }
            }
        }
    }

    ASSERT (GB_IMPLIES (freeall, GB_jit_table_populated == 0)) ;
    if (GB_jit_table_populated == 0)
    { 
        // the JIT table is now empty, so free it
        GB_FREE_STUFF (GB_jit_table) ;
        GB_jit_table_size = 0 ;
        GB_jit_table_bits = 0 ;
    }
}

//------------------------------------------------------------------------------
// GB_jitifyer_command: run a command in a child process
//------------------------------------------------------------------------------

// No error condition or status is returned.

// If burble is on, stdout is left alone, so the stdout of the command is sent
// to the stdout of the parent process.  If burble is off, stdout is sent to
// /dev/null (nul on Windows).  If there is no error log file, stderr is not
// redirected; otherwise, it is redirected to that file.  The redirects are
// handled by modifying the command string in the caller, so they do not have
// to be handled here.

// NOTE: this call to system(...) *cannot* be sanitized; CodeQL flags calls to
// GB_jitifyer_command as security issues, but this is intentional.  The JIT
// allows the end user to create arbitrary user-defined types and operators,
// which GraphBLAS then injects into C source code of a JIT kernel created at
// run time.  The end user can also specify an arbitrary compiler to compile
// JIT kernels.  This code injection is intentional, and required for the JIT.
// If a security-hardened GraphBLAS library is required, then GraphBLAS must be
// compiled with -DNJIT to disable the JIT entirely.  User-defined JIT kernels
// will not be created at run time, and thus user-defined types and operators
// will be slow, however.  This option does not disable any PreJIT kernels, so
// if fast user-defined kernels are required, they can be used with the PreJIT
// mechanism; see the GraphBLAS User Guide for details.

// The return result is unused.
#include "include/GB_unused.h"

static void GB_jitifyer_command (char *command)
{ 
    #ifndef NJIT
    int result = system (command) ;
    #endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_cmake_compile: compile a kernel with cmake
//------------------------------------------------------------------------------

// This method does not return any error/success code.  If the compilation
// fails for any reason, the subsequent load of the compiled kernel will fail.

// This method works on most platforms (not yet on MINGW, so it is not used
// there).  For Windows using MSVC, this method is always used.

// FUTURE: get this method to work in MINGW.

#define GB_BLD_DIR "%s/tmp/%016" PRIx64

void GB_jitifyer_cmake_compile (char *kernel_name, uint64_t hash)
{ 
#ifndef NJIT

    uint32_t bucket = hash & 0xFF ;
    GBURBLE ("(jit compile with cmake)\n") ;
    char *burble_stdout = GB_Global_burble_get ( ) ? "" : GB_DEV_NULL ;
    bool have_log = (GB_STRLEN (GB_jit_error_log) > 0) ;
    char *err_redirect = have_log ?  " 2>> " : " 2>&1 " ;
    char *log_quote = have_log ? "\"" : "" ;

#if defined (__MINGW32__)
#define GB_SH_C "sh -c "
#else
#define GB_SH_C
#endif

    // remove any prior build folder for this kernel, and all its contents
    snprintf (GB_jit_temp, GB_jit_temp_allocated,
        GB_SH_C "cmake -E remove_directory \"" GB_BLD_DIR "\" %s %s %s%s%s",
        GB_jit_cache_path, hash,     // build path
        burble_stdout, err_redirect,
        log_quote, GB_jit_error_log, log_quote) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // create the build folder for this kernel
    snprintf (GB_jit_temp, GB_jit_temp_allocated, GB_BLD_DIR,
        GB_jit_cache_path, hash) ;
    if (!GB_file_mkdir (GB_jit_temp)) return ;

    // create the CMakeLists.txt file in the build folder for this kernel
    snprintf (GB_jit_temp, GB_jit_temp_allocated, GB_BLD_DIR "/CMakeLists.txt",
        GB_jit_cache_path, hash) ;
    FILE *fp = fopen (GB_jit_temp, "w") ;
    if (fp == NULL) return ;
    fprintf (fp,
        "cmake_minimum_required ( VERSION 3.13 )\n" // end user needs cmake 3.13
        "project ( GBjit LANGUAGES C )\n"
        "include_directories ( \"%s/src\" \"%s/src/template\""
        " \"%s/src/include\" %s)\n"
        "add_compile_definitions ( GB_JIT_RUNTIME )\n",
        GB_jit_cache_path,          // include: cache/src
        GB_jit_cache_path,          // include: cache/src/template
        GB_jit_cache_path,          // include: cache/src/include
        ((strlen (GB_OMP_INC_DIRS) == 0) ? " " : " \"" GB_OMP_INC_DIRS "\" ")) ;
    // print the C flags, but escape any double quote characters
    fprintf (fp, "set ( CMAKE_C_FLAGS \"") ;
    for (char *p = GB_jit_C_flags ; *p != '\0' ; p++)
    {
        if (*p == '"') fprintf (fp, "\\") ;
        fprintf (fp, "%c", *p) ;
    }
    fprintf (fp, "\" )\n") ;
    fprintf (fp,
        "add_library ( %s SHARED \"%s/c/%02x/%s.c\" )\n",
        kernel_name,                // target name for add_library command
        GB_jit_cache_path, bucket, kernel_name) ; // source file for add_library
    if (GB_STRLEN (GB_jit_C_cmake_libs) > 0)
    {
        fprintf (fp,
            "target_link_libraries ( %s PUBLIC %s )\n",
            kernel_name,                // target name of the library
            GB_jit_C_cmake_libs) ;      // libraries to link against
    }

    fprintf (fp, 
        "set_target_properties ( %s PROPERTIES\n"
        "    C_STANDARD 11 C_STANDARD_REQUIRED ON )\n"
        "install ( TARGETS %s\n"
        "    LIBRARY DESTINATION \"%s/lib/%02x\"\n"
        "    ARCHIVE DESTINATION \"%s/lib/%02x\"\n"
        "    RUNTIME DESTINATION \"%s/lib/%02x\" )\n",
        kernel_name,
        kernel_name,
        GB_jit_cache_path, bucket,
        GB_jit_cache_path, bucket,
        GB_jit_cache_path, bucket) ;
    fclose (fp) ;

    // generate the build system for this kernel
    snprintf (GB_jit_temp, GB_jit_temp_allocated,
        GB_SH_C "cmake -S \"" GB_BLD_DIR "\" -B \"" GB_BLD_DIR "\""
        " -DCMAKE_C_COMPILER=\"%s\" %s %s %s%s%s",
        GB_jit_cache_path, hash,     // -S source path
        GB_jit_cache_path, hash,     // -B build path
        GB_jit_C_compiler,                  // C compiler to use
        burble_stdout, err_redirect, log_quote, GB_jit_error_log, log_quote) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // compile the library for this kernel
    snprintf (GB_jit_temp, GB_jit_temp_allocated,
        GB_SH_C "cmake --build \"" GB_BLD_DIR
        "\" --config Release %s %s %s%s%s",
        // can add "--verbose" here too
        GB_jit_cache_path, hash,     // build path
        burble_stdout, err_redirect, log_quote, GB_jit_error_log, log_quote) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // install the library
    snprintf (GB_jit_temp, GB_jit_temp_allocated,
        GB_SH_C "cmake --install \"" GB_BLD_DIR "\" %s %s %s%s%s",
        GB_jit_cache_path, hash,     // build path
        burble_stdout, err_redirect, log_quote, GB_jit_error_log, log_quote) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // remove the build folder and all its contents
    snprintf (GB_jit_temp, GB_jit_temp_allocated,
        GB_SH_C "cmake -E remove_directory \"" GB_BLD_DIR "\" %s %s %s%s%s",
        GB_jit_cache_path, hash,     // build path
        burble_stdout, err_redirect, log_quote, GB_jit_error_log, log_quote) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

#endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_nvcc_compile: compile a CUDA kernel with nvcc
//------------------------------------------------------------------------------

// Compiles a CUDA JIT kernel in a *.cu file, containing host code that
// launches one or more device kernels.

// The input file has the form:
//
//      %s/c/%02x/%s or [cache_path]/c/[bucket]/[kernel_name].cu
//
// and the libary file is linked as
//
//      %s/lib/%02x/lib%s.so or [cache_path]/lib/[bucket]/lib[kernel_name].so
//
// All other temporary files (including *.o object files) are removed.

void GB_jitifyer_nvcc_compile
(
    char *kernel_name,
    uint32_t bucket,
    uint8_t major,
    uint8_t minor
)
{

#if defined ( GRAPHBLAS_HAS_CUDA ) && !defined ( NJIT )

    char *burble_stdout = GB_Global_burble_get ( ) ? "" : GB_DEV_NULL ;
    bool have_log = (GB_STRLEN (GB_jit_error_log) > 0) ;
    char *err_redirect = have_log ?  " 2>> " : " 2>&1 " ;
    char *log_quote = have_log ? "'" : "" ;

    GBURBLE ("(jit compiling cuda kernel: %s/c/%02x/%s.cu) ",
        GB_jit_cache_path, bucket, kernel_name) ;

    snprintf (GB_jit_temp, GB_jit_temp_allocated,

    // compile:
    "sh -c \""                          // execute with POSIX shell
    // FIXME for CUDA: use GB_CUDA_COMPILER here:
    "nvcc --version ; "
    "nvcc "                             // compiler command
    "-forward-unknown-to-host-compiler "
    "-DGB_JIT_RUNTIME=1  "              // nvcc flags
    // FIXME for CUDA: add GB_CUDA_INC here:
    "-I/usr/local/cuda/include -std=c++17 " 
    " --gpu-architecture=compute_%d%d"  // major,minor
    " --gpu-code=sm_%d%d "              // major,minor
    " -fPIC " 
    // FIXME for CUDA: add GB_CUDA_FLAGS here:
    " -O3 "   // HACK FIXME for CUDA
    " -Wno-deprecated-gpu-targets "
    "-I'%s/src' "                       // include source directory
    "-I'%s/src/template' "
    "-I'%s/src/include' "
    "-o '%s/c/%02x/%s%s' "              // *.o output file
    "-c '%s/c/%02x/%s.cu' "             // *.cu input file
    "%s "                               // burble stdout
    "%s %s%s%s ; "                      // error log file

    // link:
    "nvcc "                             // compiler
    "-DGB_JIT_RUNTIME=1  "              // nvcc flags
    "-I/usr/local/cuda/include -std=c++17 "
    " -Wno-deprecated-gpu-targets "
    " --gpu-architecture=compute_%d%d"  // major,minor
    " --gpu-code=sm_%d%d "              // major,minor
    " -shared "
    "-o '%s/lib/%02x/%s%s%s' "          // lib*.so output file
    "'%s/c/%02x/%s%s' "                 // *.o input file
    " -cudart shared "
//  "%s "                               // libraries to link with (any?)
    "%s "                               // burble stdout
    "%s %s%s%s\"",                      // error log file

    // compile:
    (int) major, (int) minor,           // CUDA compute_xy architecture
    (int) major, (int) minor,           // CUDA sm_xy code
    GB_jit_cache_path,                  // include cache/src
    GB_jit_cache_path,                  // include cache/src/template
    GB_jit_cache_path,                  // include cache/src/include
    GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX,  // *.o output file
    GB_jit_cache_path, bucket, kernel_name,                 // *.cu input file
    burble_stdout,                      // burble stdout
    err_redirect, log_quote, GB_jit_error_log, log_quote,   // error log file

    // link:
    (int) major, (int) minor,           // CUDA compute_xy architecture
    (int) major, (int) minor,           // CUDA sm_xy code
    GB_jit_cache_path, bucket,  
    GB_LIB_PREFIX, kernel_name, GB_LIB_SUFFIX,              // lib*.so file
    GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX,  // *.o input file
//  GB_jit_C_libraries                  // libraries to link with
    burble_stdout,                      // burble stdout
    err_redirect, log_quote, GB_jit_error_log, log_quote) ; // error log file

    // compile the library and return result
    GBURBLE ("(jit compile cuda:)\n%s\n", GB_jit_temp) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // remove the *.o file
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/c/%02x/%s%s",
        GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX) ;
    remove (GB_jit_temp) ;

#endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_direct_compile: compile a kernel with just the compiler
//------------------------------------------------------------------------------

// This method does not return any error/success code.  If the compilation
// fails for any reason, the subsequent load of the compiled kernel will fail.

// This method does not work on Windows with MSVC.  It works for Linux, Mac,
// or Windows with MINGW (for which it is currently the only option).

// FUTURE: get this method to work in MSVC, since it's much faster than using
// cmake on Windows.

void GB_jitifyer_direct_compile (char *kernel_name, uint32_t bucket)
{ 

#ifndef NJIT

    char *burble_stdout = GB_Global_burble_get ( ) ? "" : GB_DEV_NULL ;
    bool have_log = (GB_STRLEN (GB_jit_error_log) > 0) ;
    char *err_redirect = have_log ?  " 2>> " : " 2>&1 " ;
    char *log_quote = have_log ? "'" : "" ;

    snprintf (GB_jit_temp, GB_jit_temp_allocated,

    // compile:
    "sh -c \""                          // execute with POSIX shell
    "%s "                               // compiler command
    "-DGB_JIT_RUNTIME=1 %s "            // C flags
    "-I'%s/src' "                       // include source directory
    "-I'%s/src/template' "
    "-I'%s/src/include' "
    "%s "                               // openmp include directories
    "-o '%s/c/%02x/%s%s' "              // *.o output file
    "-c '%s/c/%02x/%s.c' "              // *.c input file
    "%s "                               // burble stdout
    "%s %s%s%s ; "                      // error log file

    // link:
    "%s "                               // C compiler
    "%s "                               // C flags
    "%s "                               // C link flags
    "-o '%s/lib/%02x/%s%s%s' "          // lib*.so output file
    "'%s/c/%02x/%s%s' "                 // *.o input file
    "%s "                               // libraries to link with
    "%s "                               // burble stdout
    "%s %s%s%s\"",                      // error log file

    // compile:
    GB_jit_C_compiler,                  // C compiler
    GB_jit_C_flags,                     // C flags
    GB_jit_cache_path,                  // include cache/src
    GB_jit_cache_path,                  // include cache/src/template
    GB_jit_cache_path,                  // include cache/src/include
    GB_OMP_INC,                         // openmp include
    GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX,  // *.o output file
    GB_jit_cache_path, bucket, kernel_name,                 // *.c input file
    burble_stdout,                      // burble stdout
    err_redirect, log_quote, GB_jit_error_log, log_quote,   // error log file

    // link:
    GB_jit_C_compiler,                  // C compiler
    GB_jit_C_flags,                     // C flags
    GB_jit_C_link_flags,                // C link flags
    GB_jit_cache_path, bucket,  
    GB_LIB_PREFIX, kernel_name, GB_LIB_SUFFIX,              // lib*.so file
    GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX,  // *.o input file
    GB_jit_C_libraries,                 // libraries to link with
    burble_stdout,                      // burble stdout
    err_redirect, log_quote, GB_jit_error_log, log_quote) ; // error log file

    // compile the library and return result
    GBURBLE ("(jit compile:)\n%s\n", GB_jit_temp) ;
    GB_jitifyer_command (GB_jit_temp) ; // OK: see security comment above

    // remove the *.o file
    snprintf (GB_jit_temp, GB_jit_temp_allocated, "%s/c/%02x/%s%s",
        GB_jit_cache_path, bucket, kernel_name, GB_OBJ_SUFFIX) ;
    remove (GB_jit_temp) ;

#endif
}

//------------------------------------------------------------------------------
// GB_jitifyer_hash:  compute the hash
//------------------------------------------------------------------------------

// xxHash uses switch statements with no default case.
#if GB_COMPILER_GCC
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

#define XXH_INLINE_ALL
#define XXH_NO_STREAM
#include "xxhash.h"

// A hash value of zero is unique, and is used for all builtin operators and
// types to indicate that its hash value is not required.

// A hash value of UINT64_MAX is also special: it denotes an object that cannot
// be JIT'd.

// So in the nearly impossible case that XXH3_64bits returns a hash value that
// happens to be zero or UINT64_MAX, it is reset to GB_MAGIC instead.

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
)
{ 
    uint64_t hash ;
    hash = XXH3_64bits ((const void *) encoding, sizeof (GB_jit_encoding)) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes,          // # of bytes to hash
    bool jitable            // true if the object can be JIT'd
)
{ 
    if (!jitable) return (UINT64_MAX) ;
    if (bytes == NULL || nbytes == 0) return (0) ;
    uint64_t hash ;
    hash = XXH3_64bits (bytes, nbytes) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

