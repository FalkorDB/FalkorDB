//------------------------------------------------------------------------------
// init with GPUs
//------------------------------------------------------------------------------

    GrB_init (mode) ; // where mode is one of:

        GrB_BLOCKING            // blocking, no GPU(s)
        GrB_NONBLOCKING         // nonblocking, no GPU(s)
        GxB_BLOCKING_GPU        // blocking, with GPU(s)
        GxB_NONBLOCKING_GPU     // nonblocking with GPU(s)

    // By default: if GrB_init is called with a GxB_*_GPU mode, then any
    // call to GraphBLAS may choose to use none, 1, or all GPUs in the system.
    // No GrB_get/set (below) is required.

    // GrB_init numbers the GPUs it can see from 0 to ngpus-1.  The actual
    // GPUs can be set in the environment variable CUDA_VISIBLE_DEVICES but
    // that is outside the scope of GraphBLAS.

    // When the GPU is used, the malloc/calloc/realloc/free functions are
    // based on the Rapids Memory Manager.  They cannot be changed.  A user
    // application can query these function pointers with GrB_get.

//------------------------------------------------------------------------------
// get/set: control which GPUs may be used by any given call to GraphBLAS
//------------------------------------------------------------------------------

    // By default, any call to GraphBLAS can use any number of GPU(s).
    // Currently, all CUDA kernels use at most one GPU however.

    // A Context object can be created from GxB_Context_new to control a given
    // user thread, or GrB_GLOBAL can be used below in place of the Context,
    // to control all calls to GraphBLAS.

    // get the # of GPUs currently available in this computer system
    int32_t ngpus_max ;
    GrB_get (GrB_GLOBAL, &ngpus_max, GxB_NGPUS_MAX) ;

    // set the list of GPUs to use to ids 0 to 3 (context)
    int32_t ngpus = 4 ;
    GrB_set (Context, ngpus, GxB_NGPUS) ;

    // set the list of GPUs to use to ids 0 to 3 (global)
    int32_t ngpus = 4 ;
    GrB_set (GrB_GLOBAL, ngpus, GxB_NGPUS) ;

    // set the list of GPUs to be [3 0 2], for this context
    int32_t gpu_ids [3] = {3, 0, 2} ;
    GrB_set (Context, 3, GxB_NGPUS) ;
    GrB_set (Context, (void *) gpu_ids, GxB_GPU_IDS, 3 * sizeof (int32_t)) ;

    // set the list of GPUs to be [3 0 2], for global
    int32_t gpu_ids [3] = {3, 0, 2} ;
    GrB_set (GrB_GLOBAL, 3, GxB_NGPUS) ;
    GrB_set (GrB_GLOBAL, (void *) gpu_ids, GxB_GPU_IDS, 3 * sizeof (int32_t)) ;

    // get the # of GPUs in use, for this context
    int32_t ngpus ;
    GrB_get (Context, &ngpus, GxB_NGPUS) ;

    // get the # of GPUs in use, globally
    int32_t ngpus ;
    GrB_get (GrB_GLOBAL, &ngpus, GxB_NGPUS) ;

    // get the list of GPUs in use, in a context
    int32_t ngpus ;
    GrB_get (Context, &ngpus, GxB_NGPUS) ;
    int32_t *gpu_ids = malloc (ngpus * sizeof (int32_t)) ;
    GrB_get (Context, (void *) gpu_ids, GxB_GPU_IDS) ;

    // get the list of GPUs in use, globally
    int32_t ngpus ;
    GrB_get (GrB_GLOBAL, &ngpus, GxB_NGPUS) ;
    int32_t *gpu_ids = malloc (ngpus * sizeof (int32_t)) ;
    GrB_get (GrB_GLOBAL, (void *) gpu_ids, GxB_GPU_IDS) ;

//------------------------------------------------------------------------------
// get/set: control when to use the available GPU(s)
//------------------------------------------------------------------------------

    // Possible solution below.  This is not yet implemented.

    // get the chunk factor
    GrB_Scalar chunk ;
    GrB_Scalar_new (&chunk, GrB_FP64) ;
    GrB_get (Context, chunk, GxB_GPU_CHUNK) ;

    // set the chunk factor to 1e7
    GrB_Scalar chunk ;
    GrB_Scalar_new (&chunk, GrB_FP64) ;
    GrB_Scalar_setElement (chunk, 1e7) ;
    GrB_set (Context, chunk, GxB_GPU_CHUNK) ;

    // If the work to do in any given call to GraphBLAS exceeds the chunk
    // factor (1e7 in the example above), then one or more GPUs may be used.
    // The chunk value may be ignored if the data is already on the GPU.

//------------------------------------------------------------------------------

    get/set: tell a matrix where to live: draft (not yet implememented)

    // set the list of GPUs to be [3 0 2], for a specific matrix
    GrB_Matrix A ;
    int32_t gpu_ids [3] = {3, 0, 2} ;
    GrB_set (A, 3, GxB_NGPUS) ;
    GrB_set (A, (void *) gpu_ids, GxB_GPU_IDS, 3 * sizeof (int32_t)) ;

    // describe how to map the matrix to the GPU(s): could be added in future
    GrB_set (A, whatever, GxB_GPU_MAPPING) ; ??


//------------------------------------------------------------------------------

    streams??

//------------------------------------------------------------------------------
// info
//------------------------------------------------------------------------------

    // any method can return a GrB_Info value of GxB_GPU_ERROR if some
    // GPU error occurred.

