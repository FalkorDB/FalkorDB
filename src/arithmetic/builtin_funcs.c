/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Single translation unit that owns every static built-in AR_FuncDesc
// and the gperf-generated perfect-hash lookup table.
//
// All 144 descriptors are plain file-scope static variables.  Their
// addresses are compile-time constants, so the read-only gperf wordlist
// can embed AR_FuncDesc * entries directly — no runtime registration.
//
// The generated lookup table is #included at the end so it shares the
// TU and can reference the static _desc_* symbols defined here.

#include "RG.h"
#include "func_desc.h"
#include "builtin_funcs_lookup.h"
#include "aggregate_funcs/agg_funcs.h"
#include "../util/rmalloc.h"

#define STRINGABLE (SI_NUMERIC | T_POINT | T_STRING | T_BOOL | SI_TEMPORAL)

// TA (t0, t1, ...) — inline argument-type array + count.
// Uses a C99 file-scope compound literal (static storage duration).
// sizeof is unevaluated for the second appearance; no duplicate object.
#define TA(...)                                                 \
    .types     = (SIType[]){__VA_ARGS__},                       \
    .types_len = (uint8_t)(sizeof((SIType[]){__VA_ARGS__})      \
                           / sizeof(SIType))

// TA0() — no argument types
#define TA0()  .types = NULL, .types_len = 0

//----------------------------------------------------------------------
// Forward declarations: scalar function implementations
// (all have external linkage in their respective .c files)
//----------------------------------------------------------------------

SIValue AR_ADD                 (SIValue *, int, void *) ;
SIValue AR_SUB                 (SIValue *, int, void *) ;
SIValue AR_MUL                 (SIValue *, int, void *) ;
SIValue AR_DIV                 (SIValue *, int, void *) ;
SIValue AR_MODULO              (SIValue *, int, void *) ;
SIValue AR_ABS                 (SIValue *, int, void *) ;
SIValue AR_CEIL                (SIValue *, int, void *) ;
SIValue AR_FLOOR               (SIValue *, int, void *) ;
SIValue AR_RAND                (SIValue *, int, void *) ;
SIValue AR_ROUND               (SIValue *, int, void *) ;
SIValue AR_SIGN                (SIValue *, int, void *) ;
SIValue AR_TOINTEGER           (SIValue *, int, void *) ;
SIValue AR_TOFLOAT             (SIValue *, int, void *) ;
SIValue AR_SQRT                (SIValue *, int, void *) ;
SIValue AR_POW                 (SIValue *, int, void *) ;
SIValue AR_EXP                 (SIValue *, int, void *) ;
SIValue AR_E                   (SIValue *, int, void *) ;
SIValue AR_LOG                 (SIValue *, int, void *) ;
SIValue AR_LOG10               (SIValue *, int, void *) ;
SIValue AR_SIN                 (SIValue *, int, void *) ;
SIValue AR_COS                 (SIValue *, int, void *) ;
SIValue AR_TAN                 (SIValue *, int, void *) ;
SIValue AR_COT                 (SIValue *, int, void *) ;
SIValue AR_ASIN                (SIValue *, int, void *) ;
SIValue AR_ACOS                (SIValue *, int, void *) ;
SIValue AR_ATAN                (SIValue *, int, void *) ;
SIValue AR_ATAN2               (SIValue *, int, void *) ;
SIValue AR_DEGREES             (SIValue *, int, void *) ;
SIValue AR_RADIANS             (SIValue *, int, void *) ;
SIValue AR_PI                  (SIValue *, int, void *) ;
SIValue AR_HAVERSIN            (SIValue *, int, void *) ;
SIValue AR_LEFT                (SIValue *, int, void *) ;
SIValue AR_LTRIM               (SIValue *, int, void *) ;
SIValue AR_RIGHT               (SIValue *, int, void *) ;
SIValue AR_RTRIM               (SIValue *, int, void *) ;
SIValue AR_REVERSE             (SIValue *, int, void *) ;
SIValue AR_SUBSTRING           (SIValue *, int, void *) ;
SIValue AR_JOIN                (SIValue *, int, void *) ;
SIValue AR_MATCHREGEX          (SIValue *, int, void *) ;
SIValue AR_REPLACEREGEX        (SIValue *, int, void *) ;
SIValue AR_TOLOWER             (SIValue *, int, void *) ;
SIValue AR_TOUPPER             (SIValue *, int, void *) ;
SIValue AR_TOSTRING            (SIValue *, int, void *) ;
SIValue AR_TOJSON              (SIValue *, int, void *) ;
SIValue AR_TRIM                (SIValue *, int, void *) ;
SIValue AR_CONTAINS            (SIValue *, int, void *) ;
SIValue AR_STARTSWITH          (SIValue *, int, void *) ;
SIValue AR_ENDSWITH            (SIValue *, int, void *) ;
SIValue AR_RANDOMUUID          (SIValue *, int, void *) ;
SIValue AR_REPLACE             (SIValue *, int, void *) ;
SIValue AR_SPLIT               (SIValue *, int, void *) ;
SIValue AR_INTERN              (SIValue *, int, void *) ;
SIValue AR_AND                 (SIValue *, int, void *) ;
SIValue AR_OR                  (SIValue *, int, void *) ;
SIValue AR_XOR                 (SIValue *, int, void *) ;
SIValue AR_NOT                 (SIValue *, int, void *) ;
SIValue AR_GT                  (SIValue *, int, void *) ;
SIValue AR_GE                  (SIValue *, int, void *) ;
SIValue AR_LT                  (SIValue *, int, void *) ;
SIValue AR_LE                  (SIValue *, int, void *) ;
SIValue AR_EQ                  (SIValue *, int, void *) ;
SIValue AR_NE                  (SIValue *, int, void *) ;
SIValue AR_IS_NULL             (SIValue *, int, void *) ;
SIValue AR_IS_NOT_NULL         (SIValue *, int, void *) ;
SIValue AR_TO_BOOLEAN          (SIValue *, int, void *) ;
SIValue AR_ISEMPTY             (SIValue *, int, void *) ;
SIValue AR_TOLIST              (SIValue *, int, void *) ;
SIValue AR_TOBOOLEANLIST       (SIValue *, int, void *) ;
SIValue AR_TOFLOATLIST         (SIValue *, int, void *) ;
SIValue AR_TOINTEGERLIST       (SIValue *, int, void *) ;
SIValue AR_TOSTRINGLIST        (SIValue *, int, void *) ;
SIValue AR_SUBSCRIPT           (SIValue *, int, void *) ;
SIValue AR_SLICE               (SIValue *, int, void *) ;
SIValue AR_RANGE               (SIValue *, int, void *) ;
SIValue AR_IN                  (SIValue *, int, void *) ;
SIValue AR_SIZE                (SIValue *, int, void *) ;
SIValue AR_HEAD                (SIValue *, int, void *) ;
SIValue AR_LAST                (SIValue *, int, void *) ;
SIValue AR_TAIL                (SIValue *, int, void *) ;
SIValue AR_REMOVE              (SIValue *, int, void *) ;
SIValue AR_SORT                (SIValue *, int, void *) ;
SIValue AR_INSERT              (SIValue *, int, void *) ;
SIValue AR_INSERTLISTELEMENTS  (SIValue *, int, void *) ;
SIValue AR_DEDUP               (SIValue *, int, void *) ;
SIValue AR_REDUCE              (SIValue *, int, void *) ;
SIValue AR_CASEWHEN            (SIValue *, int, void *) ;
SIValue AR_COALESCE            (SIValue *, int, void *) ;
SIValue AR_DISTINCT            (SIValue *, int, void *) ;
SIValue AR_ANY                 (SIValue *, int, void *) ;
SIValue AR_ALL                 (SIValue *, int, void *) ;
SIValue AR_SINGLE              (SIValue *, int, void *) ;
SIValue AR_NONE                (SIValue *, int, void *) ;
SIValue AR_LIST_COMPREHENSION  (SIValue *, int, void *) ;
SIValue AR_ID                  (SIValue *, int, void *) ;
SIValue AR_LABELS              (SIValue *, int, void *) ;
SIValue AR_HAS_LABELS          (SIValue *, int, void *) ;
SIValue AR_TYPE                (SIValue *, int, void *) ;
SIValue AR_STARTNODE           (SIValue *, int, void *) ;
SIValue AR_ENDNODE             (SIValue *, int, void *) ;
SIValue AR_EXISTS              (SIValue *, int, void *) ;
SIValue AR_INCOMEDEGREE        (SIValue *, int, void *) ;
SIValue AR_OUTGOINGDEGREE      (SIValue *, int, void *) ;
SIValue AR_PROPERTY            (SIValue *, int, void *) ;
SIValue AR_TYPEOF              (SIValue *, int, void *) ;
SIValue AR_TOPATH              (SIValue *, int, void *) ;
SIValue AR_SHORTEST_PATH       (SIValue *, int, void *) ;
SIValue AR_PATH_NODES          (SIValue *, int, void *) ;
SIValue AR_PATH_RELATIONSHIPS  (SIValue *, int, void *) ;
SIValue AR_PATH_LENGTH         (SIValue *, int, void *) ;
SIValue AR_TOMAP               (SIValue *, int, void *) ;
SIValue AR_TOMAP_PROJECTION    (SIValue *, int, void *) ;
SIValue AR_KEYS                (SIValue *, int, void *) ;
SIValue AR_PROPERTIES          (SIValue *, int, void *) ;
SIValue AR_MERGEMAP            (SIValue *, int, void *) ;
SIValue AR_TIMESTAMP           (SIValue *, int, void *) ;
SIValue AR_LOCALTIME           (SIValue *, int, void *) ;
SIValue AR_DATE                (SIValue *, int, void *) ;
SIValue AR_LOCALDATETIME       (SIValue *, int, void *) ;
SIValue AR_DURATION            (SIValue *, int, void *) ;
SIValue AR_TOPOINT             (SIValue *, int, void *) ;
SIValue AR_DISTANCE            (SIValue *, int, void *) ;
SIValue AR_VECTOR32F           (SIValue *, int, void *) ;
SIValue AR_EUCLIDEAN_DISTANCE  (SIValue *, int, void *) ;
SIValue AR_COSINE_DISTANCE     (SIValue *, int, void *) ;
SIValue AR_PREV                (SIValue *, int, void *) ;
SIValue AR_NOP                 (SIValue *, int, void *) ;

// Aggregate step functions
SIValue AGG_AVG     (SIValue *, int, void *) ;
SIValue AGG_SUM     (SIValue *, int, void *) ;
SIValue AGG_MIN     (SIValue *, int, void *) ;
SIValue AGG_MAX     (SIValue *, int, void *) ;
SIValue AGG_COUNT   (SIValue *, int, void *) ;
SIValue AGG_COLLECT (SIValue *, int, void *) ;
SIValue AGG_PERC    (SIValue *, int, void *) ;
SIValue AGG_STDEV   (SIValue *, int, void *) ;

//----------------------------------------------------------------------
// Forward declarations: aggregate callbacks
//----------------------------------------------------------------------

void         Avg_Finalize            (void*) ;
AggregateCtx *Avg_PrivateData        (void) ;
AggregateCtx *SUM_PrivateData        (void) ;
AggregateCtx *Collect_PrivateData    (void) ;
AggregateCtx *Count_PrivateData      (void) ;
AggregateCtx *Min_PrivateData        (void) ;
AggregateCtx *Max_PrivateData        (void) ;
void         StDev_Free              (void *) ;
void         StDevFinalize           (void *) ;
void         StDevPFinalize          (void *) ;
AggregateCtx *STD_PrivateData        (void) ;
void         Percentile_Free         (void *) ;
void         PercDiscFinalize        (void *) ;
void         PercContFinalize        (void *) ;
AggregateCtx *Precentile_PrivateData (void) ;

//----------------------------------------------------------------------
// Forward declarations: scalar private-data callbacks
//----------------------------------------------------------------------

void Distinct_Free                    (void *) ;
void *Distinct_Clone                  (void *) ;
void ListComprehension_Free           (void *) ;
void *ListComprehension_Clone         (void *) ;
void ListComprehension_CollectAliases (const void *,rax *) ;
void ListReduceCtx_Free               (void *) ;
void *ListReduceCtx_Clone             (void *) ;
void ListReduceCtx_CollectAliases     (const void *, rax *) ;
void ShortestPath_Free                (void *) ;
void *ShortestPath_Clone              (void *) ;
void AR_PrevPrivateData_Free          (void *) ;
void *AR_PrevPrivateData_Clone        (void *) ;

//----------------------------------------------------------------------
// Static AR_FuncDesc definitions
//
// Fields absent from the initializer default to zero/false (C99).
// AR_FuncFree is never called on static descriptors, so the compound-
// literal type arrays are never freed.
//
// The gperf wordlist (included at the end) embeds &_desc_* as
// compile-time address constants — valid because static variables have
// static storage duration.
//----------------------------------------------------------------------

// --- NUMERIC ---

static AR_FuncDesc _desc_add = {
    .name="add", .func=AR_ADD,
    TA (SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL,
       SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL),
    .ret_type=SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_sub = {
    .name="sub", .func=AR_SUB,
    TA (SI_NUMERIC | SI_TEMPORAL | T_NULL, SI_NUMERIC | T_DURATION | T_NULL),
    .ret_type=SI_NUMERIC | SI_TEMPORAL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_mul = {
    .name="mul", .func=AR_MUL,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_div = {
    .name="div", .func=AR_DIV,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_mod = {
    .name="mod", .func=AR_MODULO,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_abs = {
    .name="abs", .func=AR_ABS,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_ceil = {
    .name="ceil", .func=AR_CEIL,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_floor = {
    .name="floor", .func=AR_FLOOR,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_rand = {
    .name="rand", .func=AR_RAND,
    TA0(), .ret_type=T_DOUBLE,
    .min_argc=0, .max_argc=0 };

static AR_FuncDesc _desc_round = {
    .name="round", .func=AR_ROUND,
    TA (SI_NUMERIC | T_NULL), .ret_type=SI_NUMERIC | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_sign = {
    .name="sign", .func=AR_SIGN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_INT64 | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tointeger = {
    .name="tointeger", .func=AR_TOINTEGER,
    TA (SI_NUMERIC | T_STRING | T_NULL | T_BOOL), .ret_type=T_INT64 | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tointegerornull = {
    .name="tointegerornull", .func=AR_TOINTEGER,
    TA (SI_ALL), .ret_type=T_INT64 | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tofloat = {
    .name="tofloat", .func=AR_TOFLOAT,
    TA (SI_NUMERIC | T_STRING | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tofloatornull = {
    .name="tofloatornull", .func=AR_TOFLOAT,
    TA (SI_ALL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_sqrt = {
    .name="sqrt", .func=AR_SQRT,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_pow = {
    .name="pow", .func=AR_POW,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_exp = {
    .name="exp", .func=AR_EXP,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_e = {
    .name="e", .func=AR_E,
    TA0(), .ret_type=T_DOUBLE,
    .min_argc=0, .max_argc=0,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_log = {
    .name="log", .func=AR_LOG,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_log10 = {
    .name="log10", .func=AR_LOG10,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_sin = {
    .name="sin", .func=AR_SIN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_cos = {
    .name="cos", .func=AR_COS,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tan = {
    .name="tan", .func=AR_TAN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_cot = {
    .name="cot", .func=AR_COT,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_asin = {
    .name="asin", .func=AR_ASIN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_acos = {
    .name="acos", .func=AR_ACOS,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_atan = {
    .name="atan", .func=AR_ATAN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_atan2 = {
    .name="atan2", .func=AR_ATAN2,
    TA (SI_NUMERIC | T_NULL, SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_degrees = {
    .name="degrees", .func=AR_DEGREES,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_radians = {
    .name="radians", .func=AR_RADIANS,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_pi = {
    .name="pi", .func=AR_PI,
    TA0(), .ret_type=T_DOUBLE,
    .min_argc=0, .max_argc=0,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_haversin = {
    .name="haversin", .func=AR_HAVERSIN,
    TA (SI_NUMERIC | T_NULL), .ret_type=T_DOUBLE | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

// --- STRING ---

static AR_FuncDesc _desc_left = {
    .name="left", .func=AR_LEFT,
    TA (T_STRING | T_NULL, T_INT64 | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_ltrim = {
    .name="ltrim", .func=AR_LTRIM,
    TA (T_STRING | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_right = {
    .name="right", .func=AR_RIGHT,
    TA (T_STRING | T_NULL, T_INT64 | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_rtrim = {
    .name="rtrim", .func=AR_RTRIM,
    TA (T_STRING | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_reverse = {
    .name="reverse", .func=AR_REVERSE,
    TA (T_STRING | T_ARRAY | T_NULL), .ret_type=T_STRING | T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_substring = {
    .name="substring", .func=AR_SUBSTRING,
    TA (T_STRING | T_NULL, T_INT64, T_INT64), .ret_type=T_STRING | T_NULL,
    .min_argc=2, .max_argc=3,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_string_join = {
    .name="string.join", .func=AR_JOIN,
    TA (T_ARRAY | T_NULL, T_STRING), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_string_matchregex = {
    .name="string.matchRegEx", .func=AR_MATCHREGEX,
    TA (T_STRING | T_NULL, T_STRING | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_string_replaceregex = {
    .name="string.replaceRegEx", .func=AR_REPLACEREGEX,
    TA (T_STRING | T_NULL, T_STRING | T_NULL, T_STRING | T_NULL),
    .ret_type=T_STRING | T_NULL,
    .min_argc=2, .max_argc=3,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tolower = {
    .name="tolower", .func=AR_TOLOWER,
    TA (T_STRING | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_toupper = {
    .name="toupper", .func=AR_TOUPPER,
    TA (T_STRING | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tostring = {
    .name="tostring", .func=AR_TOSTRING,
    TA (STRINGABLE | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tostringornull = {
    .name="tostringornull", .func=AR_TOSTRING,
    TA (SI_ALL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tojson = {
    .name="tojson", .func=AR_TOJSON,
    TA (SI_ALL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_trim = {
    .name="trim", .func=AR_TRIM,
    TA (T_STRING | T_NULL), .ret_type=T_STRING | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_contains = {
    .name="contains", .func=AR_CONTAINS,
    TA (T_STRING | T_NULL, T_STRING | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_starts_with = {
    .name="starts with", .func=AR_STARTSWITH,
    TA (T_STRING | T_NULL, T_STRING | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_ends_with = {
    .name="ends with", .func=AR_ENDSWITH,
    TA (T_STRING | T_NULL, T_STRING | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_randomuuid = {
    .name="randomuuid", .func=AR_RANDOMUUID,
    TA0(), .ret_type=T_STRING,
    .min_argc=0, .max_argc=0 };

static AR_FuncDesc _desc_replace = {
    .name="replace", .func=AR_REPLACE,
    TA (T_STRING | T_NULL, T_STRING | T_NULL, T_STRING | T_NULL),
    .ret_type=T_STRING | T_NULL,
    .min_argc=3, .max_argc=3,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_split = {
    .name="split", .func=AR_SPLIT,
    TA (T_STRING | T_NULL, T_STRING | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_intern = {
    .name="intern", .func=AR_INTERN,
    TA (T_STRING | T_NULL), .ret_type=T_INTERN_STRING | T_NULL,
    .min_argc=1, .max_argc=1, .deterministic=true };

// --- BOOLEAN ---

static AR_FuncDesc _desc_and = {
    .name="and", .func=AR_AND,
    TA (T_BOOL | T_NULL, T_BOOL | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_or = {
    .name="or", .func=AR_OR,
    TA (T_BOOL | T_NULL, T_BOOL | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_xor = {
    .name="xor", .func=AR_XOR,
    TA (T_BOOL | T_NULL, T_BOOL | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_not = {
    .name="not", .func=AR_NOT,
    TA (T_BOOL | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_gt = {
    .name="gt", .func=AR_GT,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_ge = {
    .name="ge", .func=AR_GE,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_lt = {
    .name="lt", .func=AR_LT,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_le = {
    .name="le", .func=AR_LE,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_eq = {
    .name="eq", .func=AR_EQ,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_neq = {
    .name="neq", .func=AR_NE,
    TA (SI_ALL, SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_is_null = {
    .name="is null", .func=AR_IS_NULL,
    TA (SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_is_not_null = {
    .name="is not null", .func=AR_IS_NOT_NULL,
    TA (SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_toboolean = {
    .name="toBoolean", .func=AR_TO_BOOLEAN,
    TA (T_BOOL | T_INT64 | T_STRING | T_NULL), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tobooleanornull = {
    .name="toBooleanOrNull", .func=AR_TO_BOOLEAN,
    TA (SI_ALL), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_isempty = {
    .name="isempty", .func=AR_ISEMPTY,
    TA (T_ARRAY | T_MAP | T_NULL | T_STRING), .ret_type=T_BOOL | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

// --- LIST ---

static AR_FuncDesc _desc_tolist = {
    .name="tolist", .func=AR_TOLIST,
    TA (SI_ALL), .ret_type=T_ARRAY,
    .min_argc=0, .max_argc=VAR_ARG_LEN,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tobooleanlist = {
    .name="toBooleanList", .func=AR_TOBOOLEANLIST,
    TA (T_ARRAY | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tofloatlist = {
    .name="toFloatList", .func=AR_TOFLOATLIST,
    TA (T_ARRAY | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tointegerlist = {
    .name="toIntegerList", .func=AR_TOINTEGERLIST,
    TA (T_ARRAY | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tostringlist = {
    .name="toStringList", .func=AR_TOSTRINGLIST,
    TA (T_ARRAY | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_subscript = {
    .name="subscript", .func=AR_SUBSCRIPT,
    TA (T_ARRAY | T_MAP | SI_GRAPHENTITY | T_NULL, T_INT64 | T_STRING | T_NULL),
    .ret_type=SI_ALL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_slice = {
    .name="slice", .func=AR_SLICE,
    TA (T_ARRAY | T_NULL, T_INT64 | T_NULL, T_INT64 | T_NULL),
    .ret_type=T_ARRAY | T_NULL,
    .min_argc=3, .max_argc=3,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_range = {
    .name="range", .func=AR_RANGE,
    TA (T_INT64, T_INT64, T_INT64), .ret_type=T_ARRAY | T_NULL,
    .min_argc=2, .max_argc=3,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_in = {
    .name="in", .func=AR_IN,
    TA (SI_ALL, T_ARRAY | T_NULL), .ret_type=T_NULL | T_BOOL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_size = {
    .name="size", .func=AR_SIZE,
    TA (T_STRING | T_ARRAY | T_NULL), .ret_type=T_NULL | T_INT64,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_head = {
    .name="head", .func=AR_HEAD,
    TA (T_ARRAY | T_NULL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_last = {
    .name="last", .func=AR_LAST,
    TA (T_ARRAY | T_NULL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tail = {
    .name="tail", .func=AR_TAIL,
    TA (T_ARRAY | T_NULL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_list_remove = {
    .name="list.remove", .func=AR_REMOVE,
    TA (T_ARRAY | T_NULL, T_INT64, T_INT64), .ret_type=T_ARRAY | T_NULL,
    .min_argc=2, .max_argc=3,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_list_sort = {
    .name="list.sort", .func=AR_SORT,
    TA (T_ARRAY | T_NULL, T_BOOL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_list_insert = {
    .name="list.insert", .func=AR_INSERT,
    TA (T_ARRAY | T_NULL, T_INT64, SI_ALL, T_BOOL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=3, .max_argc=4,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_list_insertlistelements = {
    .name="list.insertListElements",
    .func=AR_INSERTLISTELEMENTS,
    TA (T_ARRAY | T_NULL, T_ARRAY | T_NULL, T_INT64, T_BOOL),
    .ret_type=T_ARRAY | T_NULL,
    .min_argc=3, .max_argc=4,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_list_dedup = {
    .name="list.dedup", .func=AR_DEDUP,
    TA (T_ARRAY | T_NULL), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_reduce = {
    .name="reduce", .func=AR_REDUCE,
    TA (SI_ALL, T_ARRAY | T_NULL, T_PTR), .ret_type=SI_ALL,
    .min_argc=3, .max_argc=3,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListReduceCtx_Free,
                 .clone   = ListReduceCtx_Clone,
                 .aliases = ListReduceCtx_CollectAliases } };

// --- CONDITIONAL ---

static AR_FuncDesc _desc_case = {
    .name="case", .func=AR_CASEWHEN,
    TA (SI_ALL), .ret_type=SI_ALL,
    .min_argc=2, .max_argc=VAR_ARG_LEN,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_coalesce = {
    .name="coalesce", .func=AR_COALESCE,
    TA (SI_ALL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=VAR_ARG_LEN,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_distinct = {
    .name="distinct", .func=AR_DISTINCT,
    TA (SI_ALL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .internal=true, .deterministic=true,
    .callbacks={ .free  = Distinct_Free,
                 .clone = Distinct_Clone } };

// --- COMPREHENSION ---

static AR_FuncDesc _desc_any = {
    .name="any", .func=AR_ANY,
    TA (T_ARRAY | T_NULL, T_PTR), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListComprehension_Free,
                 .clone   = ListComprehension_Clone,
                 .aliases = ListComprehension_CollectAliases } };

static AR_FuncDesc _desc_all = {
    .name="all", .func=AR_ALL,
    TA (T_ARRAY | T_NULL, T_PTR), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListComprehension_Free,
                 .clone   = ListComprehension_Clone,
                 .aliases = ListComprehension_CollectAliases } };

static AR_FuncDesc _desc_single = {
    .name="single", .func=AR_SINGLE,
    TA (T_ARRAY | T_NULL, T_PTR), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListComprehension_Free,
                 .clone   = ListComprehension_Clone,
                 .aliases = ListComprehension_CollectAliases } };

static AR_FuncDesc _desc_none = {
    .name="none", .func=AR_NONE,
    TA (T_ARRAY | T_NULL, T_PTR), .ret_type=T_BOOL | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListComprehension_Free,
                 .clone   = ListComprehension_Clone,
                 .aliases = ListComprehension_CollectAliases } };

static AR_FuncDesc _desc_list_comprehension = {
    .name="list_comprehension", .func=AR_LIST_COMPREHENSION,
    TA (T_ARRAY | T_NULL, T_PTR), .ret_type=T_ARRAY | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true,
    .callbacks={ .free    = ListComprehension_Free,
                 .clone   = ListComprehension_Clone,
                 .aliases = ListComprehension_CollectAliases } };

// --- ENTITY ---

static AR_FuncDesc _desc_id = {
    .name="id", .func=AR_ID,
    TA (T_NULL | T_NODE | T_EDGE), .ret_type=T_NULL | T_INT64,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_labels = {
    .name="labels", .func=AR_LABELS,
    TA (T_NULL | T_NODE), .ret_type=T_NULL | T_ARRAY,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_haslabels = {
    .name="hasLabels", .func=AR_HAS_LABELS,
    TA (T_NULL | T_NODE, T_ARRAY), .ret_type=T_NULL | T_BOOL,
    .min_argc=2, .max_argc=2, .deterministic=true };

static AR_FuncDesc _desc_type = {
    .name="type", .func=AR_TYPE,
    TA (T_NULL | T_EDGE), .ret_type=T_NULL | T_STRING,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_startnode = {
    .name="startNode", .func=AR_STARTNODE,
    TA (T_NULL | T_EDGE), .ret_type=T_NULL | T_NODE,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_endnode = {
    .name="endNode", .func=AR_ENDNODE,
    TA (T_NULL | T_EDGE), .ret_type=T_NULL | T_NODE,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_exists = {
    .name="exists", .func=AR_EXISTS,
    TA (T_NULL | SI_ALL), .ret_type=T_NULL | T_BOOL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_indegree = {
    .name="indegree", .func=AR_INCOMEDEGREE,
    TA (T_NULL | T_NODE, T_STRING | T_ARRAY), .ret_type=T_NULL | T_INT64,
    .min_argc=1, .max_argc=VAR_ARG_LEN,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_outdegree = {
    .name="outdegree", .func=AR_OUTGOINGDEGREE,
    TA (T_NULL | T_NODE, T_STRING | T_ARRAY), .ret_type=T_NULL | T_INT64,
    .min_argc=1, .max_argc=VAR_ARG_LEN,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_property = {
    .name="property", .func=AR_PROPERTY,
    TA (T_NULL | T_NODE | T_EDGE | T_MAP | T_POINT | SI_TEMPORAL,
       T_STRING, T_INT64),
    .ret_type=SI_ALL,
    .min_argc=3, .max_argc=3,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_typeof = {
    .name="typeof", .func=AR_TYPEOF,
    TA (T_NULL | SI_ALL), .ret_type=T_STRING,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

// --- PATH ---

static AR_FuncDesc _desc_topath = {
    .name="topath", .func=AR_TOPATH,
    TA (T_PTR, T_NULL | T_NODE | T_EDGE | T_PATH), .ret_type=T_PATH | T_NULL,
    .min_argc=1, .max_argc=VAR_ARG_LEN,
    .internal=true, .deterministic=true };

static AR_FuncDesc _desc_shortestpath = {
    .name="shortestpath", .func=AR_SHORTEST_PATH,
    TA (T_NULL | T_NODE, T_NULL | T_NODE), .ret_type=T_PATH | T_NULL,
    .min_argc=2, .max_argc=2,
    .internal=true, .deterministic=true,
    .callbacks={ .free  = ShortestPath_Free,
                 .clone = ShortestPath_Clone } };

static AR_FuncDesc _desc_nodes = {
    .name="nodes", .func=AR_PATH_NODES,
    TA (T_NULL | T_PATH), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1, .deterministic=true };

static AR_FuncDesc _desc_relationships = {
    .name="relationships", .func=AR_PATH_RELATIONSHIPS,
    TA (T_NULL | T_PATH), .ret_type=T_ARRAY | T_NULL,
    .min_argc=1, .max_argc=1, .deterministic=true };

static AR_FuncDesc _desc_length = {
    .name="length", .func=AR_PATH_LENGTH,
    TA (T_NULL | T_PATH), .ret_type=T_INT64 | T_NULL,
    .min_argc=1, .max_argc=1, .deterministic=true };

// --- MAP ---

static AR_FuncDesc _desc_tomap = {
    .name="tomap", .func=AR_TOMAP,
    TA (SI_ALL), .ret_type=T_MAP,
    .min_argc=0, .max_argc=VAR_ARG_LEN,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_tomap_projection = {
    .name="tomap_projection", .func=AR_TOMAP_PROJECTION,
    TA (SI_ALL), .ret_type=T_NULL | T_MAP,
    .min_argc=1, .max_argc=VAR_ARG_LEN,
    .internal=true, .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_keys = {
    .name="keys", .func=AR_KEYS,
    TA (T_NULL | T_MAP | T_NODE | T_EDGE), .ret_type=T_NULL | T_ARRAY,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_properties = {
    .name="properties", .func=AR_PROPERTIES,
    TA (T_NULL | T_MAP | T_NODE | T_EDGE), .ret_type=T_NULL | T_MAP,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_merge_maps = {
    .name="merge_maps", .func=AR_MERGEMAP,
    TA (T_NULL | T_MAP, T_NULL | T_MAP), .ret_type=T_NULL | T_MAP,
    .min_argc=2, .max_argc=2,
    .internal=true, .reducible=true, .deterministic=true };

// --- TIME ---

static AR_FuncDesc _desc_timestamp = {
    .name="timestamp", .func=AR_TIMESTAMP,
    TA0(), .ret_type=T_INT64,
    .min_argc=0, .max_argc=0 };

static AR_FuncDesc _desc_localtime = {
    .name="localtime", .func=AR_LOCALTIME,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_TIME | T_NULL,
    .min_argc=0, .max_argc=1 };

static AR_FuncDesc _desc_localtime_transaction = {
    .name="localtime.transaction", .func=AR_LOCALTIME,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_TIME | T_NULL,
    .min_argc=0, .max_argc=1, .reducible=true };

static AR_FuncDesc _desc_date = {
    .name="date", .func=AR_DATE,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_DATE | T_NULL,
    .min_argc=0, .max_argc=1 };

static AR_FuncDesc _desc_date_transaction = {
    .name="date.transaction", .func=AR_DATE,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_DATE | T_NULL,
    .min_argc=0, .max_argc=1, .reducible=true };

static AR_FuncDesc _desc_localdatetime = {
    .name="localdatetime", .func=AR_LOCALDATETIME,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_DATETIME | T_NULL,
    .min_argc=0, .max_argc=1 };

static AR_FuncDesc _desc_localdatetime_transaction = {
    .name="localdatetime.transaction", .func=AR_LOCALDATETIME,
    TA (T_STRING | T_MAP | T_NULL), .ret_type=T_DATETIME | T_NULL,
    .min_argc=0, .max_argc=1, .reducible=true };

static AR_FuncDesc _desc_duration = {
    .name="duration", .func=AR_DURATION,
    TA (T_MAP | T_STRING | T_NULL), .ret_type=T_DURATION | T_NULL,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

// --- POINT ---

static AR_FuncDesc _desc_point = {
    .name="point", .func=AR_TOPOINT,
    TA (T_NULL | T_MAP), .ret_type=T_NULL | T_POINT,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_distance = {
    .name="distance", .func=AR_DISTANCE,
    TA (T_NULL | T_POINT, T_NULL | T_POINT), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

// --- VECTOR ---

static AR_FuncDesc _desc_vecf32 = {
    .name="vecf32", .func=AR_VECTOR32F,
    TA (T_NULL | T_ARRAY), .ret_type=T_NULL | T_VECTOR,
    .min_argc=1, .max_argc=1,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_vec_euclideandistance = {
    .name="vec.euclideanDistance", .func=AR_EUCLIDEAN_DISTANCE,
    TA (T_NULL | T_VECTOR, T_NULL | T_VECTOR), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

static AR_FuncDesc _desc_vec_cosinedistance = {
    .name="vec.cosineDistance", .func=AR_COSINE_DISTANCE,
    TA (T_NULL | T_VECTOR, T_NULL | T_VECTOR), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=2, .max_argc=2,
    .reducible=true, .deterministic=true };

// --- GENERAL ---

static AR_FuncDesc _desc_prev = {
    .name="prev", .func=AR_PREV,
    TA (T_NULL | SI_ALL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1, .deterministic=true,
    .callbacks={ .free  = AR_PrevPrivateData_Free,
                 .clone = AR_PrevPrivateData_Clone } };

// --- PLACEHOLDER ---

static AR_FuncDesc _desc_path_filter = {
    .name="path_filter", .func=AR_NOP,
    TA0(), .ret_type=T_NULL,
    .min_argc=0, .max_argc=0, .deterministic=true, .internal=true };

static AR_FuncDesc _desc_nop = {
    .name="nop", .func=AR_NOP,
    TA0(), .ret_type=T_NULL,
    .min_argc=0, .max_argc=0, .deterministic=true, .internal=true };

// --- AGGREGATE ---

static AR_FuncDesc _desc_avg = {
    .name="avg", .func=AGG_AVG,
    TA (T_NULL | T_INT64 | T_DOUBLE), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .free         = rm_free,
                 .finalize     = Avg_Finalize,
                 .private_data = Avg_PrivateData } };

static AR_FuncDesc _desc_sum = {
    .name="sum", .func=AGG_SUM,
    TA (T_NULL | T_INT64 | T_DOUBLE), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .private_data = SUM_PrivateData } };

static AR_FuncDesc _desc_collect = {
    .name="collect", .func=AGG_COLLECT,
    TA (SI_ALL), .ret_type=T_NULL | T_ARRAY,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .private_data = Collect_PrivateData } };

static AR_FuncDesc _desc_count = {
    .name="count", .func=AGG_COUNT,
    TA (SI_ALL), .ret_type=T_INT64,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .private_data = Count_PrivateData } };

static AR_FuncDesc _desc_min = {
    .name="min", .func=AGG_MIN,
    TA (SI_ALL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .private_data = Min_PrivateData } };

static AR_FuncDesc _desc_max = {
    .name="max", .func=AGG_MAX,
    TA (SI_ALL), .ret_type=SI_ALL,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .private_data = Max_PrivateData } };

static AR_FuncDesc _desc_stdev = {
    .name="stDev", .func=AGG_STDEV,
    TA (T_NULL | T_INT64 | T_DOUBLE), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .free         = StDev_Free,
                 .finalize     = StDevFinalize,
                 .private_data = STD_PrivateData } };

static AR_FuncDesc _desc_stdevp = {
    .name="stDevP", .func=AGG_STDEV,
    TA (T_NULL | T_INT64 | T_DOUBLE), .ret_type=T_NULL | T_DOUBLE,
    .min_argc=1, .max_argc=1,
    .aggregate=true, .deterministic=true,
    .callbacks={ .free         = StDev_Free,
                 .finalize     = StDevPFinalize,
                 .private_data = STD_PrivateData } };

static AR_FuncDesc _desc_percentiledisc = {
    .name="percentileDisc", .func=AGG_PERC,
    TA (T_NULL | T_INT64 | T_DOUBLE, T_NULL | T_INT64 | T_DOUBLE),
    .ret_type=T_NULL | T_DOUBLE,
    .min_argc=2, .max_argc=2,
    .aggregate=true, .deterministic=true,
    .callbacks={ .free         = Percentile_Free,
                 .finalize     = PercDiscFinalize,
                 .private_data = Precentile_PrivateData } };

static AR_FuncDesc _desc_percentilecont = {
    .name="percentileCont", .func=AGG_PERC,
    TA (T_NULL | T_INT64 | T_DOUBLE, T_NULL | T_INT64 | T_DOUBLE),
    .ret_type=T_NULL | T_DOUBLE,
    .min_argc=2, .max_argc=2,
    .aggregate=true, .deterministic=true,
    .callbacks={ .free         = Percentile_Free,
                 .finalize     = PercContFinalize,
                 .private_data = Precentile_PrivateData } };

#undef TA
#undef TA0

// Flat enumeration array — same 144 entries used by dbms.functions().
// Order matches the module grouping above (numeric → string → … → agg).
AR_FuncDesc * const AR_BUILTIN_FUNCS[NUM_BUILTIN_FUNCS] = {
	//--------------------------------------------------------------------------
    // numeric
	//--------------------------------------------------------------------------

    &_desc_add,
    &_desc_sub,
    &_desc_mul,
    &_desc_div,
    &_desc_mod,
    &_desc_abs,
    &_desc_ceil,
    &_desc_floor,
    &_desc_rand,
    &_desc_round,
    &_desc_sign,
    &_desc_tointeger,
    &_desc_tointegerornull,
    &_desc_tofloat,
    &_desc_tofloatornull,
    &_desc_sqrt,
    &_desc_pow,
    &_desc_exp,
    &_desc_e,
    &_desc_log,
    &_desc_log10,
    &_desc_sin,
    &_desc_cos,
    &_desc_tan,
    &_desc_cot,
    &_desc_asin,
    &_desc_acos,
    &_desc_atan,
    &_desc_atan2,
    &_desc_degrees,
    &_desc_radians,
    &_desc_pi,
    &_desc_haversin,

	//--------------------------------------------------------------------------
    // string
	//--------------------------------------------------------------------------

    &_desc_left,
    &_desc_ltrim,
    &_desc_right,
    &_desc_rtrim,
    &_desc_reverse,
    &_desc_substring,
    &_desc_string_join,
    &_desc_string_matchregex,
    &_desc_string_replaceregex,
    &_desc_tolower,
    &_desc_toupper,
    &_desc_tostring,
    &_desc_tostringornull,
    &_desc_tojson,
    &_desc_trim,
    &_desc_contains,
    &_desc_starts_with,
    &_desc_ends_with,
    &_desc_randomuuid,
    &_desc_replace,
    &_desc_split,
    &_desc_intern,

	//--------------------------------------------------------------------------
    // boolean
	//--------------------------------------------------------------------------

    &_desc_and,
    &_desc_or,
    &_desc_xor,
    &_desc_not,
    &_desc_gt,
    &_desc_ge,
    &_desc_lt,
    &_desc_le,
    &_desc_eq,
    &_desc_neq,
    &_desc_is_null,
    &_desc_is_not_null,
    &_desc_toboolean,
    &_desc_tobooleanornull,
    &_desc_isempty,

	//--------------------------------------------------------------------------
    // list
	//--------------------------------------------------------------------------

    &_desc_tolist,
    &_desc_tobooleanlist,
    &_desc_tofloatlist,
    &_desc_tointegerlist,
    &_desc_tostringlist,
    &_desc_subscript,
    &_desc_slice,
    &_desc_range,
    &_desc_in,
    &_desc_size,
    &_desc_head,
    &_desc_last,
    &_desc_tail,
    &_desc_list_remove,
    &_desc_list_sort,
    &_desc_list_insert,
    &_desc_list_insertlistelements,
    &_desc_list_dedup,
    &_desc_reduce,

	//--------------------------------------------------------------------------
    // conditional
	//--------------------------------------------------------------------------

    &_desc_case,
    &_desc_coalesce,
    &_desc_distinct,

	//--------------------------------------------------------------------------
    // comprehension
	//--------------------------------------------------------------------------

    &_desc_any,
    &_desc_all,
    &_desc_single,
    &_desc_none,
    &_desc_list_comprehension,

	//--------------------------------------------------------------------------
    // entity
	//--------------------------------------------------------------------------

    &_desc_id,
    &_desc_labels,
    &_desc_haslabels,
    &_desc_type,
    &_desc_startnode,
    &_desc_endnode,
    &_desc_exists,
    &_desc_indegree,
    &_desc_outdegree,
    &_desc_property,
    &_desc_typeof,

	//--------------------------------------------------------------------------
    // path
	//--------------------------------------------------------------------------

    &_desc_topath,
    &_desc_shortestpath,
    &_desc_nodes,
    &_desc_relationships,
    &_desc_length,

	//--------------------------------------------------------------------------
    // map
	//--------------------------------------------------------------------------

    &_desc_tomap,
    &_desc_tomap_projection,
    &_desc_keys,
    &_desc_properties,
    &_desc_merge_maps,

	//--------------------------------------------------------------------------
    // time
	//--------------------------------------------------------------------------

    &_desc_timestamp,
    &_desc_localtime,
    &_desc_localtime_transaction,
    &_desc_date,
    &_desc_date_transaction,
    &_desc_localdatetime,
    &_desc_localdatetime_transaction,
    &_desc_duration,

	//--------------------------------------------------------------------------
    // point
	//--------------------------------------------------------------------------

    &_desc_point,
    &_desc_distance,

	//--------------------------------------------------------------------------
    // vector
	//--------------------------------------------------------------------------

    &_desc_vecf32,
    &_desc_vec_euclideandistance,
    &_desc_vec_cosinedistance,

	//--------------------------------------------------------------------------
    // general
	//--------------------------------------------------------------------------

    &_desc_prev,

	//--------------------------------------------------------------------------
    // placeholder
	//--------------------------------------------------------------------------

    &_desc_path_filter,
    &_desc_nop,
    
	//--------------------------------------------------------------------------
    // aggregate
	//--------------------------------------------------------------------------

    &_desc_avg,
    &_desc_sum,
    &_desc_collect,
    &_desc_count,
    &_desc_min,
    &_desc_max,
    &_desc_stdev,
    &_desc_stdevp,
    &_desc_percentiledisc,
    &_desc_percentilecont,
};

//----------------------------------------------------------------------
// Perfect-hash lookup table (gperf-generated).
// Must come last: the wordlist references &_desc_* symbols above.
//----------------------------------------------------------------------
#include "builtin_funcs_lookup.inc"
