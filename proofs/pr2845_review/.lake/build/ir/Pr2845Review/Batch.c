// Lean compiler output
// Module: Pr2845Review.Batch
// Imports: public import Init public meta import Init
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_List_any___redArg(lean_object*, lean_object*);
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_instReprNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_List_repr_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Option_repr___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t l_List_instDecidableEqNil___redArg(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instReprNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__0_value;
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_repr_x27___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__1_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__2_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "len"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__3_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__3_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__4_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__4_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__5 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__5_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__6 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__6_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__6_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__7 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__7_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__5_value),((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__7_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__8 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__8_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__10 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__10_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__10_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__11 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__11_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "sel"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__12 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__12_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__12_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__13 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__13_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cols"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__14 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__14_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__14_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__15 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__15_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "origins"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__17 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__17_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__17_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__18 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__18_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__20 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__20_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__23 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__23_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__20_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__24 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__24_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_active(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Batch_0__Pr2845_Batch_originRow_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Batch_0__Pr2845_Batch_originRow_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0___boxed(lean_object*);
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyGeneral___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyGeneral(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyFast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyFast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMain___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMerged___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMerged(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandOld(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandOld___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandNew(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandNew___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_colless___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_colless___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_colless___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_colless___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__1_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_colless___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__1_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_colless___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__2_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_colless___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_colless___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__3_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_colless___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__3_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_colless___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__4_value;
LEAN_EXPORT const lean_object* lp_pr2845_x2dreview_Pr2845_colless = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_colless___closed__4_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0___redArg(lean_object*, lean_object*);
static const lean_array_object lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_concatColless___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_concatColless(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_4_ = l_instDecidableEqList___redArg(v___x_3_, v_a_1_, v_b_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0___boxed(lean_object* v_a_5_, lean_object* v_b_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__0(v_a_5_, v_b_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1(lean_object* v_inst_9_, lean_object* v_a_10_, lean_object* v_b_11_){
_start:
{
uint8_t v___x_12_; 
v___x_12_ = l_instDecidableEqList___redArg(v_inst_9_, v_a_10_, v_b_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1___boxed(lean_object* v_inst_13_, lean_object* v_a_14_, lean_object* v_b_15_){
_start:
{
uint8_t v_res_16_; lean_object* v_r_17_; 
v_res_16_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1(v_inst_13_, v_a_14_, v_b_15_);
v_r_17_ = lean_box(v_res_16_);
return v_r_17_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2(lean_object* v___f_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
uint8_t v___x_21_; 
v___x_21_ = l_Option_instDecidableEq___redArg(v___f_18_, v_a_19_, v_b_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2___boxed(lean_object* v___f_22_, lean_object* v_a_23_, lean_object* v_b_24_){
_start:
{
uint8_t v_res_25_; lean_object* v_r_26_; 
v_res_25_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2(v___f_22_, v_a_23_, v_b_24_);
v_r_26_ = lean_box(v_res_25_);
return v_r_26_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(lean_object* v_inst_28_, lean_object* v_x_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_len_31_; lean_object* v_sel_32_; lean_object* v_cols_33_; lean_object* v_origins_34_; lean_object* v_len_35_; lean_object* v_sel_36_; lean_object* v_cols_37_; lean_object* v_origins_38_; uint8_t v___x_39_; 
v_len_31_ = lean_ctor_get(v_x_29_, 0);
lean_inc(v_len_31_);
v_sel_32_ = lean_ctor_get(v_x_29_, 1);
lean_inc(v_sel_32_);
v_cols_33_ = lean_ctor_get(v_x_29_, 2);
lean_inc(v_cols_33_);
v_origins_34_ = lean_ctor_get(v_x_29_, 3);
lean_inc(v_origins_34_);
lean_dec_ref(v_x_29_);
v_len_35_ = lean_ctor_get(v_x_30_, 0);
lean_inc(v_len_35_);
v_sel_36_ = lean_ctor_get(v_x_30_, 1);
lean_inc(v_sel_36_);
v_cols_37_ = lean_ctor_get(v_x_30_, 2);
lean_inc(v_cols_37_);
v_origins_38_ = lean_ctor_get(v_x_30_, 3);
lean_inc(v_origins_38_);
lean_dec_ref(v_x_30_);
v___x_39_ = lean_nat_dec_eq(v_len_31_, v_len_35_);
lean_dec(v_len_35_);
lean_dec(v_len_31_);
if (v___x_39_ == 0)
{
lean_dec(v_origins_38_);
lean_dec(v_cols_37_);
lean_dec(v_sel_36_);
lean_dec(v_origins_34_);
lean_dec(v_cols_33_);
lean_dec(v_sel_32_);
lean_dec_ref(v_inst_28_);
return v___x_39_;
}
else
{
lean_object* v___f_40_; uint8_t v___x_41_; 
v___f_40_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___closed__0));
v___x_41_ = l_Option_instDecidableEq___redArg(v___f_40_, v_sel_32_, v_sel_36_);
if (v___x_41_ == 0)
{
lean_dec(v_origins_38_);
lean_dec(v_cols_37_);
lean_dec(v_origins_34_);
lean_dec(v_cols_33_);
lean_dec_ref(v_inst_28_);
return v___x_41_;
}
else
{
lean_object* v___f_42_; lean_object* v___f_43_; uint8_t v___x_44_; 
v___f_42_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_28_);
v___f_43_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_43_, 0, v___f_42_);
v___x_44_ = l_instDecidableEqList___redArg(v___f_43_, v_cols_33_, v_cols_37_);
if (v___x_44_ == 0)
{
lean_dec(v_origins_38_);
lean_dec(v_origins_34_);
return v___x_44_;
}
else
{
uint8_t v___x_45_; 
v___x_45_ = l_Option_instDecidableEq___redArg(v___f_40_, v_origins_34_, v_origins_38_);
return v___x_45_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg___boxed(lean_object* v_inst_46_, lean_object* v_x_47_, lean_object* v_x_48_){
_start:
{
uint8_t v_res_49_; lean_object* v_r_50_; 
v_res_49_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(v_inst_46_, v_x_47_, v_x_48_);
v_r_50_ = lean_box(v_res_49_);
return v_r_50_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq(lean_object* v_V_51_, lean_object* v_inst_52_, lean_object* v_x_53_, lean_object* v_x_54_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(v_inst_52_, v_x_53_, v_x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___boxed(lean_object* v_V_56_, lean_object* v_inst_57_, lean_object* v_x_58_, lean_object* v_x_59_){
_start:
{
uint8_t v_res_60_; lean_object* v_r_61_; 
v_res_60_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq(v_V_56_, v_inst_57_, v_x_58_, v_x_59_);
v_r_61_ = lean_box(v_res_60_);
return v_r_61_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___redArg(lean_object* v_inst_62_, lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
uint8_t v___x_65_; 
v___x_65_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(v_inst_62_, v_x_63_, v_x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___redArg___boxed(lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___redArg(v_inst_66_, v_x_67_, v_x_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch(lean_object* v_V_71_, lean_object* v_inst_72_, lean_object* v_x_73_, lean_object* v_x_74_){
_start:
{
uint8_t v___x_75_; 
v___x_75_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch_decEq___redArg(v_inst_72_, v_x_73_, v_x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch___boxed(lean_object* v_V_76_, lean_object* v_inst_77_, lean_object* v_x_78_, lean_object* v_x_79_){
_start:
{
uint8_t v_res_80_; lean_object* v_r_81_; 
v_res_80_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqBatch(v_V_76_, v_inst_77_, v_x_78_, v_x_79_);
v_r_81_ = lean_box(v_res_80_);
return v_r_81_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_unsigned_to_nat(7u);
v___x_100_ = lean_nat_to_int(v___x_99_);
return v___x_100_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = lean_unsigned_to_nat(8u);
v___x_111_ = lean_nat_to_int(v___x_110_);
return v___x_111_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = lean_unsigned_to_nat(11u);
v___x_116_ = lean_nat_to_int(v___x_115_);
return v___x_116_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__2));
v___x_119_ = lean_string_length(v___x_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21, &lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21_once, _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__21);
v___x_121_ = lean_nat_to_int(v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg(lean_object* v_inst_126_, lean_object* v_x_127_){
_start:
{
lean_object* v_len_128_; lean_object* v_sel_129_; lean_object* v_cols_130_; lean_object* v_origins_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; uint8_t v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_len_128_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_len_128_);
v_sel_129_ = lean_ctor_get(v_x_127_, 1);
lean_inc(v_sel_129_);
v_cols_130_ = lean_ctor_get(v_x_127_, 2);
lean_inc(v_cols_130_);
v_origins_131_ = lean_ctor_get(v_x_127_, 3);
lean_inc(v_origins_131_);
lean_dec_ref(v_x_127_);
v___x_132_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__1));
v___x_133_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__7));
v___x_134_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__8));
v___x_135_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__9);
v___x_136_ = l_Nat_reprFast(v_len_128_);
v___x_137_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
v___x_138_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_135_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = 0;
v___x_140_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_140_, 0, v___x_138_);
lean_ctor_set_uint8(v___x_140_, sizeof(void*)*1, v___x_139_);
v___x_141_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_134_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__11));
v___x_143_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_141_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = lean_box(1);
v___x_145_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_143_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
v___x_146_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__13));
v___x_147_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_145_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v___x_133_);
v___x_149_ = lean_unsigned_to_nat(0u);
v___x_150_ = l_Option_repr___redArg(v___x_132_, v_sel_129_, v___x_149_);
v___x_151_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_135_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
v___x_152_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set_uint8(v___x_152_, sizeof(void*)*1, v___x_139_);
v___x_153_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_148_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
v___x_154_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v___x_142_);
v___x_155_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
lean_ctor_set(v___x_155_, 1, v___x_144_);
v___x_156_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__15));
v___x_157_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_155_);
lean_ctor_set(v___x_157_, 1, v___x_156_);
v___x_158_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_133_);
v___x_159_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16, &lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16_once, _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__16);
v___x_160_ = lean_alloc_closure((void*)(l_List_repr___boxed), 4, 2);
lean_closure_set(v___x_160_, 0, lean_box(0));
lean_closure_set(v___x_160_, 1, v_inst_126_);
v___x_161_ = lean_alloc_closure((void*)(l_Option_repr___boxed), 4, 2);
lean_closure_set(v___x_161_, 0, lean_box(0));
lean_closure_set(v___x_161_, 1, v___x_160_);
v___x_162_ = l_List_repr___redArg(v___x_161_, v_cols_130_);
v___x_163_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_159_);
lean_ctor_set(v___x_163_, 1, v___x_162_);
v___x_164_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set_uint8(v___x_164_, sizeof(void*)*1, v___x_139_);
v___x_165_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_158_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
v___x_166_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v___x_142_);
v___x_167_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
lean_ctor_set(v___x_167_, 1, v___x_144_);
v___x_168_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__18));
v___x_169_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_167_);
lean_ctor_set(v___x_169_, 1, v___x_168_);
v___x_170_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___x_133_);
v___x_171_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19, &lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19_once, _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__19);
v___x_172_ = l_Option_repr___redArg(v___x_132_, v_origins_131_, v___x_149_);
v___x_173_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_171_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
v___x_174_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set_uint8(v___x_174_, sizeof(void*)*1, v___x_139_);
v___x_175_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_170_);
lean_ctor_set(v___x_175_, 1, v___x_174_);
v___x_176_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22, &lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22_once, _init_lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__22);
v___x_177_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__23));
v___x_178_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_175_);
v___x_179_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg___closed__24));
v___x_180_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_178_);
lean_ctor_set(v___x_180_, 1, v___x_179_);
v___x_181_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_176_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
v___x_182_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set_uint8(v___x_182_, sizeof(void*)*1, v___x_139_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr(lean_object* v_V_183_, lean_object* v_inst_184_, lean_object* v_x_185_, lean_object* v_prec_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___redArg(v_inst_184_, v_x_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___boxed(lean_object* v_V_188_, lean_object* v_inst_189_, lean_object* v_x_190_, lean_object* v_prec_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_pr2845_x2dreview_Pr2845_instReprBatch_repr(v_V_188_, v_inst_189_, v_x_190_, v_prec_191_);
lean_dec(v_prec_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch___redArg(lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___boxed), 4, 2);
lean_closure_set(v___x_194_, 0, lean_box(0));
lean_closure_set(v___x_194_, 1, v_inst_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprBatch(lean_object* v_V_195_, lean_object* v_inst_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_instReprBatch_repr___boxed), 4, 2);
lean_closure_set(v___x_197_, 0, lean_box(0));
lean_closure_set(v___x_197_, 1, v_inst_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(lean_object* v_b_198_){
_start:
{
lean_object* v_sel_199_; 
v_sel_199_ = lean_ctor_get(v_b_198_, 1);
if (lean_obj_tag(v_sel_199_) == 0)
{
lean_object* v_len_200_; lean_object* v___x_201_; 
v_len_200_ = lean_ctor_get(v_b_198_, 0);
lean_inc(v_len_200_);
lean_dec_ref(v_b_198_);
v___x_201_ = l_List_range(v_len_200_);
return v___x_201_;
}
else
{
lean_object* v_val_202_; 
lean_inc_ref(v_sel_199_);
lean_dec_ref(v_b_198_);
v_val_202_ = lean_ctor_get(v_sel_199_, 0);
lean_inc(v_val_202_);
lean_dec_ref_known(v_sel_199_, 1);
return v_val_202_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_active(lean_object* v_V_203_, lean_object* v_b_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_b_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg(lean_object* v_b_206_, lean_object* v_r_207_){
_start:
{
lean_object* v_origins_208_; 
v_origins_208_ = lean_ctor_get(v_b_206_, 3);
if (lean_obj_tag(v_origins_208_) == 0)
{
lean_object* v___x_209_; 
lean_dec(v_r_207_);
v___x_209_ = lean_unsigned_to_nat(0u);
return v___x_209_;
}
else
{
lean_object* v_val_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v_val_210_ = lean_ctor_get(v_origins_208_, 0);
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = l_List_getD___redArg(v_val_210_, v_r_207_, v___x_211_);
return v___x_212_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg___boxed(lean_object* v_b_213_, lean_object* v_r_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg(v_b_213_, v_r_214_);
lean_dec_ref(v_b_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow(lean_object* v_V_216_, lean_object* v_b_217_, lean_object* v_r_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg(v_b_217_, v_r_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_originRow___boxed(lean_object* v_V_220_, lean_object* v_b_221_, lean_object* v_r_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_pr2845_x2dreview_Pr2845_Batch_originRow(v_V_220_, v_b_221_, v_r_222_);
lean_dec_ref(v_b_221_);
return v_res_223_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg(lean_object* v_b_224_){
_start:
{
lean_object* v_origins_225_; 
v_origins_225_ = lean_ctor_get(v_b_224_, 3);
if (lean_obj_tag(v_origins_225_) == 0)
{
uint8_t v___x_226_; 
v___x_226_ = 0;
return v___x_226_;
}
else
{
uint8_t v___x_227_; 
v___x_227_ = 1;
return v___x_227_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg___boxed(lean_object* v_b_228_){
_start:
{
uint8_t v_res_229_; lean_object* v_r_230_; 
v_res_229_ = lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg(v_b_228_);
lean_dec_ref(v_b_228_);
v_r_230_ = lean_box(v_res_229_);
return v_r_230_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins(lean_object* v_V_231_, lean_object* v_b_232_){
_start:
{
uint8_t v___x_233_; 
v___x_233_ = lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg(v_b_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___boxed(lean_object* v_V_234_, lean_object* v_b_235_){
_start:
{
uint8_t v_res_236_; lean_object* v_r_237_; 
v_res_236_ = lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins(v_V_234_, v_b_235_);
lean_dec_ref(v_b_235_);
v_r_237_ = lean_box(v_res_236_);
return v_r_237_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly___redArg(lean_object* v_n_238_){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_239_ = lean_box(0);
v___x_240_ = lean_box(0);
v___x_241_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_241_, 0, v_n_238_);
lean_ctor_set(v___x_241_, 1, v___x_239_);
lean_ctor_set(v___x_241_, 2, v___x_240_);
lean_ctor_set(v___x_241_, 3, v___x_239_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly(lean_object* v_V_242_, lean_object* v_n_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly___redArg(v_n_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows___redArg(lean_object* v_b_245_, lean_object* v_o_246_){
_start:
{
lean_object* v_len_247_; lean_object* v_sel_248_; lean_object* v_cols_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_257_; 
v_len_247_ = lean_ctor_get(v_b_245_, 0);
v_sel_248_ = lean_ctor_get(v_b_245_, 1);
v_cols_249_ = lean_ctor_get(v_b_245_, 2);
v_isSharedCheck_257_ = !lean_is_exclusive(v_b_245_);
if (v_isSharedCheck_257_ == 0)
{
lean_object* v_unused_258_; 
v_unused_258_ = lean_ctor_get(v_b_245_, 3);
lean_dec(v_unused_258_);
v___x_251_ = v_b_245_;
v_isShared_252_ = v_isSharedCheck_257_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_cols_249_);
lean_inc(v_sel_248_);
lean_inc(v_len_247_);
lean_dec(v_b_245_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_257_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_253_; lean_object* v___x_255_; 
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v_o_246_);
if (v_isShared_252_ == 0)
{
lean_ctor_set(v___x_251_, 3, v___x_253_);
v___x_255_ = v___x_251_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v_len_247_);
lean_ctor_set(v_reuseFailAlloc_256_, 1, v_sel_248_);
lean_ctor_set(v_reuseFailAlloc_256_, 2, v_cols_249_);
lean_ctor_set(v_reuseFailAlloc_256_, 3, v___x_253_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows(lean_object* v_V_259_, lean_object* v_b_260_, lean_object* v_o_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows___redArg(v_b_260_, v_o_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0(lean_object* v_c_263_, lean_object* v_inst_264_, lean_object* v_x_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = l_List_getD___redArg(v_c_263_, v_x_265_, v_inst_264_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0___boxed(lean_object* v_c_267_, lean_object* v_inst_268_, lean_object* v_x_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0(v_c_267_, v_inst_268_, v_x_269_);
lean_dec(v_inst_268_);
lean_dec(v_c_267_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__1(lean_object* v_inst_271_, lean_object* v_idx_272_, lean_object* v_c_273_){
_start:
{
lean_object* v___f_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___f_274_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_274_, 0, v_c_273_);
lean_closure_set(v___f_274_, 1, v_inst_271_);
v___x_275_ = lean_box(0);
v___x_276_ = l_List_mapTR_loop___redArg(v___f_274_, v_idx_272_, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2(lean_object* v_x_277_){
_start:
{
lean_object* v___x_278_; uint8_t v___x_279_; 
v___x_278_ = lean_unsigned_to_nat(0u);
v___x_279_ = lean_nat_dec_eq(v_x_277_, v___x_278_);
if (v___x_279_ == 0)
{
uint8_t v___x_280_; 
v___x_280_ = 1;
return v___x_280_;
}
else
{
uint8_t v___x_281_; 
v___x_281_ = 0;
return v___x_281_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2___boxed(lean_object* v_x_282_){
_start:
{
uint8_t v_res_283_; lean_object* v_r_284_; 
v_res_283_ = lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__2(v_x_282_);
lean_dec(v_x_282_);
v_r_284_ = lean_box(v_res_283_);
return v_r_284_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3(lean_object* v_val_285_, lean_object* v_x_286_){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_287_ = lean_unsigned_to_nat(0u);
v___x_288_ = l_List_getD___redArg(v_val_285_, v_x_286_, v___x_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3___boxed(lean_object* v_val_289_, lean_object* v_x_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3(v_val_289_, v_x_290_);
lean_dec(v_val_289_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg(lean_object* v_inst_293_, lean_object* v_b_294_, lean_object* v_idx_295_){
_start:
{
lean_object* v_cols_296_; lean_object* v_origins_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_328_; 
v_cols_296_ = lean_ctor_get(v_b_294_, 2);
v_origins_297_ = lean_ctor_get(v_b_294_, 3);
v_isSharedCheck_328_ = !lean_is_exclusive(v_b_294_);
if (v_isSharedCheck_328_ == 0)
{
lean_object* v_unused_329_; lean_object* v_unused_330_; 
v_unused_329_ = lean_ctor_get(v_b_294_, 1);
lean_dec(v_unused_329_);
v_unused_330_ = lean_ctor_get(v_b_294_, 0);
lean_dec(v_unused_330_);
v___x_299_ = v_b_294_;
v_isShared_300_ = v_isSharedCheck_328_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_origins_297_);
lean_inc(v_cols_296_);
lean_dec(v_b_294_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_328_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
lean_object* v___f_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; 
lean_inc(v_idx_295_);
v___f_301_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__1), 3, 2);
lean_closure_set(v___f_301_, 0, v_inst_293_);
lean_closure_set(v___f_301_, 1, v_idx_295_);
v___x_302_ = l_List_lengthTR___redArg(v_idx_295_);
v___x_303_ = lean_box(0);
v___x_304_ = lean_alloc_closure((void*)(l_Option_map), 4, 3);
lean_closure_set(v___x_304_, 0, lean_box(0));
lean_closure_set(v___x_304_, 1, lean_box(0));
lean_closure_set(v___x_304_, 2, v___f_301_);
v___x_305_ = lean_box(0);
v___x_306_ = l_List_mapTR_loop___redArg(v___x_304_, v_cols_296_, v___x_305_);
if (lean_obj_tag(v_origins_297_) == 0)
{
lean_object* v___x_308_; 
lean_dec(v_idx_295_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 2, v___x_306_);
lean_ctor_set(v___x_299_, 1, v___x_303_);
lean_ctor_set(v___x_299_, 0, v___x_302_);
v___x_308_ = v___x_299_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v___x_302_);
lean_ctor_set(v_reuseFailAlloc_309_, 1, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_309_, 2, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_309_, 3, v_origins_297_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
else
{
lean_object* v_val_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_327_; 
v_val_310_ = lean_ctor_get(v_origins_297_, 0);
v_isSharedCheck_327_ = !lean_is_exclusive(v_origins_297_);
if (v_isSharedCheck_327_ == 0)
{
v___x_312_ = v_origins_297_;
v_isShared_313_ = v_isSharedCheck_327_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_val_310_);
lean_dec(v_origins_297_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_327_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___f_314_; lean_object* v___f_315_; lean_object* v_g_316_; uint8_t v___x_317_; 
v___f_314_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___closed__0));
v___f_315_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_315_, 0, v_val_310_);
v_g_316_ = l_List_mapTR_loop___redArg(v___f_315_, v_idx_295_, v___x_305_);
lean_inc(v_g_316_);
v___x_317_ = l_List_any___redArg(v_g_316_, v___f_314_);
if (v___x_317_ == 0)
{
lean_object* v___x_319_; 
lean_dec(v_g_316_);
lean_del_object(v___x_312_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 3, v___x_303_);
lean_ctor_set(v___x_299_, 2, v___x_306_);
lean_ctor_set(v___x_299_, 1, v___x_303_);
lean_ctor_set(v___x_299_, 0, v___x_302_);
v___x_319_ = v___x_299_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_302_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_320_, 2, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_320_, 3, v___x_303_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
else
{
lean_object* v___x_322_; 
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v_g_316_);
v___x_322_ = v___x_312_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_g_316_);
v___x_322_ = v_reuseFailAlloc_326_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
lean_object* v___x_324_; 
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 3, v___x_322_);
lean_ctor_set(v___x_299_, 2, v___x_306_);
lean_ctor_set(v___x_299_, 1, v___x_303_);
lean_ctor_set(v___x_299_, 0, v___x_302_);
v___x_324_ = v___x_299_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_302_);
lean_ctor_set(v_reuseFailAlloc_325_, 1, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_325_, 2, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_325_, 3, v___x_322_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Batch_gather(lean_object* v_V_331_, lean_object* v_inst_332_, lean_object* v_b_333_, lean_object* v_idx_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg(v_inst_332_, v_b_333_, v_idx_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Batch_0__Pr2845_Batch_originRow_match__1_splitter___redArg(lean_object* v_x_336_, lean_object* v_h__1_337_, lean_object* v_h__2_338_){
_start:
{
if (lean_obj_tag(v_x_336_) == 0)
{
lean_object* v___x_339_; lean_object* v___x_340_; 
lean_dec(v_h__2_338_);
v___x_339_ = lean_box(0);
v___x_340_ = lean_apply_1(v_h__1_337_, v___x_339_);
return v___x_340_;
}
else
{
lean_object* v_val_341_; lean_object* v___x_342_; 
lean_dec(v_h__1_337_);
v_val_341_ = lean_ctor_get(v_x_336_, 0);
lean_inc(v_val_341_);
lean_dec_ref_known(v_x_336_, 1);
v___x_342_ = lean_apply_1(v_h__2_338_, v_val_341_);
return v___x_342_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Batch_0__Pr2845_Batch_originRow_match__1_splitter(lean_object* v_motive_343_, lean_object* v_x_344_, lean_object* v_h__1_345_, lean_object* v_h__2_346_){
_start:
{
if (lean_obj_tag(v_x_344_) == 0)
{
lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v_h__2_346_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_apply_1(v_h__1_345_, v___x_347_);
return v___x_348_;
}
else
{
lean_object* v_val_349_; lean_object* v___x_350_; 
lean_dec(v_h__1_345_);
v_val_349_ = lean_ctor_get(v_x_344_, 0);
lean_inc(v_val_349_);
lean_dec_ref_known(v_x_344_, 1);
v___x_350_ = lean_apply_1(v_h__2_346_, v_val_349_);
return v___x_350_;
}
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0(lean_object* v_x_351_){
_start:
{
if (lean_obj_tag(v_x_351_) == 0)
{
uint8_t v___x_352_; 
v___x_352_ = 0;
return v___x_352_;
}
else
{
lean_object* v_head_353_; lean_object* v_tail_354_; lean_object* v___x_355_; uint8_t v___x_356_; 
v_head_353_ = lean_ctor_get(v_x_351_, 0);
v_tail_354_ = lean_ctor_get(v_x_351_, 1);
v___x_355_ = lean_unsigned_to_nat(0u);
v___x_356_ = lean_nat_dec_eq(v_head_353_, v___x_355_);
if (v___x_356_ == 0)
{
uint8_t v___x_357_; 
v___x_357_ = 1;
return v___x_357_;
}
else
{
v_x_351_ = v_tail_354_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0___boxed(lean_object* v_x_359_){
_start:
{
uint8_t v_res_360_; lean_object* v_r_361_; 
v_res_360_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0(v_x_359_);
lean_dec(v_x_359_);
v_r_361_ = lean_box(v_res_360_);
return v_r_361_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg(lean_object* v_origins_366_){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_367_ = l_List_lengthTR___redArg(v_origins_366_);
v___x_368_ = lean_unsigned_to_nat(0u);
v___x_369_ = lean_nat_dec_eq(v___x_367_, v___x_368_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v___x_370_ = lean_box(0);
v___x_371_ = lean_box(0);
v___x_372_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0(v_origins_366_);
if (v___x_372_ == 0)
{
lean_object* v___x_373_; 
lean_dec(v_origins_366_);
v___x_373_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_373_, 0, v___x_367_);
lean_ctor_set(v___x_373_, 1, v___x_370_);
lean_ctor_set(v___x_373_, 2, v___x_371_);
lean_ctor_set(v___x_373_, 3, v___x_370_);
return v___x_373_;
}
else
{
lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_374_, 0, v_origins_366_);
v___x_375_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_375_, 0, v___x_367_);
lean_ctor_set(v___x_375_, 1, v___x_370_);
lean_ctor_set(v___x_375_, 2, v___x_371_);
lean_ctor_set(v___x_375_, 3, v___x_374_);
return v___x_375_;
}
}
else
{
lean_object* v___x_376_; 
lean_dec(v___x_367_);
lean_dec(v_origins_366_);
v___x_376_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg___closed__0));
return v___x_376_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows(lean_object* v_V_377_, lean_object* v_origins_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg(v_origins_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(lean_object* v_b_380_, lean_object* v_a_381_, lean_object* v_a_382_){
_start:
{
if (lean_obj_tag(v_a_381_) == 0)
{
lean_object* v___x_383_; 
v___x_383_ = l_List_reverse___redArg(v_a_382_);
return v___x_383_;
}
else
{
lean_object* v_head_384_; lean_object* v_tail_385_; lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_394_; 
v_head_384_ = lean_ctor_get(v_a_381_, 0);
v_tail_385_ = lean_ctor_get(v_a_381_, 1);
v_isSharedCheck_394_ = !lean_is_exclusive(v_a_381_);
if (v_isSharedCheck_394_ == 0)
{
v___x_387_ = v_a_381_;
v_isShared_388_ = v_isSharedCheck_394_;
goto v_resetjp_386_;
}
else
{
lean_inc(v_tail_385_);
lean_inc(v_head_384_);
lean_dec(v_a_381_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_394_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
lean_object* v___x_389_; lean_object* v___x_391_; 
v___x_389_ = lp_pr2845_x2dreview_Pr2845_Batch_originRow___redArg(v_b_380_, v_head_384_);
if (v_isShared_388_ == 0)
{
lean_ctor_set(v___x_387_, 1, v_a_382_);
lean_ctor_set(v___x_387_, 0, v___x_389_);
v___x_391_ = v___x_387_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_389_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_a_382_);
v___x_391_ = v_reuseFailAlloc_393_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
v_a_381_ = v_tail_385_;
v_a_382_ = v___x_391_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg___boxed(lean_object* v_b_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(v_b_395_, v_a_396_, v_a_397_);
lean_dec_ref(v_b_395_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyGeneral___redArg(lean_object* v_b_399_){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
lean_inc_ref(v_b_399_);
v___x_400_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_b_399_);
v___x_401_ = lean_box(0);
v___x_402_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(v_b_399_, v___x_400_, v___x_401_);
lean_dec_ref(v_b_399_);
v___x_403_ = lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg(v___x_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyGeneral(lean_object* v_V_404_, lean_object* v_b_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_pr2845_x2dreview_Pr2845_projectEmptyGeneral___redArg(v_b_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0(lean_object* v_V_407_, lean_object* v_b_408_, lean_object* v_a_409_, lean_object* v_a_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(v_b_408_, v_a_409_, v_a_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___boxed(lean_object* v_V_412_, lean_object* v_b_413_, lean_object* v_a_414_, lean_object* v_a_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0(v_V_412_, v_b_413_, v_a_414_, v_a_415_);
lean_dec_ref(v_b_413_);
return v_res_416_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0(lean_object* v_x_417_){
_start:
{
if (lean_obj_tag(v_x_417_) == 0)
{
uint8_t v___x_418_; 
v___x_418_ = 0;
return v___x_418_;
}
else
{
lean_object* v_head_419_; lean_object* v_tail_420_; lean_object* v___x_421_; uint8_t v___x_422_; 
v_head_419_ = lean_ctor_get(v_x_417_, 0);
v_tail_420_ = lean_ctor_get(v_x_417_, 1);
v___x_421_ = lean_unsigned_to_nat(0u);
v___x_422_ = lean_nat_dec_eq(v_head_419_, v___x_421_);
if (v___x_422_ == 0)
{
uint8_t v___x_423_; 
v___x_423_ = 1;
return v___x_423_;
}
else
{
v_x_417_ = v_tail_420_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0___boxed(lean_object* v_x_425_){
_start:
{
uint8_t v_res_426_; lean_object* v_r_427_; 
v_res_426_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0(v_x_425_);
lean_dec(v_x_425_);
v_r_427_ = lean_box(v_res_426_);
return v_r_427_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyFast___redArg(lean_object* v_b_428_){
_start:
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v_os_431_; lean_object* v___x_432_; lean_object* v_out_433_; uint8_t v___x_434_; 
lean_inc_ref(v_b_428_);
v___x_429_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_b_428_);
v___x_430_ = lean_box(0);
lean_inc(v___x_429_);
v_os_431_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(v_b_428_, v___x_429_, v___x_430_);
lean_dec_ref(v_b_428_);
v___x_432_ = l_List_lengthTR___redArg(v___x_429_);
lean_dec(v___x_429_);
v_out_433_ = lp_pr2845_x2dreview_Pr2845_Batch_rowsOnly___redArg(v___x_432_);
v___x_434_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_projectEmptyFast_spec__0(v_os_431_);
if (v___x_434_ == 0)
{
lean_dec(v_os_431_);
return v_out_433_;
}
else
{
lean_object* v___x_435_; 
v___x_435_ = lp_pr2845_x2dreview_Pr2845_Batch_setOriginRows___redArg(v_out_433_, v_os_431_);
return v___x_435_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectEmptyFast(lean_object* v_V_436_, lean_object* v_b_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_pr2845_x2dreview_Pr2845_projectEmptyFast___redArg(v_b_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMain___redArg(lean_object* v_fastCols_439_, lean_object* v_general_440_, lean_object* v_trees_441_, lean_object* v_copies_442_, lean_object* v_b_443_){
_start:
{
lean_object* v___x_444_; uint8_t v___x_445_; 
lean_inc_ref(v_b_443_);
v___x_444_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_b_443_);
v___x_445_ = l_List_instDecidableEqNil___redArg(v___x_444_);
lean_dec(v___x_444_);
if (v___x_445_ == 0)
{
uint8_t v___x_446_; 
v___x_446_ = l_List_instDecidableEqNil___redArg(v_trees_441_);
if (v___x_446_ == 0)
{
uint8_t v___x_447_; 
v___x_447_ = l_List_instDecidableEqNil___redArg(v_copies_442_);
if (v___x_447_ == 0)
{
lean_object* v___x_448_; 
lean_dec_ref(v_fastCols_439_);
v___x_448_ = lean_apply_3(v_general_440_, v_trees_441_, v_copies_442_, v_b_443_);
return v___x_448_;
}
else
{
lean_object* v___x_449_; 
lean_dec_ref(v_general_440_);
v___x_449_ = lean_apply_3(v_fastCols_439_, v_trees_441_, v_copies_442_, v_b_443_);
return v___x_449_;
}
}
else
{
lean_object* v___x_450_; 
lean_dec_ref(v_fastCols_439_);
v___x_450_ = lean_apply_3(v_general_440_, v_trees_441_, v_copies_442_, v_b_443_);
return v___x_450_;
}
}
else
{
lean_object* v___x_451_; 
lean_dec_ref(v_fastCols_439_);
v___x_451_ = lean_apply_3(v_general_440_, v_trees_441_, v_copies_442_, v_b_443_);
return v___x_451_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMain(lean_object* v_T_452_, lean_object* v_C_453_, lean_object* v_V_454_, lean_object* v_fastCols_455_, lean_object* v_general_456_, lean_object* v_trees_457_, lean_object* v_copies_458_, lean_object* v_b_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_pr2845_x2dreview_Pr2845_projectMain___redArg(v_fastCols_455_, v_general_456_, v_trees_457_, v_copies_458_, v_b_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMerged___redArg(lean_object* v_fastCols_461_, lean_object* v_general_462_, lean_object* v_trees_463_, lean_object* v_copies_464_, lean_object* v_b_465_){
_start:
{
uint8_t v___x_466_; uint8_t v___x_467_; lean_object* v___x_472_; uint8_t v___x_473_; 
v___x_466_ = l_List_instDecidableEqNil___redArg(v_copies_464_);
v___x_467_ = l_List_instDecidableEqNil___redArg(v_trees_463_);
lean_inc_ref(v_b_465_);
v___x_472_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_b_465_);
v___x_473_ = l_List_instDecidableEqNil___redArg(v___x_472_);
lean_dec(v___x_472_);
if (v___x_473_ == 0)
{
if (v___x_467_ == 0)
{
if (v___x_466_ == 0)
{
lean_dec_ref(v_fastCols_461_);
goto v___jp_468_;
}
else
{
lean_object* v___x_474_; 
lean_dec_ref(v_general_462_);
v___x_474_ = lean_apply_3(v_fastCols_461_, v_trees_463_, v_copies_464_, v_b_465_);
return v___x_474_;
}
}
else
{
lean_dec_ref(v_fastCols_461_);
goto v___jp_468_;
}
}
else
{
lean_dec_ref(v_fastCols_461_);
goto v___jp_468_;
}
v___jp_468_:
{
if (v___x_467_ == 0)
{
lean_object* v___x_469_; 
v___x_469_ = lean_apply_3(v_general_462_, v_trees_463_, v_copies_464_, v_b_465_);
return v___x_469_;
}
else
{
if (v___x_466_ == 0)
{
lean_object* v___x_470_; 
v___x_470_ = lean_apply_3(v_general_462_, v_trees_463_, v_copies_464_, v_b_465_);
return v___x_470_;
}
else
{
lean_object* v___x_471_; 
lean_dec(v_copies_464_);
lean_dec(v_trees_463_);
lean_dec_ref(v_general_462_);
v___x_471_ = lp_pr2845_x2dreview_Pr2845_projectEmptyFast___redArg(v_b_465_);
return v___x_471_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectMerged(lean_object* v_T_475_, lean_object* v_C_476_, lean_object* v_V_477_, lean_object* v_fastCols_478_, lean_object* v_general_479_, lean_object* v_trees_480_, lean_object* v_copies_481_, lean_object* v_b_482_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lp_pr2845_x2dreview_Pr2845_projectMerged___redArg(v_fastCols_478_, v_general_479_, v_trees_480_, v_copies_481_, v_b_482_);
return v___x_483_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg(lean_object* v_b_484_){
_start:
{
lean_object* v_cols_485_; lean_object* v___x_486_; lean_object* v___x_487_; uint8_t v___x_488_; 
v_cols_485_ = lean_ctor_get(v_b_484_, 2);
v___x_486_ = lean_unsigned_to_nat(0u);
v___x_487_ = l_List_lengthTR___redArg(v_cols_485_);
v___x_488_ = lean_nat_dec_lt(v___x_486_, v___x_487_);
lean_dec(v___x_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg___boxed(lean_object* v_b_489_){
_start:
{
uint8_t v_res_490_; lean_object* v_r_491_; 
v_res_490_ = lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg(v_b_489_);
lean_dec_ref(v_b_489_);
v_r_491_ = lean_box(v_res_490_);
return v_r_491_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandOld(lean_object* v_V_492_, lean_object* v_b_493_){
_start:
{
uint8_t v___x_494_; 
v___x_494_ = lp_pr2845_x2dreview_Pr2845_shouldExpandOld___redArg(v_b_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandOld___boxed(lean_object* v_V_495_, lean_object* v_b_496_){
_start:
{
uint8_t v_res_497_; lean_object* v_r_498_; 
v_res_497_ = lp_pr2845_x2dreview_Pr2845_shouldExpandOld(v_V_495_, v_b_496_);
lean_dec_ref(v_b_496_);
v_r_498_ = lean_box(v_res_497_);
return v_r_498_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg(lean_object* v_b_499_){
_start:
{
lean_object* v_cols_500_; lean_object* v___x_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v_cols_500_ = lean_ctor_get(v_b_499_, 2);
v___x_501_ = lean_unsigned_to_nat(0u);
v___x_502_ = l_List_lengthTR___redArg(v_cols_500_);
v___x_503_ = lean_nat_dec_lt(v___x_501_, v___x_502_);
lean_dec(v___x_502_);
if (v___x_503_ == 0)
{
uint8_t v___x_504_; 
v___x_504_ = lp_pr2845_x2dreview_Pr2845_Batch_hasOrigins___redArg(v_b_499_);
return v___x_504_;
}
else
{
return v___x_503_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg___boxed(lean_object* v_b_505_){
_start:
{
uint8_t v_res_506_; lean_object* v_r_507_; 
v_res_506_ = lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg(v_b_505_);
lean_dec_ref(v_b_505_);
v_r_507_ = lean_box(v_res_506_);
return v_r_507_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_shouldExpandNew(lean_object* v_V_508_, lean_object* v_b_509_){
_start:
{
uint8_t v___x_510_; 
v___x_510_ = lp_pr2845_x2dreview_Pr2845_shouldExpandNew___redArg(v_b_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_shouldExpandNew___boxed(lean_object* v_V_511_, lean_object* v_b_512_){
_start:
{
uint8_t v_res_513_; lean_object* v_r_514_; 
v_res_513_ = lp_pr2845_x2dreview_Pr2845_shouldExpandNew(v_V_511_, v_b_512_);
lean_dec_ref(v_b_512_);
v_r_514_ = lean_box(v_res_513_);
return v_r_514_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___redArg(lean_object* v_inst_515_, uint8_t v_expand_516_, lean_object* v_parent_517_, lean_object* v_idx_518_, lean_object* v_lane_519_){
_start:
{
lean_object* v_sel_521_; lean_object* v_cols_522_; lean_object* v_origins_523_; lean_object* v___y_524_; lean_object* v___y_530_; 
if (v_expand_516_ == 0)
{
lean_object* v___x_539_; 
lean_dec(v_idx_518_);
lean_dec_ref(v_parent_517_);
lean_dec(v_inst_515_);
v___x_539_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_builderFinishEmptyRows___redArg___closed__0));
v___y_530_ = v___x_539_;
goto v___jp_529_;
}
else
{
lean_object* v___x_540_; 
v___x_540_ = lp_pr2845_x2dreview_Pr2845_Batch_gather___redArg(v_inst_515_, v_parent_517_, v_idx_518_);
v___y_530_ = v___x_540_;
goto v___jp_529_;
}
v___jp_520_:
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_525_ = lean_box(0);
v___x_526_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_526_, 0, v_lane_519_);
lean_ctor_set(v___x_526_, 1, v___x_525_);
v___x_527_ = l_List_appendTR___redArg(v_cols_522_, v___x_526_);
v___x_528_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_528_, 0, v___y_524_);
lean_ctor_set(v___x_528_, 1, v_sel_521_);
lean_ctor_set(v___x_528_, 2, v___x_527_);
lean_ctor_set(v___x_528_, 3, v_origins_523_);
return v___x_528_;
}
v___jp_529_:
{
if (lean_obj_tag(v_lane_519_) == 0)
{
return v___y_530_;
}
else
{
lean_object* v_val_531_; lean_object* v_len_532_; lean_object* v_sel_533_; lean_object* v_cols_534_; lean_object* v_origins_535_; lean_object* v___x_536_; uint8_t v___x_537_; 
v_val_531_ = lean_ctor_get(v_lane_519_, 0);
v_len_532_ = lean_ctor_get(v___y_530_, 0);
lean_inc(v_len_532_);
v_sel_533_ = lean_ctor_get(v___y_530_, 1);
lean_inc(v_sel_533_);
v_cols_534_ = lean_ctor_get(v___y_530_, 2);
lean_inc(v_cols_534_);
v_origins_535_ = lean_ctor_get(v___y_530_, 3);
lean_inc(v_origins_535_);
lean_dec_ref(v___y_530_);
v___x_536_ = lean_unsigned_to_nat(0u);
v___x_537_ = lean_nat_dec_eq(v_len_532_, v___x_536_);
if (v___x_537_ == 0)
{
v_sel_521_ = v_sel_533_;
v_cols_522_ = v_cols_534_;
v_origins_523_ = v_origins_535_;
v___y_524_ = v_len_532_;
goto v___jp_520_;
}
else
{
lean_object* v___x_538_; 
lean_dec(v_len_532_);
v___x_538_ = l_List_lengthTR___redArg(v_val_531_);
v_sel_521_ = v_sel_533_;
v_cols_522_ = v_cols_534_;
v_origins_523_ = v_origins_535_;
v___y_524_ = v___x_538_;
goto v___jp_520_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___redArg___boxed(lean_object* v_inst_541_, lean_object* v_expand_542_, lean_object* v_parent_543_, lean_object* v_idx_544_, lean_object* v_lane_545_){
_start:
{
uint8_t v_expand_boxed_546_; lean_object* v_res_547_; 
v_expand_boxed_546_ = lean_unbox(v_expand_542_);
v_res_547_ = lp_pr2845_x2dreview_Pr2845_finishBatch___redArg(v_inst_541_, v_expand_boxed_546_, v_parent_543_, v_idx_544_, v_lane_545_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch(lean_object* v_V_548_, lean_object* v_inst_549_, uint8_t v_expand_550_, lean_object* v_parent_551_, lean_object* v_idx_552_, lean_object* v_lane_553_){
_start:
{
lean_object* v___x_554_; 
v___x_554_ = lp_pr2845_x2dreview_Pr2845_finishBatch___redArg(v_inst_549_, v_expand_550_, v_parent_551_, v_idx_552_, v_lane_553_);
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_finishBatch___boxed(lean_object* v_V_555_, lean_object* v_inst_556_, lean_object* v_expand_557_, lean_object* v_parent_558_, lean_object* v_idx_559_, lean_object* v_lane_560_){
_start:
{
uint8_t v_expand_boxed_561_; lean_object* v_res_562_; 
v_expand_boxed_561_ = lean_unbox(v_expand_557_);
v_res_562_ = lp_pr2845_x2dreview_Pr2845_finishBatch(v_V_555_, v_inst_556_, v_expand_boxed_561_, v_parent_558_, v_idx_559_, v_lane_560_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1(lean_object* v_init_580_, lean_object* v_x_581_){
_start:
{
if (lean_obj_tag(v_x_581_) == 0)
{
lean_inc(v_init_580_);
return v_init_580_;
}
else
{
lean_object* v_head_582_; lean_object* v_tail_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v_head_582_ = lean_ctor_get(v_x_581_, 0);
v_tail_583_ = lean_ctor_get(v_x_581_, 1);
v___x_584_ = lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1(v_init_580_, v_tail_583_);
v___x_585_ = lean_nat_add(v_head_582_, v___x_584_);
lean_dec(v___x_584_);
return v___x_585_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1___boxed(lean_object* v_init_586_, lean_object* v_x_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1(v_init_586_, v_x_587_);
lean_dec(v_x_587_);
lean_dec(v_init_586_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1(lean_object* v_l_589_){
_start:
{
lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_590_ = lean_unsigned_to_nat(0u);
v___x_591_ = lp_pr2845_x2dreview_List_foldr___at___00List_sum___at___00Pr2845_concatColless_spec__1_spec__1(v___x_590_, v_l_589_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1___boxed(lean_object* v_l_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1(v_l_592_);
lean_dec(v_l_592_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2___redArg(lean_object* v_a_594_, lean_object* v_a_595_){
_start:
{
if (lean_obj_tag(v_a_594_) == 0)
{
lean_object* v___x_596_; 
v___x_596_ = lean_array_to_list(v_a_595_);
return v___x_596_;
}
else
{
lean_object* v_head_597_; lean_object* v_tail_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v_head_597_ = lean_ctor_get(v_a_594_, 0);
lean_inc_n(v_head_597_, 2);
v_tail_598_ = lean_ctor_get(v_a_594_, 1);
lean_inc(v_tail_598_);
lean_dec_ref_known(v_a_594_, 2);
v___x_599_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_head_597_);
v___x_600_ = lean_box(0);
v___x_601_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_projectEmptyGeneral_spec__0___redArg(v_head_597_, v___x_599_, v___x_600_);
lean_dec(v_head_597_);
v___x_602_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_595_, v___x_601_);
v_a_594_ = v_tail_598_;
v_a_595_ = v___x_602_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0___redArg(lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
if (lean_obj_tag(v_a_604_) == 0)
{
lean_object* v___x_606_; 
v___x_606_ = l_List_reverse___redArg(v_a_605_);
return v___x_606_;
}
else
{
lean_object* v_head_607_; lean_object* v_tail_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_618_; 
v_head_607_ = lean_ctor_get(v_a_604_, 0);
v_tail_608_ = lean_ctor_get(v_a_604_, 1);
v_isSharedCheck_618_ = !lean_is_exclusive(v_a_604_);
if (v_isSharedCheck_618_ == 0)
{
v___x_610_ = v_a_604_;
v_isShared_611_ = v_isSharedCheck_618_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_tail_608_);
lean_inc(v_head_607_);
lean_dec(v_a_604_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_618_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_615_; 
v___x_612_ = lp_pr2845_x2dreview_Pr2845_Batch_active___redArg(v_head_607_);
v___x_613_ = l_List_lengthTR___redArg(v___x_612_);
lean_dec(v___x_612_);
if (v_isShared_611_ == 0)
{
lean_ctor_set(v___x_610_, 1, v_a_605_);
lean_ctor_set(v___x_610_, 0, v___x_613_);
v___x_615_ = v___x_610_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v___x_613_);
lean_ctor_set(v_reuseFailAlloc_617_, 1, v_a_605_);
v___x_615_ = v_reuseFailAlloc_617_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
v_a_604_ = v_tail_608_;
v_a_605_ = v___x_615_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_concatColless___redArg(lean_object* v_bs_625_){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v_total_629_; uint8_t v___x_630_; 
v___x_626_ = lean_unsigned_to_nat(0u);
v___x_627_ = lean_box(0);
lean_inc(v_bs_625_);
v___x_628_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0___redArg(v_bs_625_, v___x_627_);
v_total_629_ = lp_pr2845_x2dreview_List_sum___at___00Pr2845_concatColless_spec__1(v___x_628_);
lean_dec(v___x_628_);
v___x_630_ = lean_nat_dec_eq(v_total_629_, v___x_626_);
if (v___x_630_ == 0)
{
lean_object* v___x_631_; lean_object* v_os_632_; lean_object* v___x_633_; uint8_t v___x_634_; 
v___x_631_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__0));
v_os_632_ = lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2___redArg(v_bs_625_, v___x_631_);
v___x_633_ = lean_box(0);
v___x_634_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_builderFinishEmptyRows_spec__0(v_os_632_);
if (v___x_634_ == 0)
{
lean_object* v___x_635_; 
lean_dec(v_os_632_);
v___x_635_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_635_, 0, v_total_629_);
lean_ctor_set(v___x_635_, 1, v___x_633_);
lean_ctor_set(v___x_635_, 2, v___x_627_);
lean_ctor_set(v___x_635_, 3, v___x_633_);
return v___x_635_;
}
else
{
lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_636_, 0, v_os_632_);
v___x_637_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_637_, 0, v_total_629_);
lean_ctor_set(v___x_637_, 1, v___x_633_);
lean_ctor_set(v___x_637_, 2, v___x_627_);
lean_ctor_set(v___x_637_, 3, v___x_636_);
return v___x_637_;
}
}
else
{
lean_object* v___x_638_; 
lean_dec(v_total_629_);
lean_dec(v_bs_625_);
v___x_638_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_concatColless___redArg___closed__1));
return v___x_638_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_concatColless(lean_object* v_V_639_, lean_object* v_bs_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_pr2845_x2dreview_Pr2845_concatColless___redArg(v_bs_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0(lean_object* v_V_642_, lean_object* v_a_643_, lean_object* v_a_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_concatColless_spec__0___redArg(v_a_643_, v_a_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2(lean_object* v_V_646_, lean_object* v_a_647_, lean_object* v_a_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_pr2845_x2dreview___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Pr2845_concatColless_spec__2___redArg(v_a_647_, v_a_648_);
return v___x_649_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_pr2845_x2dreview_Pr2845Review_Batch(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
