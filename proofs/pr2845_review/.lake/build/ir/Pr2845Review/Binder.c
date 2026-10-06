// Lean compiler output
// Module: Pr2845Review.Binder
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
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Std_Format_joinSep___at___00Lean_Syntax_formatStxAux_spec__2(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Ty_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqTy(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqTy___boxed(lean_object*, lean_object*);
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Pr2845.Ty.any"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__1_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Pr2845.Ty.node"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__2_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__3_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Pr2845.Ty.rel"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__4_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__4_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__5 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__5_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Pr2845.Ty.path"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__6 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__6_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__6_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__7 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__7_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instReprTy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_instReprTy_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy___closed__0_value;
LEAN_EXPORT const lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprTy___closed__0_value;
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqVar(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqVar___boxed(lean_object*, lean_object*);
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__0_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__1_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__1_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__2_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__3_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__4_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__4_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__5 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__5_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__3_value),((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__5_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__6 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__6_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__8 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__8_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__8_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__9 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__9_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__10 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__10_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__10_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__11 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__11_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "scope"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__13 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__13_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__13_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__14 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__14_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ty"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__16 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__16_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__16_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__17 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__17_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__18 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__18_value;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19;
static lean_once_cell_t lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__21 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__21_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__18_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__22 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__22_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instReprVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_instReprVar_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar___closed__0_value;
LEAN_EXPORT const lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar___closed__0_value;
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Env_has(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_has___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_insert_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_insert(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_ids_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_ids(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_freshVar(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_freshVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectName(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectName___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_importProj(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_with___00elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_with___00elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_other_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_other_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___closed__0_value;
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause___boxed(lean_object*, lean_object*);
static const lean_string_object lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__1_value;
static const lean_string_object lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3;
static lean_once_cell_t lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4;
static const lean_ctor_object lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__5 = (const lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__5_value;
static const lean_ctor_object lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__6 = (const lean_object*)&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__2_value;
static const lean_string_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__3_value;
static lean_once_cell_t lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4;
static lean_once_cell_t lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5;
static const lean_ctor_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__6 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__6_value;
static const lean_ctor_object lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__3_value)}};
static const lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Pr2845.Clause.with_"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__0_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__0_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__1_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__2_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Pr2845.Clause.other"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__3_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__3_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__4_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__4_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__5 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__5_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_pr2845_x2dreview_Pr2845_instReprClause___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_pr2845_x2dreview_Pr2845_instReprClause_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause___closed__0_value;
LEAN_EXPORT const lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_instReprClause___closed__0_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyNew(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyOld(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchScope(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchScope___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchNew(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchNew___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__5_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__5_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__3_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__3_splitter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__1_splitter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_relabel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_relabel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_outerLabels___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__0_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_outerLabels___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "A"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels___closed__1 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__1_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_outerLabels___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels___closed__2 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__2_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_outerLabels___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__0_value),((lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__2_value)}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels___closed__3 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__3_value;
static const lean_ctor_object lp_pr2845_x2dreview_Pr2845_outerLabels___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels___closed__4 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__4_value;
LEAN_EXPORT const lean_object* lp_pr2845_x2dreview_Pr2845_outerLabels = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_outerLabels___closed__4_value;
static const lean_string_object lp_pr2845_x2dreview_Pr2845_branchQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "q"};
static const lean_object* lp_pr2845_x2dreview_Pr2845_branchQ___closed__0 = (const lean_object*)&lp_pr2845_x2dreview_Pr2845_branchQ___closed__0_value;
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchQ(uint8_t);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchQ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
default: 
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx___boxed(lean_object* v_x_6_){
_start:
{
uint8_t v_x_boxed_7_; lean_object* v_res_8_; 
v_x_boxed_7_ = lean_unbox(v_x_6_);
v_res_8_ = lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx(v_x_boxed_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___redArg(lean_object* v_k_9_){
_start:
{
lean_inc(v_k_9_);
return v_k_9_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___redArg___boxed(lean_object* v_k_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___redArg(v_k_10_);
lean_dec(v_k_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim(lean_object* v_motive_12_, lean_object* v_ctorIdx_13_, uint8_t v_t_14_, lean_object* v_h_15_, lean_object* v_k_16_){
_start:
{
lean_inc(v_k_16_);
return v_k_16_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ctorElim___boxed(lean_object* v_motive_17_, lean_object* v_ctorIdx_18_, lean_object* v_t_19_, lean_object* v_h_20_, lean_object* v_k_21_){
_start:
{
uint8_t v_t_boxed_22_; lean_object* v_res_23_; 
v_t_boxed_22_ = lean_unbox(v_t_19_);
v_res_23_ = lp_pr2845_x2dreview_Pr2845_Ty_ctorElim(v_motive_17_, v_ctorIdx_18_, v_t_boxed_22_, v_h_20_, v_k_21_);
lean_dec(v_k_21_);
lean_dec(v_ctorIdx_18_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___redArg(lean_object* v_any_24_){
_start:
{
lean_inc(v_any_24_);
return v_any_24_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___redArg___boxed(lean_object* v_any_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_pr2845_x2dreview_Pr2845_Ty_any_elim___redArg(v_any_25_);
lean_dec(v_any_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim(lean_object* v_motive_27_, uint8_t v_t_28_, lean_object* v_h_29_, lean_object* v_any_30_){
_start:
{
lean_inc(v_any_30_);
return v_any_30_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_any_elim___boxed(lean_object* v_motive_31_, lean_object* v_t_32_, lean_object* v_h_33_, lean_object* v_any_34_){
_start:
{
uint8_t v_t_boxed_35_; lean_object* v_res_36_; 
v_t_boxed_35_ = lean_unbox(v_t_32_);
v_res_36_ = lp_pr2845_x2dreview_Pr2845_Ty_any_elim(v_motive_31_, v_t_boxed_35_, v_h_33_, v_any_34_);
lean_dec(v_any_34_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___redArg(lean_object* v_node_37_){
_start:
{
lean_inc(v_node_37_);
return v_node_37_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___redArg___boxed(lean_object* v_node_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_pr2845_x2dreview_Pr2845_Ty_node_elim___redArg(v_node_38_);
lean_dec(v_node_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim(lean_object* v_motive_40_, uint8_t v_t_41_, lean_object* v_h_42_, lean_object* v_node_43_){
_start:
{
lean_inc(v_node_43_);
return v_node_43_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_node_elim___boxed(lean_object* v_motive_44_, lean_object* v_t_45_, lean_object* v_h_46_, lean_object* v_node_47_){
_start:
{
uint8_t v_t_boxed_48_; lean_object* v_res_49_; 
v_t_boxed_48_ = lean_unbox(v_t_45_);
v_res_49_ = lp_pr2845_x2dreview_Pr2845_Ty_node_elim(v_motive_44_, v_t_boxed_48_, v_h_46_, v_node_47_);
lean_dec(v_node_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___redArg(lean_object* v_rel_50_){
_start:
{
lean_inc(v_rel_50_);
return v_rel_50_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___redArg___boxed(lean_object* v_rel_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___redArg(v_rel_51_);
lean_dec(v_rel_51_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim(lean_object* v_motive_53_, uint8_t v_t_54_, lean_object* v_h_55_, lean_object* v_rel_56_){
_start:
{
lean_inc(v_rel_56_);
return v_rel_56_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_rel_elim___boxed(lean_object* v_motive_57_, lean_object* v_t_58_, lean_object* v_h_59_, lean_object* v_rel_60_){
_start:
{
uint8_t v_t_boxed_61_; lean_object* v_res_62_; 
v_t_boxed_61_ = lean_unbox(v_t_58_);
v_res_62_ = lp_pr2845_x2dreview_Pr2845_Ty_rel_elim(v_motive_57_, v_t_boxed_61_, v_h_59_, v_rel_60_);
lean_dec(v_rel_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___redArg(lean_object* v_path_63_){
_start:
{
lean_inc(v_path_63_);
return v_path_63_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___redArg___boxed(lean_object* v_path_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_pr2845_x2dreview_Pr2845_Ty_path_elim___redArg(v_path_64_);
lean_dec(v_path_64_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim(lean_object* v_motive_66_, uint8_t v_t_67_, lean_object* v_h_68_, lean_object* v_path_69_){
_start:
{
lean_inc(v_path_69_);
return v_path_69_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_path_elim___boxed(lean_object* v_motive_70_, lean_object* v_t_71_, lean_object* v_h_72_, lean_object* v_path_73_){
_start:
{
uint8_t v_t_boxed_74_; lean_object* v_res_75_; 
v_t_boxed_74_ = lean_unbox(v_t_71_);
v_res_75_ = lp_pr2845_x2dreview_Pr2845_Ty_path_elim(v_motive_70_, v_t_boxed_74_, v_h_72_, v_path_73_);
lean_dec(v_path_73_);
return v_res_75_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Ty_ofNat(lean_object* v_n_76_){
_start:
{
lean_object* v___x_77_; uint8_t v___x_78_; 
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_nat_dec_le(v_n_76_, v___x_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = lean_unsigned_to_nat(2u);
v___x_80_ = lean_nat_dec_le(v_n_76_, v___x_79_);
if (v___x_80_ == 0)
{
uint8_t v___x_81_; 
v___x_81_ = 3;
return v___x_81_;
}
else
{
uint8_t v___x_82_; 
v___x_82_ = 2;
return v___x_82_;
}
}
else
{
lean_object* v___x_83_; uint8_t v___x_84_; 
v___x_83_ = lean_unsigned_to_nat(0u);
v___x_84_ = lean_nat_dec_le(v_n_76_, v___x_83_);
if (v___x_84_ == 0)
{
uint8_t v___x_85_; 
v___x_85_ = 1;
return v___x_85_;
}
else
{
uint8_t v___x_86_; 
v___x_86_ = 0;
return v___x_86_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Ty_ofNat___boxed(lean_object* v_n_87_){
_start:
{
uint8_t v_res_88_; lean_object* v_r_89_; 
v_res_88_ = lp_pr2845_x2dreview_Pr2845_Ty_ofNat(v_n_87_);
lean_dec(v_n_87_);
v_r_89_ = lean_box(v_res_88_);
return v_r_89_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqTy(uint8_t v_x_90_, uint8_t v_y_91_){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; 
v___x_92_ = lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx(v_x_90_);
v___x_93_ = lp_pr2845_x2dreview_Pr2845_Ty_ctorIdx(v_y_91_);
v___x_94_ = lean_nat_dec_eq(v___x_92_, v___x_93_);
lean_dec(v___x_93_);
lean_dec(v___x_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqTy___boxed(lean_object* v_x_95_, lean_object* v_y_96_){
_start:
{
uint8_t v_x_13__boxed_97_; uint8_t v_y_14__boxed_98_; uint8_t v_res_99_; lean_object* v_r_100_; 
v_x_13__boxed_97_ = lean_unbox(v_x_95_);
v_y_14__boxed_98_ = lean_unbox(v_y_96_);
v_res_99_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqTy(v_x_13__boxed_97_, v_y_14__boxed_98_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = lean_unsigned_to_nat(2u);
v___x_114_ = lean_nat_to_int(v___x_113_);
return v___x_114_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = lean_unsigned_to_nat(1u);
v___x_116_ = lean_nat_to_int(v___x_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr(uint8_t v_x_117_, lean_object* v_prec_118_){
_start:
{
lean_object* v___y_120_; lean_object* v___y_127_; lean_object* v___y_134_; lean_object* v___y_141_; 
switch(v_x_117_)
{
case 0:
{
lean_object* v___x_147_; uint8_t v___x_148_; 
v___x_147_ = lean_unsigned_to_nat(1024u);
v___x_148_ = lean_nat_dec_le(v___x_147_, v_prec_118_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; 
v___x_149_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_120_ = v___x_149_;
goto v___jp_119_;
}
else
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_120_ = v___x_150_;
goto v___jp_119_;
}
}
case 1:
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = lean_unsigned_to_nat(1024u);
v___x_152_ = lean_nat_dec_le(v___x_151_, v_prec_118_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; 
v___x_153_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_127_ = v___x_153_;
goto v___jp_126_;
}
else
{
lean_object* v___x_154_; 
v___x_154_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_127_ = v___x_154_;
goto v___jp_126_;
}
}
case 2:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = lean_unsigned_to_nat(1024u);
v___x_156_ = lean_nat_dec_le(v___x_155_, v_prec_118_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; 
v___x_157_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_134_ = v___x_157_;
goto v___jp_133_;
}
else
{
lean_object* v___x_158_; 
v___x_158_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_134_ = v___x_158_;
goto v___jp_133_;
}
}
default: 
{
lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_159_ = lean_unsigned_to_nat(1024u);
v___x_160_ = lean_nat_dec_le(v___x_159_, v_prec_118_);
if (v___x_160_ == 0)
{
lean_object* v___x_161_; 
v___x_161_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_141_ = v___x_161_;
goto v___jp_140_;
}
else
{
lean_object* v___x_162_; 
v___x_162_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_141_ = v___x_162_;
goto v___jp_140_;
}
}
}
v___jp_119_:
{
lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_121_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__1));
lean_inc(v___y_120_);
v___x_122_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_122_, 0, v___y_120_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
v___x_123_ = 0;
v___x_124_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_124_, 0, v___x_122_);
lean_ctor_set_uint8(v___x_124_, sizeof(void*)*1, v___x_123_);
v___x_125_ = l_Repr_addAppParen(v___x_124_, v_prec_118_);
return v___x_125_;
}
v___jp_126_:
{
lean_object* v___x_128_; lean_object* v___x_129_; uint8_t v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_128_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__3));
lean_inc(v___y_127_);
v___x_129_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_129_, 0, v___y_127_);
lean_ctor_set(v___x_129_, 1, v___x_128_);
v___x_130_ = 0;
v___x_131_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_131_, 0, v___x_129_);
lean_ctor_set_uint8(v___x_131_, sizeof(void*)*1, v___x_130_);
v___x_132_ = l_Repr_addAppParen(v___x_131_, v_prec_118_);
return v___x_132_;
}
v___jp_133_:
{
lean_object* v___x_135_; lean_object* v___x_136_; uint8_t v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_135_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__5));
lean_inc(v___y_134_);
v___x_136_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_136_, 0, v___y_134_);
lean_ctor_set(v___x_136_, 1, v___x_135_);
v___x_137_ = 0;
v___x_138_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_138_, 0, v___x_136_);
lean_ctor_set_uint8(v___x_138_, sizeof(void*)*1, v___x_137_);
v___x_139_ = l_Repr_addAppParen(v___x_138_, v_prec_118_);
return v___x_139_;
}
v___jp_140_:
{
lean_object* v___x_142_; lean_object* v___x_143_; uint8_t v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_142_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__7));
lean_inc(v___y_141_);
v___x_143_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_143_, 0, v___y_141_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = 0;
v___x_145_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_145_, 0, v___x_143_);
lean_ctor_set_uint8(v___x_145_, sizeof(void*)*1, v___x_144_);
v___x_146_ = l_Repr_addAppParen(v___x_145_, v_prec_118_);
return v___x_146_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprTy_repr___boxed(lean_object* v_x_163_, lean_object* v_prec_164_){
_start:
{
uint8_t v_x_233__boxed_165_; lean_object* v_res_166_; 
v_x_233__boxed_165_ = lean_unbox(v_x_163_);
v_res_166_ = lp_pr2845_x2dreview_Pr2845_instReprTy_repr(v_x_233__boxed_165_, v_prec_164_);
lean_dec(v_prec_164_);
return v_res_166_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq(lean_object* v_x_169_, lean_object* v_x_170_){
_start:
{
lean_object* v_name_171_; lean_object* v_id_172_; lean_object* v_scope_173_; uint8_t v_ty_174_; lean_object* v_name_175_; lean_object* v_id_176_; lean_object* v_scope_177_; uint8_t v_ty_178_; uint8_t v___x_179_; 
v_name_171_ = lean_ctor_get(v_x_169_, 0);
v_id_172_ = lean_ctor_get(v_x_169_, 1);
v_scope_173_ = lean_ctor_get(v_x_169_, 2);
v_ty_174_ = lean_ctor_get_uint8(v_x_169_, sizeof(void*)*3);
v_name_175_ = lean_ctor_get(v_x_170_, 0);
v_id_176_ = lean_ctor_get(v_x_170_, 1);
v_scope_177_ = lean_ctor_get(v_x_170_, 2);
v_ty_178_ = lean_ctor_get_uint8(v_x_170_, sizeof(void*)*3);
v___x_179_ = lean_string_dec_eq(v_name_171_, v_name_175_);
if (v___x_179_ == 0)
{
return v___x_179_;
}
else
{
uint8_t v___x_180_; 
v___x_180_ = lean_nat_dec_eq(v_id_172_, v_id_176_);
if (v___x_180_ == 0)
{
return v___x_180_;
}
else
{
uint8_t v___x_181_; 
v___x_181_ = lean_nat_dec_eq(v_scope_173_, v_scope_177_);
if (v___x_181_ == 0)
{
return v___x_181_;
}
else
{
uint8_t v___x_182_; 
v___x_182_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqTy(v_ty_174_, v_ty_178_);
return v___x_182_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq___boxed(lean_object* v_x_183_, lean_object* v_x_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq(v_x_183_, v_x_184_);
lean_dec_ref(v_x_184_);
lean_dec_ref(v_x_183_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqVar(lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
uint8_t v___x_189_; 
v___x_189_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqVar_decEq(v_x_187_, v_x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqVar___boxed(lean_object* v_x_190_, lean_object* v_x_191_){
_start:
{
uint8_t v_res_192_; lean_object* v_r_193_; 
v_res_192_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqVar(v_x_190_, v_x_191_);
lean_dec_ref(v_x_191_);
lean_dec_ref(v_x_190_);
v_r_193_ = lean_box(v_res_192_);
return v_r_193_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = lean_unsigned_to_nat(8u);
v___x_208_ = lean_nat_to_int(v___x_207_);
return v___x_208_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_215_ = lean_unsigned_to_nat(6u);
v___x_216_ = lean_nat_to_int(v___x_215_);
return v___x_216_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = lean_unsigned_to_nat(9u);
v___x_221_ = lean_nat_to_int(v___x_220_);
return v___x_221_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_226_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__0));
v___x_227_ = lean_string_length(v___x_226_);
return v___x_227_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20(void){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19, &lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19_once, _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__19);
v___x_229_ = lean_nat_to_int(v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg(lean_object* v_x_234_){
_start:
{
lean_object* v_name_235_; lean_object* v_id_236_; lean_object* v_scope_237_; uint8_t v_ty_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; uint8_t v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v_name_235_ = lean_ctor_get(v_x_234_, 0);
lean_inc_ref(v_name_235_);
v_id_236_ = lean_ctor_get(v_x_234_, 1);
lean_inc(v_id_236_);
v_scope_237_ = lean_ctor_get(v_x_234_, 2);
lean_inc(v_scope_237_);
v_ty_238_ = lean_ctor_get_uint8(v_x_234_, sizeof(void*)*3);
lean_dec_ref(v_x_234_);
v___x_239_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__5));
v___x_240_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__6));
v___x_241_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7, &lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7_once, _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__7);
v___x_242_ = l_String_quote(v_name_235_);
v___x_243_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
v___x_244_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_241_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = 0;
v___x_246_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_246_, 0, v___x_244_);
lean_ctor_set_uint8(v___x_246_, sizeof(void*)*1, v___x_245_);
v___x_247_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_240_);
lean_ctor_set(v___x_247_, 1, v___x_246_);
v___x_248_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__9));
v___x_249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_247_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
v___x_250_ = lean_box(1);
v___x_251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_249_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__11));
v___x_253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_251_);
lean_ctor_set(v___x_253_, 1, v___x_252_);
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v___x_239_);
v___x_255_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12, &lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12_once, _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__12);
v___x_256_ = l_Nat_reprFast(v_id_236_);
v___x_257_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
v___x_258_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_255_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
v___x_259_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*1, v___x_245_);
v___x_260_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_254_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
lean_ctor_set(v___x_261_, 1, v___x_248_);
v___x_262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_250_);
v___x_263_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__14));
v___x_264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_262_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
lean_ctor_set(v___x_265_, 1, v___x_239_);
v___x_266_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15, &lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15_once, _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__15);
v___x_267_ = l_Nat_reprFast(v_scope_237_);
v___x_268_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_268_, 0, v___x_267_);
v___x_269_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_266_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
v___x_270_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_270_, 0, v___x_269_);
lean_ctor_set_uint8(v___x_270_, sizeof(void*)*1, v___x_245_);
v___x_271_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_265_);
lean_ctor_set(v___x_271_, 1, v___x_270_);
v___x_272_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v___x_248_);
v___x_273_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
lean_ctor_set(v___x_273_, 1, v___x_250_);
v___x_274_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__17));
v___x_275_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_273_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
lean_ctor_set(v___x_276_, 1, v___x_239_);
v___x_277_ = lean_unsigned_to_nat(0u);
v___x_278_ = lp_pr2845_x2dreview_Pr2845_instReprTy_repr(v_ty_238_, v___x_277_);
v___x_279_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_255_);
lean_ctor_set(v___x_279_, 1, v___x_278_);
v___x_280_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set_uint8(v___x_280_, sizeof(void*)*1, v___x_245_);
v___x_281_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_276_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v___x_282_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20, &lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20_once, _init_lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__20);
v___x_283_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__21));
v___x_284_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set(v___x_284_, 1, v___x_281_);
v___x_285_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg___closed__22));
v___x_286_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_284_);
lean_ctor_set(v___x_286_, 1, v___x_285_);
v___x_287_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_282_);
lean_ctor_set(v___x_287_, 1, v___x_286_);
v___x_288_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_288_, 0, v___x_287_);
lean_ctor_set_uint8(v___x_288_, sizeof(void*)*1, v___x_245_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr(lean_object* v_x_289_, lean_object* v_prec_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg(v_x_289_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprVar_repr___boxed(lean_object* v_x_292_, lean_object* v_prec_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_pr2845_x2dreview_Pr2845_instReprVar_repr(v_x_292_, v_prec_293_);
lean_dec(v_prec_293_);
return v_res_294_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0(lean_object* v_k_297_, lean_object* v_x_298_){
_start:
{
if (lean_obj_tag(v_x_298_) == 0)
{
uint8_t v___x_299_; 
v___x_299_ = 0;
return v___x_299_;
}
else
{
lean_object* v_head_300_; lean_object* v_tail_301_; lean_object* v_fst_302_; uint8_t v___x_303_; 
v_head_300_ = lean_ctor_get(v_x_298_, 0);
v_tail_301_ = lean_ctor_get(v_x_298_, 1);
v_fst_302_ = lean_ctor_get(v_head_300_, 0);
v___x_303_ = lean_string_dec_eq(v_fst_302_, v_k_297_);
if (v___x_303_ == 0)
{
v_x_298_ = v_tail_301_;
goto _start;
}
else
{
return v___x_303_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0___boxed(lean_object* v_k_305_, lean_object* v_x_306_){
_start:
{
uint8_t v_res_307_; lean_object* v_r_308_; 
v_res_307_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0(v_k_305_, v_x_306_);
lean_dec(v_x_306_);
lean_dec_ref(v_k_305_);
v_r_308_ = lean_box(v_res_307_);
return v_r_308_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_Env_has(lean_object* v_e_309_, lean_object* v_k_310_){
_start:
{
uint8_t v___x_311_; 
v___x_311_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0(v_k_310_, v_e_309_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_has___boxed(lean_object* v_e_312_, lean_object* v_k_313_){
_start:
{
uint8_t v_res_314_; lean_object* v_r_315_; 
v_res_314_ = lp_pr2845_x2dreview_Pr2845_Env_has(v_e_312_, v_k_313_);
lean_dec_ref(v_k_313_);
lean_dec(v_e_312_);
v_r_315_ = lean_box(v_res_314_);
return v_r_315_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_insert_spec__0(lean_object* v_k_316_, lean_object* v_v_317_, lean_object* v_a_318_, lean_object* v_a_319_){
_start:
{
if (lean_obj_tag(v_a_318_) == 0)
{
lean_object* v___x_320_; 
lean_dec_ref(v_v_317_);
lean_dec_ref(v_k_316_);
v___x_320_ = l_List_reverse___redArg(v_a_319_);
return v___x_320_;
}
else
{
lean_object* v_head_321_; lean_object* v_tail_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_343_; 
v_head_321_ = lean_ctor_get(v_a_318_, 0);
v_tail_322_ = lean_ctor_get(v_a_318_, 1);
v_isSharedCheck_343_ = !lean_is_exclusive(v_a_318_);
if (v_isSharedCheck_343_ == 0)
{
v___x_324_ = v_a_318_;
v_isShared_325_ = v_isSharedCheck_343_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_tail_322_);
lean_inc(v_head_321_);
lean_dec(v_a_318_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_343_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___y_327_; lean_object* v_fst_332_; uint8_t v___x_333_; 
v_fst_332_ = lean_ctor_get(v_head_321_, 0);
v___x_333_ = lean_string_dec_eq(v_fst_332_, v_k_316_);
if (v___x_333_ == 0)
{
v___y_327_ = v_head_321_;
goto v___jp_326_;
}
else
{
lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_340_; 
v_isSharedCheck_340_ = !lean_is_exclusive(v_head_321_);
if (v_isSharedCheck_340_ == 0)
{
lean_object* v_unused_341_; lean_object* v_unused_342_; 
v_unused_341_ = lean_ctor_get(v_head_321_, 1);
lean_dec(v_unused_341_);
v_unused_342_ = lean_ctor_get(v_head_321_, 0);
lean_dec(v_unused_342_);
v___x_335_ = v_head_321_;
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
else
{
lean_dec(v_head_321_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_338_; 
lean_inc_ref(v_v_317_);
lean_inc_ref(v_k_316_);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 1, v_v_317_);
lean_ctor_set(v___x_335_, 0, v_k_316_);
v___x_338_ = v___x_335_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v_k_316_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v_v_317_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
v___y_327_ = v___x_338_;
goto v___jp_326_;
}
}
}
v___jp_326_:
{
lean_object* v___x_329_; 
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 1, v_a_319_);
lean_ctor_set(v___x_324_, 0, v___y_327_);
v___x_329_ = v___x_324_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___y_327_);
lean_ctor_set(v_reuseFailAlloc_331_, 1, v_a_319_);
v___x_329_ = v_reuseFailAlloc_331_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
v_a_318_ = v_tail_322_;
v_a_319_ = v___x_329_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_insert(lean_object* v_e_344_, lean_object* v_k_345_, lean_object* v_v_346_){
_start:
{
uint8_t v___x_347_; 
v___x_347_ = lp_pr2845_x2dreview_List_any___at___00Pr2845_Env_has_spec__0(v_k_345_, v_e_344_);
if (v___x_347_ == 0)
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_348_, 0, v_k_345_);
lean_ctor_set(v___x_348_, 1, v_v_346_);
v___x_349_ = lean_box(0);
v___x_350_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_348_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = l_List_appendTR___redArg(v_e_344_, v___x_350_);
return v___x_351_;
}
else
{
lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_352_ = lean_box(0);
v___x_353_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_insert_spec__0(v_k_345_, v_v_346_, v_e_344_, v___x_352_);
return v___x_353_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_ids_spec__0(lean_object* v_a_354_, lean_object* v_a_355_){
_start:
{
if (lean_obj_tag(v_a_354_) == 0)
{
lean_object* v___x_356_; 
v___x_356_ = l_List_reverse___redArg(v_a_355_);
return v___x_356_;
}
else
{
lean_object* v_head_357_; lean_object* v_snd_358_; lean_object* v_tail_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_368_; 
v_head_357_ = lean_ctor_get(v_a_354_, 0);
v_snd_358_ = lean_ctor_get(v_head_357_, 1);
lean_inc(v_snd_358_);
v_tail_359_ = lean_ctor_get(v_a_354_, 1);
v_isSharedCheck_368_ = !lean_is_exclusive(v_a_354_);
if (v_isSharedCheck_368_ == 0)
{
lean_object* v_unused_369_; 
v_unused_369_ = lean_ctor_get(v_a_354_, 0);
lean_dec(v_unused_369_);
v___x_361_ = v_a_354_;
v_isShared_362_ = v_isSharedCheck_368_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_tail_359_);
lean_dec(v_a_354_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_368_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v_id_363_; lean_object* v___x_365_; 
v_id_363_ = lean_ctor_get(v_snd_358_, 1);
lean_inc(v_id_363_);
lean_dec(v_snd_358_);
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 1, v_a_355_);
lean_ctor_set(v___x_361_, 0, v_id_363_);
v___x_365_ = v___x_361_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_id_363_);
lean_ctor_set(v_reuseFailAlloc_367_, 1, v_a_355_);
v___x_365_ = v_reuseFailAlloc_367_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
v_a_354_ = v_tail_359_;
v_a_355_ = v___x_365_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Env_ids(lean_object* v_e_370_){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_371_ = lean_box(0);
v___x_372_ = lp_pr2845_x2dreview_List_mapTR_loop___at___00Pr2845_Env_ids_spec__0(v_e_370_, v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_freshVar(lean_object* v_e_373_, lean_object* v_name_374_, uint8_t v_ty_375_, lean_object* v_scope_376_){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_377_ = l_List_lengthTR___redArg(v_e_373_);
v___x_378_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_378_, 0, v_name_374_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
lean_ctor_set(v___x_378_, 2, v_scope_376_);
lean_ctor_set_uint8(v___x_378_, sizeof(void*)*3, v_ty_375_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_freshVar___boxed(lean_object* v_e_379_, lean_object* v_name_380_, lean_object* v_ty_381_, lean_object* v_scope_382_){
_start:
{
uint8_t v_ty_boxed_383_; lean_object* v_res_384_; 
v_ty_boxed_383_ = lean_unbox(v_ty_381_);
v_res_384_ = lp_pr2845_x2dreview_Pr2845_freshVar(v_e_379_, v_name_380_, v_ty_boxed_383_, v_scope_382_);
lean_dec(v_e_379_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectName(lean_object* v_e_385_, lean_object* v_scope_386_, lean_object* v_n_387_, uint8_t v_t_388_){
_start:
{
lean_object* v_v_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
lean_inc_ref(v_n_387_);
v_v_389_ = lp_pr2845_x2dreview_Pr2845_freshVar(v_e_385_, v_n_387_, v_t_388_, v_scope_386_);
lean_inc_ref(v_v_389_);
v___x_390_ = lp_pr2845_x2dreview_Pr2845_Env_insert(v_e_385_, v_n_387_, v_v_389_);
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
lean_ctor_set(v___x_391_, 1, v_v_389_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projectName___boxed(lean_object* v_e_392_, lean_object* v_scope_393_, lean_object* v_n_394_, lean_object* v_t_395_){
_start:
{
uint8_t v_t_boxed_396_; lean_object* v_res_397_; 
v_t_boxed_396_ = lean_unbox(v_t_395_);
v_res_397_ = lp_pr2845_x2dreview_Pr2845_projectName(v_e_392_, v_scope_393_, v_n_394_, v_t_boxed_396_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_importProj(lean_object* v_e_398_, lean_object* v_scope_399_, lean_object* v_x_400_){
_start:
{
if (lean_obj_tag(v_x_400_) == 0)
{
lean_object* v___x_401_; lean_object* v___x_402_; 
lean_dec(v_scope_399_);
v___x_401_ = lean_box(0);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v_e_398_);
lean_ctor_set(v___x_402_, 1, v___x_401_);
return v___x_402_;
}
else
{
lean_object* v_head_403_; lean_object* v_snd_404_; lean_object* v_tail_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_434_; 
v_head_403_ = lean_ctor_get(v_x_400_, 0);
lean_inc(v_head_403_);
v_snd_404_ = lean_ctor_get(v_head_403_, 1);
lean_inc(v_snd_404_);
v_tail_405_ = lean_ctor_get(v_x_400_, 1);
v_isSharedCheck_434_ = !lean_is_exclusive(v_x_400_);
if (v_isSharedCheck_434_ == 0)
{
lean_object* v_unused_435_; 
v_unused_435_ = lean_ctor_get(v_x_400_, 0);
lean_dec(v_unused_435_);
v___x_407_ = v_x_400_;
v_isShared_408_ = v_isSharedCheck_434_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_tail_405_);
lean_dec(v_x_400_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_434_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v_fst_409_; uint8_t v_ty_410_; lean_object* v___x_411_; lean_object* v_fst_412_; lean_object* v_snd_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_433_; 
v_fst_409_ = lean_ctor_get(v_head_403_, 0);
lean_inc(v_fst_409_);
lean_dec(v_head_403_);
v_ty_410_ = lean_ctor_get_uint8(v_snd_404_, sizeof(void*)*3);
lean_inc(v_scope_399_);
v___x_411_ = lp_pr2845_x2dreview_Pr2845_projectName(v_e_398_, v_scope_399_, v_fst_409_, v_ty_410_);
v_fst_412_ = lean_ctor_get(v___x_411_, 0);
v_snd_413_ = lean_ctor_get(v___x_411_, 1);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_433_ == 0)
{
v___x_415_ = v___x_411_;
v_isShared_416_ = v_isSharedCheck_433_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_snd_413_);
lean_inc(v_fst_412_);
lean_dec(v___x_411_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_433_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_417_; lean_object* v_fst_418_; lean_object* v_snd_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_432_; 
v___x_417_ = lp_pr2845_x2dreview_Pr2845_importProj(v_fst_412_, v_scope_399_, v_tail_405_);
v_fst_418_ = lean_ctor_get(v___x_417_, 0);
v_snd_419_ = lean_ctor_get(v___x_417_, 1);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_432_ == 0)
{
v___x_421_ = v___x_417_;
v_isShared_422_ = v_isSharedCheck_432_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_snd_419_);
lean_inc(v_fst_418_);
lean_dec(v___x_417_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_432_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 1, v_snd_404_);
lean_ctor_set(v___x_421_, 0, v_snd_413_);
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_snd_413_);
lean_ctor_set(v_reuseFailAlloc_431_, 1, v_snd_404_);
v___x_424_ = v_reuseFailAlloc_431_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
lean_object* v___x_426_; 
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 1, v_snd_419_);
lean_ctor_set(v___x_407_, 0, v___x_424_);
v___x_426_ = v___x_407_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_424_);
lean_ctor_set(v_reuseFailAlloc_430_, 1, v_snd_419_);
v___x_426_ = v_reuseFailAlloc_430_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
lean_object* v___x_428_; 
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 1, v___x_426_);
lean_ctor_set(v___x_415_, 0, v_fst_418_);
v___x_428_ = v___x_415_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v_fst_418_);
lean_ctor_set(v_reuseFailAlloc_429_, 1, v___x_426_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorIdx(lean_object* v_x_436_){
_start:
{
if (lean_obj_tag(v_x_436_) == 0)
{
lean_object* v___x_437_; 
v___x_437_ = lean_unsigned_to_nat(0u);
return v___x_437_;
}
else
{
lean_object* v___x_438_; 
v___x_438_ = lean_unsigned_to_nat(1u);
return v___x_438_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorIdx___boxed(lean_object* v_x_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorIdx(v_x_439_);
lean_dec_ref(v_x_439_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(lean_object* v_t_441_, lean_object* v_k_442_){
_start:
{
lean_object* v_exprs_443_; lean_object* v___x_444_; 
v_exprs_443_ = lean_ctor_get(v_t_441_, 0);
lean_inc(v_exprs_443_);
lean_dec_ref(v_t_441_);
v___x_444_ = lean_apply_1(v_k_442_, v_exprs_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim(lean_object* v_motive_445_, lean_object* v_ctorIdx_446_, lean_object* v_t_447_, lean_object* v_h_448_, lean_object* v_k_449_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(v_t_447_, v_k_449_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___boxed(lean_object* v_motive_451_, lean_object* v_ctorIdx_452_, lean_object* v_t_453_, lean_object* v_h_454_, lean_object* v_k_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim(v_motive_451_, v_ctorIdx_452_, v_t_453_, v_h_454_, v_k_455_);
lean_dec(v_ctorIdx_452_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_with___00elim___redArg(lean_object* v_t_457_, lean_object* v_with___458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(v_t_457_, v_with___458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_with___00elim(lean_object* v_motive_460_, lean_object* v_t_461_, lean_object* v_h_462_, lean_object* v_with___463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(v_t_461_, v_with___463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_other_elim___redArg(lean_object* v_t_465_, lean_object* v_other_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(v_t_465_, v_other_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_Clause_other_elim(lean_object* v_motive_468_, lean_object* v_t_469_, lean_object* v_h_470_, lean_object* v_other_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_pr2845_x2dreview_Pr2845_Clause_ctorElim___redArg(v_t_469_, v_other_471_);
return v___x_472_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0(lean_object* v_a_473_, lean_object* v_b_474_){
_start:
{
lean_object* v___x_475_; uint8_t v___x_476_; 
v___x_475_ = lean_alloc_closure((void*)(lp_pr2845_x2dreview_Pr2845_instDecidableEqVar___boxed), 2, 0);
lean_inc_ref(v___x_475_);
v___x_476_ = l_instDecidableEqProd___redArg(v___x_475_, v___x_475_, v_a_473_, v_b_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0___boxed(lean_object* v_a_477_, lean_object* v_b_478_){
_start:
{
uint8_t v_res_479_; lean_object* v_r_480_; 
v_res_479_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___lam__0(v_a_477_, v_b_478_);
v_r_480_ = lean_box(v_res_479_);
return v_r_480_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq(lean_object* v_x_482_, lean_object* v_x_483_){
_start:
{
if (lean_obj_tag(v_x_482_) == 0)
{
if (lean_obj_tag(v_x_483_) == 0)
{
lean_object* v_exprs_484_; lean_object* v_exprs_485_; lean_object* v___f_486_; uint8_t v___x_487_; 
v_exprs_484_ = lean_ctor_get(v_x_482_, 0);
lean_inc(v_exprs_484_);
lean_dec_ref_known(v_x_482_, 1);
v_exprs_485_ = lean_ctor_get(v_x_483_, 0);
lean_inc(v_exprs_485_);
lean_dec_ref_known(v_x_483_, 1);
v___f_486_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___closed__0));
v___x_487_ = l_instDecidableEqList___redArg(v___f_486_, v_exprs_484_, v_exprs_485_);
return v___x_487_;
}
else
{
uint8_t v___x_488_; 
lean_dec_ref_known(v_x_483_, 1);
lean_dec_ref_known(v_x_482_, 1);
v___x_488_ = 0;
return v___x_488_;
}
}
else
{
if (lean_obj_tag(v_x_483_) == 0)
{
uint8_t v___x_489_; 
lean_dec_ref_known(v_x_483_, 1);
lean_dec_ref_known(v_x_482_, 1);
v___x_489_ = 0;
return v___x_489_;
}
else
{
lean_object* v_tag_490_; lean_object* v_tag_491_; uint8_t v___x_492_; 
v_tag_490_ = lean_ctor_get(v_x_482_, 0);
lean_inc(v_tag_490_);
lean_dec_ref_known(v_x_482_, 1);
v_tag_491_ = lean_ctor_get(v_x_483_, 0);
lean_inc(v_tag_491_);
lean_dec_ref_known(v_x_483_, 1);
v___x_492_ = lean_nat_dec_eq(v_tag_490_, v_tag_491_);
lean_dec(v_tag_491_);
lean_dec(v_tag_490_);
return v___x_492_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq___boxed(lean_object* v_x_493_, lean_object* v_x_494_){
_start:
{
uint8_t v_res_495_; lean_object* v_r_496_; 
v_res_495_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq(v_x_493_, v_x_494_);
v_r_496_ = lean_box(v_res_495_);
return v_r_496_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_instDecidableEqClause(lean_object* v_x_497_, lean_object* v_x_498_){
_start:
{
uint8_t v___x_499_; 
v___x_499_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqClause_decEq(v_x_497_, v_x_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instDecidableEqClause___boxed(lean_object* v_x_500_, lean_object* v_x_501_){
_start:
{
uint8_t v_res_502_; lean_object* v_r_503_; 
v_res_502_ = lp_pr2845_x2dreview_Pr2845_instDecidableEqClause(v_x_500_, v_x_501_);
v_r_503_ = lean_box(v_res_502_);
return v_r_503_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_509_ = ((lean_object*)(lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__0));
v___x_510_ = lean_string_length(v___x_509_);
return v___x_510_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = lean_obj_once(&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3, &lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3_once, _init_lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__3);
v___x_512_ = lean_nat_to_int(v___x_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(lean_object* v_x_517_){
_start:
{
lean_object* v_fst_518_; lean_object* v_snd_519_; lean_object* v___x_521_; uint8_t v_isShared_522_; uint8_t v_isSharedCheck_541_; 
v_fst_518_ = lean_ctor_get(v_x_517_, 0);
v_snd_519_ = lean_ctor_get(v_x_517_, 1);
v_isSharedCheck_541_ = !lean_is_exclusive(v_x_517_);
if (v_isSharedCheck_541_ == 0)
{
v___x_521_ = v_x_517_;
v_isShared_522_ = v_isSharedCheck_541_;
goto v_resetjp_520_;
}
else
{
lean_inc(v_snd_519_);
lean_inc(v_fst_518_);
lean_dec(v_x_517_);
v___x_521_ = lean_box(0);
v_isShared_522_ = v_isSharedCheck_541_;
goto v_resetjp_520_;
}
v_resetjp_520_:
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_526_; 
v___x_523_ = lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg(v_fst_518_);
v___x_524_ = lean_box(0);
if (v_isShared_522_ == 0)
{
lean_ctor_set_tag(v___x_521_, 1);
lean_ctor_set(v___x_521_, 1, v___x_524_);
lean_ctor_set(v___x_521_, 0, v___x_523_);
v___x_526_ = v___x_521_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v___x_523_);
lean_ctor_set(v_reuseFailAlloc_540_, 1, v___x_524_);
v___x_526_ = v_reuseFailAlloc_540_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; uint8_t v___x_538_; lean_object* v___x_539_; 
v___x_527_ = lp_pr2845_x2dreview_Pr2845_instReprVar_repr___redArg(v_snd_519_);
v___x_528_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_528_, 0, v___x_527_);
lean_ctor_set(v___x_528_, 1, v___x_526_);
v___x_529_ = l_List_reverse___redArg(v___x_528_);
v___x_530_ = ((lean_object*)(lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__1));
v___x_531_ = l_Std_Format_joinSep___at___00Lean_Syntax_formatStxAux_spec__2(v___x_529_, v___x_530_);
v___x_532_ = lean_obj_once(&lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4, &lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4_once, _init_lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__4);
v___x_533_ = ((lean_object*)(lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__5));
v___x_534_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
lean_ctor_set(v___x_534_, 1, v___x_531_);
v___x_535_ = ((lean_object*)(lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__6));
v___x_536_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_534_);
lean_ctor_set(v___x_536_, 1, v___x_535_);
v___x_537_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_532_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = 0;
v___x_539_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_539_, 0, v___x_537_);
lean_ctor_set_uint8(v___x_539_, sizeof(void*)*1, v___x_538_);
return v___x_539_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2_spec__3(lean_object* v_x_542_, lean_object* v_x_543_, lean_object* v_x_544_){
_start:
{
if (lean_obj_tag(v_x_544_) == 0)
{
lean_dec(v_x_542_);
return v_x_543_;
}
else
{
lean_object* v_head_545_; lean_object* v_tail_546_; lean_object* v___x_548_; uint8_t v_isShared_549_; uint8_t v_isSharedCheck_556_; 
v_head_545_ = lean_ctor_get(v_x_544_, 0);
v_tail_546_ = lean_ctor_get(v_x_544_, 1);
v_isSharedCheck_556_ = !lean_is_exclusive(v_x_544_);
if (v_isSharedCheck_556_ == 0)
{
v___x_548_ = v_x_544_;
v_isShared_549_ = v_isSharedCheck_556_;
goto v_resetjp_547_;
}
else
{
lean_inc(v_tail_546_);
lean_inc(v_head_545_);
lean_dec(v_x_544_);
v___x_548_ = lean_box(0);
v_isShared_549_ = v_isSharedCheck_556_;
goto v_resetjp_547_;
}
v_resetjp_547_:
{
lean_object* v___x_551_; 
lean_inc(v_x_542_);
if (v_isShared_549_ == 0)
{
lean_ctor_set_tag(v___x_548_, 5);
lean_ctor_set(v___x_548_, 1, v_x_542_);
lean_ctor_set(v___x_548_, 0, v_x_543_);
v___x_551_ = v___x_548_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_x_543_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v_x_542_);
v___x_551_ = v_reuseFailAlloc_555_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_552_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(v_head_545_);
v___x_553_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_553_, 0, v___x_551_);
lean_ctor_set(v___x_553_, 1, v___x_552_);
v_x_543_ = v___x_553_;
v_x_544_ = v_tail_546_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2(lean_object* v_x_557_, lean_object* v_x_558_, lean_object* v_x_559_){
_start:
{
if (lean_obj_tag(v_x_559_) == 0)
{
lean_dec(v_x_557_);
return v_x_558_;
}
else
{
lean_object* v_head_560_; lean_object* v_tail_561_; lean_object* v___x_563_; uint8_t v_isShared_564_; uint8_t v_isSharedCheck_571_; 
v_head_560_ = lean_ctor_get(v_x_559_, 0);
v_tail_561_ = lean_ctor_get(v_x_559_, 1);
v_isSharedCheck_571_ = !lean_is_exclusive(v_x_559_);
if (v_isSharedCheck_571_ == 0)
{
v___x_563_ = v_x_559_;
v_isShared_564_ = v_isSharedCheck_571_;
goto v_resetjp_562_;
}
else
{
lean_inc(v_tail_561_);
lean_inc(v_head_560_);
lean_dec(v_x_559_);
v___x_563_ = lean_box(0);
v_isShared_564_ = v_isSharedCheck_571_;
goto v_resetjp_562_;
}
v_resetjp_562_:
{
lean_object* v___x_566_; 
lean_inc(v_x_557_);
if (v_isShared_564_ == 0)
{
lean_ctor_set_tag(v___x_563_, 5);
lean_ctor_set(v___x_563_, 1, v_x_557_);
lean_ctor_set(v___x_563_, 0, v_x_558_);
v___x_566_ = v___x_563_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_x_558_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v_x_557_);
v___x_566_ = v_reuseFailAlloc_570_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_567_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(v_head_560_);
v___x_568_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_568_, 0, v___x_566_);
lean_ctor_set(v___x_568_, 1, v___x_567_);
v___x_569_ = lp_pr2845_x2dreview_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2_spec__3(v_x_557_, v___x_568_, v_tail_561_);
return v___x_569_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1(lean_object* v_x_572_, lean_object* v_x_573_){
_start:
{
if (lean_obj_tag(v_x_572_) == 0)
{
lean_object* v___x_574_; 
lean_dec(v_x_573_);
v___x_574_ = lean_box(0);
return v___x_574_;
}
else
{
lean_object* v_tail_575_; 
v_tail_575_ = lean_ctor_get(v_x_572_, 1);
if (lean_obj_tag(v_tail_575_) == 0)
{
lean_object* v_head_576_; lean_object* v___x_577_; 
lean_dec(v_x_573_);
v_head_576_ = lean_ctor_get(v_x_572_, 0);
lean_inc(v_head_576_);
lean_dec_ref_known(v_x_572_, 2);
v___x_577_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(v_head_576_);
return v___x_577_;
}
else
{
lean_object* v_head_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
lean_inc(v_tail_575_);
v_head_578_ = lean_ctor_get(v_x_572_, 0);
lean_inc(v_head_578_);
lean_dec_ref_known(v_x_572_, 2);
v___x_579_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(v_head_578_);
v___x_580_ = lp_pr2845_x2dreview_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1_spec__2(v_x_573_, v___x_579_, v_tail_575_);
return v___x_580_;
}
}
}
}
static lean_object* _init_lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_586_; lean_object* v___x_587_; 
v___x_586_ = ((lean_object*)(lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__2));
v___x_587_ = lean_string_length(v___x_586_);
return v___x_587_;
}
}
static lean_object* _init_lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_588_ = lean_obj_once(&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4, &lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4_once, _init_lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__4);
v___x_589_ = lean_nat_to_int(v___x_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg(lean_object* v_a_594_){
_start:
{
if (lean_obj_tag(v_a_594_) == 0)
{
lean_object* v___x_595_; 
v___x_595_ = ((lean_object*)(lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__1));
return v___x_595_;
}
else
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; uint8_t v___x_604_; lean_object* v___x_605_; 
v___x_596_ = ((lean_object*)(lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg___closed__1));
v___x_597_ = lp_pr2845_x2dreview_Std_Format_joinSep___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__1(v_a_594_, v___x_596_);
v___x_598_ = lean_obj_once(&lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5, &lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5_once, _init_lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__5);
v___x_599_ = ((lean_object*)(lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__6));
v___x_600_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_599_);
lean_ctor_set(v___x_600_, 1, v___x_597_);
v___x_601_ = ((lean_object*)(lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg___closed__7));
v___x_602_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_602_, 0, v___x_600_);
lean_ctor_set(v___x_602_, 1, v___x_601_);
v___x_603_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_598_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = 0;
v___x_605_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_605_, 0, v___x_603_);
lean_ctor_set_uint8(v___x_605_, sizeof(void*)*1, v___x_604_);
return v___x_605_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr(lean_object* v_x_618_, lean_object* v_prec_619_){
_start:
{
if (lean_obj_tag(v_x_618_) == 0)
{
lean_object* v_exprs_620_; lean_object* v___y_622_; lean_object* v___x_630_; uint8_t v___x_631_; 
v_exprs_620_ = lean_ctor_get(v_x_618_, 0);
lean_inc(v_exprs_620_);
lean_dec_ref_known(v_x_618_, 1);
v___x_630_ = lean_unsigned_to_nat(1024u);
v___x_631_ = lean_nat_dec_le(v___x_630_, v_prec_619_);
if (v___x_631_ == 0)
{
lean_object* v___x_632_; 
v___x_632_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_622_ = v___x_632_;
goto v___jp_621_;
}
else
{
lean_object* v___x_633_; 
v___x_633_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_622_ = v___x_633_;
goto v___jp_621_;
}
v___jp_621_:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; uint8_t v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_623_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__2));
v___x_624_ = lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg(v_exprs_620_);
v___x_625_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_625_, 0, v___x_623_);
lean_ctor_set(v___x_625_, 1, v___x_624_);
lean_inc(v___y_622_);
v___x_626_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_626_, 0, v___y_622_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
v___x_627_ = 0;
v___x_628_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_628_, 0, v___x_626_);
lean_ctor_set_uint8(v___x_628_, sizeof(void*)*1, v___x_627_);
v___x_629_ = l_Repr_addAppParen(v___x_628_, v_prec_619_);
return v___x_629_;
}
}
else
{
lean_object* v_tag_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_654_; 
v_tag_634_ = lean_ctor_get(v_x_618_, 0);
v_isSharedCheck_654_ = !lean_is_exclusive(v_x_618_);
if (v_isSharedCheck_654_ == 0)
{
v___x_636_ = v_x_618_;
v_isShared_637_ = v_isSharedCheck_654_;
goto v_resetjp_635_;
}
else
{
lean_inc(v_tag_634_);
lean_dec(v_x_618_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_654_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v___y_639_; lean_object* v___x_650_; uint8_t v___x_651_; 
v___x_650_ = lean_unsigned_to_nat(1024u);
v___x_651_ = lean_nat_dec_le(v___x_650_, v_prec_619_);
if (v___x_651_ == 0)
{
lean_object* v___x_652_; 
v___x_652_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__8);
v___y_639_ = v___x_652_;
goto v___jp_638_;
}
else
{
lean_object* v___x_653_; 
v___x_653_ = lean_obj_once(&lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9, &lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9_once, _init_lp_pr2845_x2dreview_Pr2845_instReprTy_repr___closed__9);
v___y_639_ = v___x_653_;
goto v___jp_638_;
}
v___jp_638_:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_643_; 
v___x_640_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_instReprClause_repr___closed__5));
v___x_641_ = l_Nat_reprFast(v_tag_634_);
if (v_isShared_637_ == 0)
{
lean_ctor_set_tag(v___x_636_, 3);
lean_ctor_set(v___x_636_, 0, v___x_641_);
v___x_643_ = v___x_636_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v___x_641_);
v___x_643_ = v_reuseFailAlloc_649_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
lean_object* v___x_644_; lean_object* v___x_645_; uint8_t v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_644_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_644_, 0, v___x_640_);
lean_ctor_set(v___x_644_, 1, v___x_643_);
lean_inc(v___y_639_);
v___x_645_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_645_, 0, v___y_639_);
lean_ctor_set(v___x_645_, 1, v___x_644_);
v___x_646_ = 0;
v___x_647_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_647_, 0, v___x_645_);
lean_ctor_set_uint8(v___x_647_, sizeof(void*)*1, v___x_646_);
v___x_648_ = l_Repr_addAppParen(v___x_647_, v_prec_619_);
return v___x_648_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_instReprClause_repr___boxed(lean_object* v_x_655_, lean_object* v_prec_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_pr2845_x2dreview_Pr2845_instReprClause_repr(v_x_655_, v_prec_656_);
lean_dec(v_prec_656_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0(lean_object* v_a_658_, lean_object* v_n_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___redArg(v_a_658_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0___boxed(lean_object* v_a_661_, lean_object* v_n_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_pr2845_x2dreview_List_repr___at___00Pr2845_instReprClause_repr_spec__0(v_a_661_, v_n_662_);
lean_dec(v_n_662_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0(lean_object* v_x_664_, lean_object* v_x_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___redArg(v_x_664_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0___boxed(lean_object* v_x_667_, lean_object* v_x_668_){
_start:
{
lean_object* v_res_669_; 
v_res_669_ = lp_pr2845_x2dreview_Prod_repr___at___00List_repr___at___00Pr2845_instReprClause_repr_spec__0_spec__0(v_x_667_, v_x_668_);
lean_dec(v_x_668_);
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyNew(lean_object* v_outerLen_672_, lean_object* v_imported_673_, lean_object* v_rest_674_){
_start:
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v_snd_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_685_; 
v___x_675_ = lean_box(0);
v___x_676_ = lp_pr2845_x2dreview_Pr2845_importProj(v___x_675_, v_outerLen_672_, v_imported_673_);
v_snd_677_ = lean_ctor_get(v___x_676_, 1);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_676_);
if (v_isSharedCheck_685_ == 0)
{
lean_object* v_unused_686_; 
v_unused_686_ = lean_ctor_get(v___x_676_, 0);
lean_dec(v_unused_686_);
v___x_679_ = v___x_676_;
v_isShared_680_ = v_isSharedCheck_685_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_snd_677_);
lean_dec(v___x_676_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_685_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_681_; lean_object* v___x_683_; 
v___x_681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_681_, 0, v_snd_677_);
if (v_isShared_680_ == 0)
{
lean_ctor_set_tag(v___x_679_, 1);
lean_ctor_set(v___x_679_, 1, v_rest_674_);
lean_ctor_set(v___x_679_, 0, v___x_681_);
v___x_683_ = v___x_679_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v___x_681_);
lean_ctor_set(v_reuseFailAlloc_684_, 1, v_rest_674_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyOld(lean_object* v_outerLen_687_, lean_object* v_imported_688_, lean_object* v_rest_689_){
_start:
{
uint8_t v___x_690_; 
v___x_690_ = l_List_isEmpty___redArg(v_imported_688_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v_snd_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_701_; 
v___x_691_ = lean_box(0);
v___x_692_ = lp_pr2845_x2dreview_Pr2845_importProj(v___x_691_, v_outerLen_687_, v_imported_688_);
v_snd_693_ = lean_ctor_get(v___x_692_, 1);
v_isSharedCheck_701_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_701_ == 0)
{
lean_object* v_unused_702_; 
v_unused_702_ = lean_ctor_get(v___x_692_, 0);
lean_dec(v_unused_702_);
v___x_695_ = v___x_692_;
v_isShared_696_ = v_isSharedCheck_701_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_snd_693_);
lean_dec(v___x_692_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_701_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v___x_697_; lean_object* v___x_699_; 
v___x_697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_697_, 0, v_snd_693_);
if (v_isShared_696_ == 0)
{
lean_ctor_set_tag(v___x_695_, 1);
lean_ctor_set(v___x_695_, 1, v_rest_689_);
lean_ctor_set(v___x_695_, 0, v___x_697_);
v___x_699_ = v___x_695_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v___x_697_);
lean_ctor_set(v_reuseFailAlloc_700_, 1, v_rest_689_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
else
{
lean_dec(v_imported_688_);
lean_dec(v_outerLen_687_);
return v_rest_689_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchScope(lean_object* v_outerLen_703_, uint8_t v_with3027_704_){
_start:
{
if (v_with3027_704_ == 0)
{
lean_object* v___x_705_; 
v___x_705_ = lean_unsigned_to_nat(0u);
return v___x_705_;
}
else
{
lean_inc(v_outerLen_703_);
return v_outerLen_703_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchScope___boxed(lean_object* v_outerLen_706_, lean_object* v_with3027_707_){
_start:
{
uint8_t v_with3027_boxed_708_; lean_object* v_res_709_; 
v_with3027_boxed_708_ = lean_unbox(v_with3027_707_);
v_res_709_ = lp_pr2845_x2dreview_Pr2845_branchScope(v_outerLen_706_, v_with3027_boxed_708_);
lean_dec(v_outerLen_706_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchNew(lean_object* v_outerLen_710_, uint8_t v_with3027_711_, lean_object* v_imported_712_, lean_object* v_rest_713_){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v_snd_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_725_; 
v___x_714_ = lean_box(0);
v___x_715_ = lp_pr2845_x2dreview_Pr2845_branchScope(v_outerLen_710_, v_with3027_711_);
v___x_716_ = lp_pr2845_x2dreview_Pr2845_importProj(v___x_714_, v___x_715_, v_imported_712_);
v_snd_717_ = lean_ctor_get(v___x_716_, 1);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_716_);
if (v_isSharedCheck_725_ == 0)
{
lean_object* v_unused_726_; 
v_unused_726_ = lean_ctor_get(v___x_716_, 0);
lean_dec(v_unused_726_);
v___x_719_ = v___x_716_;
v_isShared_720_ = v_isSharedCheck_725_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_snd_717_);
lean_dec(v___x_716_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_725_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_721_; lean_object* v___x_723_; 
v___x_721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_721_, 0, v_snd_717_);
if (v_isShared_720_ == 0)
{
lean_ctor_set_tag(v___x_719_, 1);
lean_ctor_set(v___x_719_, 1, v_rest_713_);
lean_ctor_set(v___x_719_, 0, v___x_721_);
v___x_723_ = v___x_719_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v___x_721_);
lean_ctor_set(v_reuseFailAlloc_724_, 1, v_rest_713_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchNew___boxed(lean_object* v_outerLen_727_, lean_object* v_with3027_728_, lean_object* v_imported_729_, lean_object* v_rest_730_){
_start:
{
uint8_t v_with3027_boxed_731_; lean_object* v_res_732_; 
v_with3027_boxed_731_ = lean_unbox(v_with3027_728_);
v_res_732_ = lp_pr2845_x2dreview_Pr2845_branchNew(v_outerLen_727_, v_with3027_boxed_731_, v_imported_729_, v_rest_730_);
lean_dec(v_outerLen_727_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__5_splitter___redArg(lean_object* v_x_733_, lean_object* v_h__1_734_, lean_object* v_h__2_735_){
_start:
{
if (lean_obj_tag(v_x_733_) == 0)
{
lean_object* v___x_736_; lean_object* v___x_737_; 
lean_dec(v_h__2_735_);
v___x_736_ = lean_box(0);
v___x_737_ = lean_apply_1(v_h__1_734_, v___x_736_);
return v___x_737_;
}
else
{
lean_object* v_head_738_; lean_object* v_tail_739_; lean_object* v_fst_740_; lean_object* v_snd_741_; lean_object* v___x_742_; 
lean_dec(v_h__1_734_);
v_head_738_ = lean_ctor_get(v_x_733_, 0);
lean_inc(v_head_738_);
v_tail_739_ = lean_ctor_get(v_x_733_, 1);
lean_inc(v_tail_739_);
lean_dec_ref_known(v_x_733_, 2);
v_fst_740_ = lean_ctor_get(v_head_738_, 0);
lean_inc(v_fst_740_);
v_snd_741_ = lean_ctor_get(v_head_738_, 1);
lean_inc(v_snd_741_);
lean_dec(v_head_738_);
v___x_742_ = lean_apply_3(v_h__2_735_, v_fst_740_, v_snd_741_, v_tail_739_);
return v___x_742_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__5_splitter(lean_object* v_motive_743_, lean_object* v_x_744_, lean_object* v_h__1_745_, lean_object* v_h__2_746_){
_start:
{
if (lean_obj_tag(v_x_744_) == 0)
{
lean_object* v___x_747_; lean_object* v___x_748_; 
lean_dec(v_h__2_746_);
v___x_747_ = lean_box(0);
v___x_748_ = lean_apply_1(v_h__1_745_, v___x_747_);
return v___x_748_;
}
else
{
lean_object* v_head_749_; lean_object* v_tail_750_; lean_object* v_fst_751_; lean_object* v_snd_752_; lean_object* v___x_753_; 
lean_dec(v_h__1_745_);
v_head_749_ = lean_ctor_get(v_x_744_, 0);
lean_inc(v_head_749_);
v_tail_750_ = lean_ctor_get(v_x_744_, 1);
lean_inc(v_tail_750_);
lean_dec_ref_known(v_x_744_, 2);
v_fst_751_ = lean_ctor_get(v_head_749_, 0);
lean_inc(v_fst_751_);
v_snd_752_ = lean_ctor_get(v_head_749_, 1);
lean_inc(v_snd_752_);
lean_dec(v_head_749_);
v___x_753_ = lean_apply_3(v_h__2_746_, v_fst_751_, v_snd_752_, v_tail_750_);
return v___x_753_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__3_splitter___redArg(lean_object* v_x_754_, lean_object* v_h__1_755_){
_start:
{
lean_object* v_fst_756_; lean_object* v_snd_757_; lean_object* v___x_758_; 
v_fst_756_ = lean_ctor_get(v_x_754_, 0);
lean_inc(v_fst_756_);
v_snd_757_ = lean_ctor_get(v_x_754_, 1);
lean_inc(v_snd_757_);
lean_dec_ref(v_x_754_);
v___x_758_ = lean_apply_2(v_h__1_755_, v_fst_756_, v_snd_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__3_splitter(lean_object* v_motive_759_, lean_object* v_x_760_, lean_object* v_h__1_761_){
_start:
{
lean_object* v_fst_762_; lean_object* v_snd_763_; lean_object* v___x_764_; 
v_fst_762_ = lean_ctor_get(v_x_760_, 0);
lean_inc(v_fst_762_);
v_snd_763_ = lean_ctor_get(v_x_760_, 1);
lean_inc(v_snd_763_);
lean_dec_ref(v_x_760_);
v___x_764_ = lean_apply_2(v_h__1_761_, v_fst_762_, v_snd_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__1_splitter___redArg(lean_object* v_x_765_, lean_object* v_h__1_766_){
_start:
{
lean_object* v_fst_767_; lean_object* v_snd_768_; lean_object* v___x_769_; 
v_fst_767_ = lean_ctor_get(v_x_765_, 0);
lean_inc(v_fst_767_);
v_snd_768_ = lean_ctor_get(v_x_765_, 1);
lean_inc(v_snd_768_);
lean_dec_ref(v_x_765_);
v___x_769_ = lean_apply_2(v_h__1_766_, v_fst_767_, v_snd_768_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Binder_0__Pr2845_importProj_match__1_splitter(lean_object* v_motive_770_, lean_object* v_x_771_, lean_object* v_h__1_772_){
_start:
{
lean_object* v_fst_773_; lean_object* v_snd_774_; lean_object* v___x_775_; 
v_fst_773_ = lean_ctor_get(v_x_771_, 0);
lean_inc(v_fst_773_);
v_snd_774_ = lean_ctor_get(v_x_771_, 1);
lean_inc(v_snd_774_);
lean_dec_ref(v_x_771_);
v___x_775_ = lean_apply_2(v_h__1_772_, v_fst_773_, v_snd_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg(lean_object* v_x_776_, lean_object* v_x_777_){
_start:
{
if (lean_obj_tag(v_x_777_) == 0)
{
lean_object* v___x_778_; 
v___x_778_ = lean_box(0);
return v___x_778_;
}
else
{
lean_object* v_head_779_; lean_object* v_tail_780_; lean_object* v_fst_781_; lean_object* v_snd_782_; uint8_t v___y_784_; lean_object* v_fst_787_; lean_object* v_snd_788_; lean_object* v_fst_789_; lean_object* v_snd_790_; uint8_t v___x_791_; 
v_head_779_ = lean_ctor_get(v_x_777_, 0);
v_tail_780_ = lean_ctor_get(v_x_777_, 1);
v_fst_781_ = lean_ctor_get(v_head_779_, 0);
v_snd_782_ = lean_ctor_get(v_head_779_, 1);
v_fst_787_ = lean_ctor_get(v_x_776_, 0);
v_snd_788_ = lean_ctor_get(v_x_776_, 1);
v_fst_789_ = lean_ctor_get(v_fst_781_, 0);
v_snd_790_ = lean_ctor_get(v_fst_781_, 1);
v___x_791_ = lean_nat_dec_eq(v_fst_787_, v_fst_789_);
if (v___x_791_ == 0)
{
v___y_784_ = v___x_791_;
goto v___jp_783_;
}
else
{
uint8_t v___x_792_; 
v___x_792_ = lean_nat_dec_eq(v_snd_788_, v_snd_790_);
v___y_784_ = v___x_792_;
goto v___jp_783_;
}
v___jp_783_:
{
if (v___y_784_ == 0)
{
v_x_777_ = v_tail_780_;
goto _start;
}
else
{
lean_object* v___x_786_; 
lean_inc(v_snd_782_);
v___x_786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_786_, 0, v_snd_782_);
return v___x_786_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg___boxed(lean_object* v_x_793_, lean_object* v_x_794_){
_start:
{
lean_object* v_res_795_; 
v_res_795_ = lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg(v_x_793_, v_x_794_);
lean_dec(v_x_794_);
lean_dec_ref(v_x_793_);
return v_res_795_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_relabel(lean_object* v_L_796_, lean_object* v_v_797_, lean_object* v_own_798_){
_start:
{
lean_object* v_id_799_; lean_object* v_scope_800_; lean_object* v___x_801_; lean_object* v___x_802_; 
v_id_799_ = lean_ctor_get(v_v_797_, 1);
v_scope_800_ = lean_ctor_get(v_v_797_, 2);
lean_inc(v_id_799_);
lean_inc(v_scope_800_);
v___x_801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_801_, 0, v_scope_800_);
lean_ctor_set(v___x_801_, 1, v_id_799_);
v___x_802_ = lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg(v___x_801_, v_L_796_);
lean_dec_ref_known(v___x_801_, 2);
if (lean_obj_tag(v___x_802_) == 0)
{
lean_inc(v_own_798_);
return v_own_798_;
}
else
{
lean_object* v_val_803_; 
v_val_803_ = lean_ctor_get(v___x_802_, 0);
lean_inc(v_val_803_);
lean_dec_ref_known(v___x_802_, 1);
return v_val_803_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_relabel___boxed(lean_object* v_L_804_, lean_object* v_v_805_, lean_object* v_own_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_pr2845_x2dreview_Pr2845_relabel(v_L_804_, v_v_805_, v_own_806_);
lean_dec(v_own_806_);
lean_dec_ref(v_v_805_);
lean_dec(v_L_804_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0(lean_object* v_00_u03b2_808_, lean_object* v_x_809_, lean_object* v_x_810_){
_start:
{
lean_object* v___x_811_; 
v___x_811_ = lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___redArg(v_x_809_, v_x_810_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0___boxed(lean_object* v_00_u03b2_812_, lean_object* v_x_813_, lean_object* v_x_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_pr2845_x2dreview_List_lookup___at___00Pr2845_relabel_spec__0(v_00_u03b2_812_, v_x_813_, v_x_814_);
lean_dec(v_x_814_);
lean_dec_ref(v_x_813_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchQ(uint8_t v_with3027_830_){
_start:
{
lean_object* v___x_831_; lean_object* v_s_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v_fst_835_; lean_object* v___x_836_; uint8_t v___x_837_; lean_object* v___x_838_; 
v___x_831_ = lean_unsigned_to_nat(1u);
v_s_832_ = lp_pr2845_x2dreview_Pr2845_branchScope(v___x_831_, v_with3027_830_);
v___x_833_ = lean_box(0);
lean_inc(v_s_832_);
v___x_834_ = lp_pr2845_x2dreview_Pr2845_importProj(v___x_833_, v_s_832_, v___x_833_);
v_fst_835_ = lean_ctor_get(v___x_834_, 0);
lean_inc(v_fst_835_);
lean_dec_ref(v___x_834_);
v___x_836_ = ((lean_object*)(lp_pr2845_x2dreview_Pr2845_branchQ___closed__0));
v___x_837_ = 1;
v___x_838_ = lp_pr2845_x2dreview_Pr2845_freshVar(v_fst_835_, v___x_836_, v___x_837_, v_s_832_);
lean_dec(v_fst_835_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_branchQ___boxed(lean_object* v_with3027_839_){
_start:
{
uint8_t v_with3027_boxed_840_; lean_object* v_res_841_; 
v_with3027_boxed_840_ = lean_unbox(v_with3027_839_);
v_res_841_ = lp_pr2845_x2dreview_Pr2845_branchQ(v_with3027_boxed_840_);
return v_res_841_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_pr2845_x2dreview_Pr2845Review_Binder(uint8_t builtin) {
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
