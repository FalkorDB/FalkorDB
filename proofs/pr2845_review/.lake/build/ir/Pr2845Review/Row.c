// Lean compiler output
// Module: Pr2845Review.Row
// Imports: public import Init public meta import Init public import Pr2845Review.Binder
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__Pr2845_projRow_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__Pr2845_projRow_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_scanMode___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_scanMode___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_scanMode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_scanMode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0(lean_object* v_i_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
else
{
lean_object* v_head_4_; lean_object* v_fst_5_; lean_object* v_tail_6_; lean_object* v_id_7_; uint8_t v___x_8_; 
v_head_4_ = lean_ctor_get(v_x_2_, 0);
v_fst_5_ = lean_ctor_get(v_head_4_, 0);
v_tail_6_ = lean_ctor_get(v_x_2_, 1);
v_id_7_ = lean_ctor_get(v_fst_5_, 1);
v___x_8_ = lean_nat_dec_eq(v_id_7_, v_i_1_);
if (v___x_8_ == 0)
{
v_x_2_ = v_tail_6_;
goto _start;
}
else
{
lean_object* v___x_10_; 
lean_inc(v_head_4_);
v___x_10_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_10_, 0, v_head_4_);
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0___boxed(lean_object* v_i_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0(v_i_11_, v_x_12_);
lean_dec(v_x_12_);
lean_dec(v_i_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___redArg(lean_object* v_ps_14_, lean_object* v_r_15_, lean_object* v_i_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_pr2845_x2dreview_List_find_x3f___at___00Pr2845_projRow_spec__0(v_i_16_, v_ps_14_);
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v___x_18_; 
lean_dec_ref(v_r_15_);
v___x_18_ = lean_box(0);
return v___x_18_;
}
else
{
lean_object* v_val_19_; lean_object* v_snd_20_; lean_object* v_id_21_; lean_object* v___x_22_; 
v_val_19_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_val_19_);
lean_dec_ref_known(v___x_17_, 1);
v_snd_20_ = lean_ctor_get(v_val_19_, 1);
lean_inc(v_snd_20_);
lean_dec(v_val_19_);
v_id_21_ = lean_ctor_get(v_snd_20_, 1);
lean_inc(v_id_21_);
lean_dec(v_snd_20_);
v___x_22_ = lean_apply_1(v_r_15_, v_id_21_);
return v___x_22_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___redArg___boxed(lean_object* v_ps_23_, lean_object* v_r_24_, lean_object* v_i_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_pr2845_x2dreview_Pr2845_projRow___redArg(v_ps_23_, v_r_24_, v_i_25_);
lean_dec(v_i_25_);
lean_dec(v_ps_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow(lean_object* v_V_27_, lean_object* v_ps_28_, lean_object* v_r_29_, lean_object* v_i_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_pr2845_x2dreview_Pr2845_projRow___redArg(v_ps_28_, v_r_29_, v_i_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_projRow___boxed(lean_object* v_V_32_, lean_object* v_ps_33_, lean_object* v_r_34_, lean_object* v_i_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_pr2845_x2dreview_Pr2845_projRow(v_V_32_, v_ps_33_, v_r_34_, v_i_35_);
lean_dec(v_i_35_);
lean_dec(v_ps_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg(lean_object* v_ps_37_, uint8_t v_imported_38_, lean_object* v_r_39_, lean_object* v_a_40_){
_start:
{
if (v_imported_38_ == 0)
{
lean_object* v___x_41_; 
v___x_41_ = lean_apply_1(v_r_39_, v_a_40_);
return v___x_41_;
}
else
{
lean_object* v___x_42_; 
v___x_42_ = lp_pr2845_x2dreview_Pr2845_projRow___redArg(v_ps_37_, v_r_39_, v_a_40_);
lean_dec(v_a_40_);
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg___boxed(lean_object* v_ps_43_, lean_object* v_imported_44_, lean_object* v_r_45_, lean_object* v_a_46_){
_start:
{
uint8_t v_imported_boxed_47_; lean_object* v_res_48_; 
v_imported_boxed_47_ = lean_unbox(v_imported_44_);
v_res_48_ = lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg(v_ps_43_, v_imported_boxed_47_, v_r_45_, v_a_46_);
lean_dec(v_ps_43_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld(lean_object* v_V_49_, lean_object* v_ps_50_, uint8_t v_imported_51_, lean_object* v_r_52_, lean_object* v_a_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_pr2845_x2dreview_Pr2845_bodyInputOld___redArg(v_ps_50_, v_imported_51_, v_r_52_, v_a_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputOld___boxed(lean_object* v_V_55_, lean_object* v_ps_56_, lean_object* v_imported_57_, lean_object* v_r_58_, lean_object* v_a_59_){
_start:
{
uint8_t v_imported_boxed_60_; lean_object* v_res_61_; 
v_imported_boxed_60_ = lean_unbox(v_imported_57_);
v_res_61_ = lp_pr2845_x2dreview_Pr2845_bodyInputOld(v_V_55_, v_ps_56_, v_imported_boxed_60_, v_r_58_, v_a_59_);
lean_dec(v_ps_56_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___redArg(lean_object* v_ps_62_, lean_object* v_r_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_pr2845_x2dreview_Pr2845_projRow___redArg(v_ps_62_, v_r_63_, v_a_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___redArg___boxed(lean_object* v_ps_66_, lean_object* v_r_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_pr2845_x2dreview_Pr2845_bodyInputNew___redArg(v_ps_66_, v_r_67_, v_a_68_);
lean_dec(v_a_68_);
lean_dec(v_ps_66_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew(lean_object* v_V_70_, lean_object* v_ps_71_, lean_object* v_r_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_pr2845_x2dreview_Pr2845_projRow___redArg(v_ps_71_, v_r_72_, v_a_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_bodyInputNew___boxed(lean_object* v_V_75_, lean_object* v_ps_76_, lean_object* v_r_77_, lean_object* v_a_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_pr2845_x2dreview_Pr2845_bodyInputNew(v_V_75_, v_ps_76_, v_r_77_, v_a_78_);
lean_dec(v_a_78_);
lean_dec(v_ps_76_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___redArg(uint8_t v_x_80_, lean_object* v_h__1_81_, lean_object* v_h__2_82_){
_start:
{
if (v_x_80_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_h__1_81_);
v___x_83_ = lean_box(0);
v___x_84_ = lean_apply_1(v_h__2_82_, v___x_83_);
return v___x_84_;
}
else
{
lean_object* v___x_85_; lean_object* v___x_86_; 
lean_dec(v_h__2_82_);
v___x_85_ = lean_box(0);
v___x_86_ = lean_apply_1(v_h__1_81_, v___x_85_);
return v___x_86_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___redArg___boxed(lean_object* v_x_87_, lean_object* v_h__1_88_, lean_object* v_h__2_89_){
_start:
{
uint8_t v_x_24__boxed_90_; lean_object* v_res_91_; 
v_x_24__boxed_90_ = lean_unbox(v_x_87_);
v_res_91_ = lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___redArg(v_x_24__boxed_90_, v_h__1_88_, v_h__2_89_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter(lean_object* v_motive_92_, uint8_t v_x_93_, lean_object* v_h__1_94_, lean_object* v_h__2_95_){
_start:
{
if (v_x_93_ == 0)
{
lean_object* v___x_96_; lean_object* v___x_97_; 
lean_dec(v_h__1_94_);
v___x_96_ = lean_box(0);
v___x_97_ = lean_apply_1(v_h__2_95_, v___x_96_);
return v___x_97_;
}
else
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v_h__2_95_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_apply_1(v_h__1_94_, v___x_98_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter___boxed(lean_object* v_motive_100_, lean_object* v_x_101_, lean_object* v_h__1_102_, lean_object* v_h__2_103_){
_start:
{
uint8_t v_x_35__boxed_104_; lean_object* v_res_105_; 
v_x_35__boxed_104_ = lean_unbox(v_x_101_);
v_res_105_ = lp_pr2845_x2dreview___private_Pr2845Review_Row_0__List_filter_match__1_splitter(v_motive_100_, v_x_35__boxed_104_, v_h__1_102_, v_h__2_103_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__Pr2845_projRow_match__1_splitter___redArg(lean_object* v_x_106_, lean_object* v_h__1_107_, lean_object* v_h__2_108_){
_start:
{
if (lean_obj_tag(v_x_106_) == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v_h__1_107_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_apply_1(v_h__2_108_, v___x_109_);
return v___x_110_;
}
else
{
lean_object* v_val_111_; lean_object* v___x_112_; 
lean_dec(v_h__2_108_);
v_val_111_ = lean_ctor_get(v_x_106_, 0);
lean_inc(v_val_111_);
lean_dec_ref_known(v_x_106_, 1);
v___x_112_ = lean_apply_1(v_h__1_107_, v_val_111_);
return v___x_112_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview___private_Pr2845Review_Row_0__Pr2845_projRow_match__1_splitter(lean_object* v_motive_113_, lean_object* v_x_114_, lean_object* v_h__1_115_, lean_object* v_h__2_116_){
_start:
{
if (lean_obj_tag(v_x_114_) == 0)
{
lean_object* v___x_117_; lean_object* v___x_118_; 
lean_dec(v_h__1_115_);
v___x_117_ = lean_box(0);
v___x_118_ = lean_apply_1(v_h__2_116_, v___x_117_);
return v___x_118_;
}
else
{
lean_object* v_val_119_; lean_object* v___x_120_; 
lean_dec(v_h__2_116_);
v_val_119_ = lean_ctor_get(v_x_114_, 0);
lean_inc(v_val_119_);
lean_dec_ref_known(v_x_114_, 1);
v___x_120_ = lean_apply_1(v_h__1_115_, v_val_119_);
return v___x_120_;
}
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_scanMode___redArg(lean_object* v_r_121_, lean_object* v_endpoint_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lean_apply_1(v_r_121_, v_endpoint_122_);
if (lean_obj_tag(v___x_123_) == 0)
{
uint8_t v___x_124_; 
v___x_124_ = 0;
return v___x_124_;
}
else
{
uint8_t v___x_125_; 
lean_dec_ref_known(v___x_123_, 1);
v___x_125_ = 1;
return v___x_125_;
}
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_scanMode___redArg___boxed(lean_object* v_r_126_, lean_object* v_endpoint_127_){
_start:
{
uint8_t v_res_128_; lean_object* v_r_129_; 
v_res_128_ = lp_pr2845_x2dreview_Pr2845_scanMode___redArg(v_r_126_, v_endpoint_127_);
v_r_129_ = lean_box(v_res_128_);
return v_r_129_;
}
}
LEAN_EXPORT uint8_t lp_pr2845_x2dreview_Pr2845_scanMode(lean_object* v_V_130_, lean_object* v_r_131_, lean_object* v_endpoint_132_){
_start:
{
uint8_t v___x_133_; 
v___x_133_ = lp_pr2845_x2dreview_Pr2845_scanMode___redArg(v_r_131_, v_endpoint_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_pr2845_x2dreview_Pr2845_scanMode___boxed(lean_object* v_V_134_, lean_object* v_r_135_, lean_object* v_endpoint_136_){
_start:
{
uint8_t v_res_137_; lean_object* v_r_138_; 
v_res_137_ = lp_pr2845_x2dreview_Pr2845_scanMode(v_V_134_, v_r_135_, v_endpoint_136_);
v_r_138_ = lean_box(v_res_137_);
return v_r_138_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_pr2845_x2dreview_Pr2845Review_Binder(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_pr2845_x2dreview_Pr2845Review_Row(uint8_t builtin) {
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
res = initialize_pr2845_x2dreview_Pr2845Review_Binder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
