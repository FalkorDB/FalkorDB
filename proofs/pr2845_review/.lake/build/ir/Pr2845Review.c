// Lean compiler output
// Module: Pr2845Review
// Imports: public import Init public meta import Init public import Pr2845Review.Binder public import Pr2845Review.Row public import Pr2845Review.Optimizer public import Pr2845Review.Batch
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_pr2845_x2dreview_Pr2845Review_Binder(uint8_t builtin);
lean_object* initialize_pr2845_x2dreview_Pr2845Review_Row(uint8_t builtin);
lean_object* initialize_pr2845_x2dreview_Pr2845Review_Optimizer(uint8_t builtin);
lean_object* initialize_pr2845_x2dreview_Pr2845Review_Batch(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_pr2845_x2dreview_Pr2845Review(uint8_t builtin) {
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
res = initialize_pr2845_x2dreview_Pr2845Review_Row(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_pr2845_x2dreview_Pr2845Review_Optimizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_pr2845_x2dreview_Pr2845Review_Batch(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
