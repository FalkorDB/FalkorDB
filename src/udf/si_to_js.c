/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "quickjs.h"
#include "classes.h"
#include "../value.h"
#include "../datatypes/datatypes.h"

// convert an SIValue to a JavaScript object
JSValue UDF_SIValueToJS
(
	JSContext *js_ctx,
	SIValue val
) {
	JSValue js_val ;

	SIType t = SI_TYPE (val) ;
	switch (t) {

		case T_MAP:
		{
			js_val = JS_NewObject (js_ctx) ;
			uint n = Map_KeyCount (val) ;
			for (int i = 0; i < n; i++) {
				SIValue key;
				SIValue value;

				Map_GetIdx (val, i, &key, &value) ;
				JS_SetPropertyStr (js_ctx, js_val, key.stringval,
						UDF_SIValueToJS (js_ctx, value)) ;
			}
			break ;
		}

		case T_NODE:
		{
			js_val = js_create_node (js_ctx, val.ptrval) ;
			break ;
		}

		case T_EDGE :
		{
			js_val = js_create_edge (js_ctx, val.ptrval) ;
			break ;
		}

		case T_ARRAY:
		{
			js_val = JS_NewArray (js_ctx) ;
			int n = SIArray_Length (val) ;
			for (int i = 0; i < n; i++) {
				JS_SetPropertyUint32 (js_ctx, js_val, i,
						UDF_SIValueToJS (js_ctx, SIArray_Get (val, i))) ;
			}
			break ;
		}

		case T_PATH:
		{
			js_val = js_create_path (js_ctx, val.ptrval) ;
			break ;
		}

		case T_DATETIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_LOCALDATETIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_DATE:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_TIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_LOCALTIME:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_DURATION:
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_STRING:
		{
			js_val = JS_NewString (js_ctx, val.stringval) ;
			break ;
		}

		case T_BOOL:
		{
			js_val = JS_NewBool (js_ctx, val.longval) ;
			break ;
		}

		case T_INT64:
		{
			js_val = JS_NewInt64 (js_ctx, val.longval) ;
			break ;
		}

		case T_DOUBLE:
		{
			js_val = JS_NewFloat64 (js_ctx, val.doubleval) ;
			break ;
		}

		case T_NULL:
		{
			js_val = JS_NULL ;
			break ;
		}

		case T_PTR: 
		{
			assert (false && "Not implemented") ;
			break ;
		}

		case T_POINT:
		{
			js_val = JS_NewObject (js_ctx) ;
			JS_SetPropertyStr (js_ctx, js_val, "latitude", JS_NewFloat64 (js_ctx, Point_lat (val))) ;
			JS_SetPropertyStr (js_ctx, js_val, "longitude", JS_NewFloat64 (js_ctx, Point_lon (val))) ;
			break ;
		}

		case T_VECTOR_F32:
		{
			js_val = JS_NewArray (js_ctx) ;
			int n = SIVector_Dim (val) ;
			float *elements = SIVector_Elements (val) ;
			for (int i = 0; i < n; i++) {
				JS_SetPropertyUint32 (js_ctx, js_val, i,
						JS_NewFloat64(js_ctx, elements[i])) ;
			}
			break ;
		}

		case T_INTERN_STRING: 
		{
			js_val = JS_NewString (js_ctx, val.stringval) ;
			break ;
		}
		
		default:
		{
			assert (false && "Unknown SIValue type") ;
			break ;
		}
	}

	return js_val ;
}

