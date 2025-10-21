/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

/* Include all available execution plan operations. */
#include "op_sort.h"
#include "op_skip.h"
#include "op_join.h"
#include "op_apply.h"
#include "op_merge.h"
#include "op_limit.h"
#include "op_create.h"
#include "op_delete.h"
#include "op_filter.h"
#include "op_update.h"
#include "op_unwind.h"
#include "op_results.h"
#include "op_project.h"
#include "op_foreach.h"
#include "op_optional.h"
#include "op_argument.h"
#include "op_load_csv.h"
#include "op_distinct.h"
#include "op_aggregate.h"
#include "op_empty_row.h"
#include "op_semi_apply.h"
#include "op_expand_into.h"
#include "op_merge_create.h"
#include "op_argument_list.h"
#include "op_all_node_scan.h"
#include "op_procedure_call.h"
#include "op_node_by_id_seek.h"
#include "op_value_hash_join.h"
#include "op_subquery_foreach.h"
#include "op_apply_multiplexer.h"
#include "op_cartesian_product.h"
#include "op_edge_by_index_scan.h"
#include "op_node_by_label_scan.h"
#include "op_node_by_index_scan.h"
#include "op_conditional_traverse.h"
#include "op_cond_var_len_traverse.h"
