/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cron/cron.h"
#include "util/rmalloc.h"
#include "reconf_handler.h"
#include "util/thpool/pools.h"

// handler function invoked when config changes
void reconf_handler(Config_Option_Field type) {
	switch (type)
	{
		//----------------------------------------------------------------------
		// max queued queries
		//----------------------------------------------------------------------

		case Config_MAX_QUEUED_QUERIES:
			{
				uint64_t max_queued_queries;
				bool res = Config_Option_get(type, &max_queued_queries);
				ASSERT(res);
				ThreadPools_SetMaxPendingWork(max_queued_queries);
			}
			break;

		//----------------------------------------------------------------------
		// query mem capacity
		//----------------------------------------------------------------------

		case Config_QUERY_MEM_CAPACITY:
			{
				int64_t query_mem_capacity;
				bool res = Config_Option_get(type, &query_mem_capacity);
				ASSERT(res);
				rm_set_mem_capacity(query_mem_capacity);
			}
			break;

		//----------------------------------------------------------------------
		// graph info
		//----------------------------------------------------------------------

		case Config_CMD_INFO:
			{
				bool info_enabled;
				bool res = Config_Option_get(type, &info_enabled);
				ASSERT(res);
				if(info_enabled) {
					CronTask_AddStreamFinishedQueries();
				}
			}
			break;

        //----------------------------------------------------------------------
        // all other options
        //----------------------------------------------------------------------

        default:
			return;
    }
}

