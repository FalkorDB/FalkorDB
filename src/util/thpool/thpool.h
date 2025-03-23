/**********************************
 * @author      Johan Hanssen Seferidis
 * License:     MIT
 *
 **********************************/

#ifndef _THPOOL_
#define _THPOOL_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <sys/types.h>

/* =================================== API ======================================= */


typedef struct thpool_* threadpool;


/**
 * @brief  Initialize threadpool
 *
 * Initializes a threadpool. This function will not return untill all
 * threads have initialized successfully.
 *
 * @example
 *
 *    ..
 *    threadpool thpool;                     //First we declare a threadpool
 *    thpool = thpool_init(4);               //then we initialize it to 4 threads
 *    ..
 *
 * @param  num_threads   number of threads to be created in the threadpool
 * @param  name          name associated with pool
 * @return threadpool    created threadpool on success,
 *                       NULL on error
 */
threadpool thpool_init(int num_threads, const char *name);


/**
 * @brief Add work to the job queue
 *
 * Takes an action and its argument and adds it to the threadpool's job queue.
 * If you want to add to work a function with more than one arguments then
 * a way to implement this is by passing a pointer to a structure.
 *
 * NOTICE: You have to cast both the function and argument to not get warnings.
 *
 * @example
 *
 *    void print_num(int num){
 *       printf("%d\n", num);
 *    }
 *
 *    int main() {
 *       ..
 *       int a = 10;
 *       thpool_add_work(thpool, (void*)print_num, (void*)a);
 *       ..
 *    }
 *
 * @param  threadpool    threadpool to which the work will be added
 * @param  function_p    pointer to function to add as work
 * @param  arg_p         pointer to an argument
 * @return 0 on successs -1 otherwise
 */
int thpool_add_work(threadpool, void (*function_p)(void*), void* arg_p);


/**
 * @brief Wait for all queued jobs to finish
 *
 * Will wait for all jobs - both queued and currently running to finish.
 * Once the queue is empty and all work has completed, the calling thread
 * (probably the main program) will continue.
 *
 * Smart polling is used in wait. The polling is initially 0 - meaning that
 * there is virtually no polling at all. If after 1 seconds the threads
 * haven't finished, the polling interval starts growing exponentially
 * untill it reaches max_secs seconds. Then it jumps down to a maximum polling
 * interval assuming that heavy processing is being used in the threadpool.
 *
 * @example
 *
 *    ..
 *    threadpool thpool = thpool_init(4);
 *    ..
 *    // Add a bunch of work
 *    ..
 *    thpool_wait(thpool);
 *    puts("All added work has finished");
 *    ..
 *
 * @param threadpool     the threadpool to wait for
 * @return nothing
 */
void thpool_wait(threadpool);


/**
 * @brief Pauses all threads immediately
 *
 * The threads will be paused no matter if they are idle or working.
 * The threads return to their previous states once thpool_resume
 * is called.
 *
 * While the thread is being paused, new work can be added.
 *
 * @example
 *
 *    threadpool thpool = thpool_init(4);
 *    thpool_pause(thpool);
 *    ..
 *    // Add a bunch of work
 *    ..
 *    thpool_resume(thpool); // Let the threads start their magic
 *
 * @param threadpool    the threadpool where the threads should be paused
 * @return nothing
 */
void thpool_pause(threadpool);


/**
 * @brief Unpauses all threads if they are paused
 *
 * @example
 *    ..
 *    thpool_pause(thpool);
 *    sleep(10);              // Delay execution 10 seconds
 *    thpool_resume(thpool);
 *    ..
 *
 * @param threadpool     the threadpool where the threads should be unpaused
 * @return nothing
 */
void thpool_resume(threadpool);


/**
 * @brief Destroy the threadpool
 *
 * This will wait for the currently active threads to finish and then 'kill'
 * the whole threadpool to free up memory.
 *
 * @example
 * int main() {
 *    threadpool thpool1 = thpool_init(2);
 *    threadpool thpool2 = thpool_init(2);
 *    ..
 *    thpool_destroy(thpool1);
 *    ..
 *    return 0;
 * }
 *
 * @param threadpool     the threadpool to destroy
 * @return nothing
 */
void thpool_destroy(threadpool);


/**
 * @brief Show currently working threads
 *
 * Working threads are the threads that are performing work (not idle).
 *
 * @example
 * int main() {
 *    threadpool thpool1 = thpool_init(2);
 *    threadpool thpool2 = thpool_init(2);
 *    ..
 *    printf("Working threads: %d\n", thpool_num_threads_working(thpool1));
 *    ..
 *    return 0;
 * }
 *
 * @param threadpool     the threadpool of interest
 * @return integer       number of threads working
 */
int thpool_num_threads_working(threadpool);


/**
 * @brief Returns number of threads in pool.
 *
 * Alive threads are the threads that are either performing work or are idle.
 *
 * @param threadpool     the threadpool of interest
 * @return integer       number of alive threads
 */
int thpool_num_threads(threadpool);


/**
 * @brief Returns friendly id associated with thread.
 *
 * @param threadpool    the threadpool of interest
 * @param pthread_t     the thread of interest
 * @return integer      friendly thread id
 */
int thpool_get_thread_id(threadpool, pthread_t);

// collect threads ids
void thpool_collect_thread_ids
(
	threadpool thpool_p,  // thread pool
	pthread_t *ids,       // thread ids
	uint *n               // size of ids array
);

/**
 * @brief return true if thread pool internal queue is full with pending work
 *
 * @param threadpool    the threadpool of interest
 * @return bool         is the queue full
 */
bool thpool_queue_full(threadpool);

/**
 * @brief Sets jobqueue capacity.
 *
 * @param threadpool    the threadpool of interest
 * @param uint64_t      capacity of the queue
 */
void thpool_set_jobqueue_cap
(
	threadpool,
	uint64_t
);

/**
 * @brief Gets jobqueue capacity.
 *
 * @param threadpool    the threadpool of interest
 */
uint64_t thpool_get_jobqueue_cap
(
	threadpool
);

/**
 * @brief Gets jobqueue length.
 *
 * @param threadpool    the threadpool of interest
 */
uint64_t thpool_get_jobqueue_len
(
	threadpool
);

// collects tasks matching given handler
void thpool_get_tasks
(
	threadpool thpool_p,      // thread pool
	void **tasks,             // array of tasks
	uint32_t *num_tasks,      // number of tasks collected
	void (*handler)(void *),  // handler function
	void (*match)(void*)      // [optional] executed on every match task
);

#ifdef __cplusplus
}
#endif

#endif
