/* ********************************
 * Author:       Johan Hanssen Seferidis
 * License:	     MIT
 * Description:  Library providing a threading pool where you can add
 *               work. For usage, check the thpool.h file or README.md
 *
 */ /** @file thpool.h */ /*
   *
   ********************************/

//#define _POSIX_C_SOURCE 200809L
#include "RG.h"
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <assert.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "thpool.h"

#include "rmalloc.h"

#ifdef THPOOL_DEBUG
#define THPOOL_DEBUG 1
#else
#define THPOOL_DEBUG 0
#endif

#if !defined(DISABLE_PRINT) || defined(THPOOL_DEBUG)
#define err(str) fprintf(stderr, str)
#else
#define err(str)
#endif

static atomic_uint_fast32_t threads_keepalive;
static atomic_uint_fast32_t threads_on_hold;

/* ========================== STRUCTURES ============================ */

/* Binary semaphore */
typedef struct bsem {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	int v;
} bsem;

/* Job */
typedef struct job {
	struct job *prev;            /* pointer to previous job   */
	void (*function)(void *arg); /* function pointer          */
	void *arg;                   /* function's argument       */
} job;

/* Job queue */
typedef struct jobqueue {
	pthread_mutex_t rwmutex; 		/* used for queue r/w access */
	job *front;              		/* pointer to front of queue */
	job *rear;               		/* pointer to rear  of queue */
	bsem *has_jobs;          		/* flag as binary semaphore  */
	int len;                 		/* number of jobs in queue   */
	uint64_t cap;                   /* capacity of the queue     */
} jobqueue;

/* Thread */
typedef struct thread {
	int id;                   /* friendly id               */
	pthread_t pthread;        /* pointer to actual thread  */
	struct thpool_ *thpool_p; /* access to thpool          */
} thread;

/* Threadpool */
typedef struct thpool_ {
	thread **threads;                         /* pointer to threads              */
	const char *name;                         /* name associated with pool       */
	atomic_uint_fast32_t num_threads_alive;   /* threads currently alive         */
	atomic_uint_fast32_t num_threads_working; /* threads currently working       */
	pthread_mutex_t thcount_lock;             /* used for the condition variable */
	pthread_cond_t threads_all_idle;          /* signal to thpool_wait           */
	jobqueue jobqueue;                        /* job queue                       */
} thpool_;

/* ========================== PROTOTYPES ============================ */

static int thread_init(thpool_* thpool_p, struct thread **thread_p, int id);
static void *thread_do(struct thread *thread_p);
static void thread_hold(int sig_id);
static void thread_destroy(struct thread *thread_p);

static int jobqueue_init(jobqueue *jobqueue_p);
static void jobqueue_clear(jobqueue *jobqueue_p);
static void jobqueue_push(jobqueue *jobqueue_p, struct job *newjob_p);
static struct job *jobqueue_pull(jobqueue *jobqueue_p);
static void jobqueue_destroy(jobqueue *jobqueue_p);

static void bsem_init(struct bsem *bsem_p, int value);
static void bsem_reset(struct bsem *bsem_p);
static void bsem_post(struct bsem *bsem_p);
static void bsem_post_all(struct bsem *bsem_p);
static void bsem_wait(struct bsem *bsem_p);

/* ========================== THREADPOOL ============================ */

/* Initialise thread pool */
struct thpool_ *thpool_init(int num_threads, const char *name) {

	threads_on_hold = 0;
	threads_keepalive = 1;

	if(num_threads < 0) {
		num_threads = 0;
	}

	/* Make new thread pool */
	thpool_* thpool_p;
	thpool_p = (struct thpool_*)rm_calloc(1, sizeof(struct thpool_));
	if(thpool_p == NULL) {
		err("thpool_init(): Could not allocate memory for thread pool\n");
		return NULL;
	}
	if(name == NULL) {
		err("thpool_init(): missing thread pool name\n");
		return NULL;
	}

	thpool_p->name = name;
	thpool_p->num_threads_alive = 0;
	thpool_p->num_threads_working = 0;

	/* Initialise the job queue */
	if(jobqueue_init(&thpool_p->jobqueue) == -1) {
		err("thpool_init(): Could not allocate memory for job queue\n");
		rm_free(thpool_p);
		return NULL;
	}

	/* Make threads in pool */
	thpool_p->threads = (struct thread **)rm_calloc(num_threads, sizeof(struct thread *));
	if(thpool_p->threads == NULL) {
		err("thpool_init(): Could not allocate memory for threads\n");
		jobqueue_destroy(&thpool_p->jobqueue);
		rm_free(thpool_p);
		return NULL;
	}

	pthread_mutex_init(&(thpool_p->thcount_lock), NULL);
	pthread_cond_init(&thpool_p->threads_all_idle, NULL);

	/* Thread init */
	int n;
	for(n = 0; n < num_threads; n++) {
		thread_init(thpool_p, &thpool_p->threads[n], n);
#if THPOOL_DEBUG
		printf("THPOOL_DEBUG: Created thread %d in pool \n", n);
#endif
	}

	/* Wait for threads to initialize */
	while(thpool_p->num_threads_alive != num_threads) {
	}

	return thpool_p;
}

/* Add work to the thread pool */
int thpool_add_work(thpool_* thpool_p, void (*function_p)(void *), void *arg_p) {
	job *newjob;

	newjob = (struct job *)rm_calloc(1, sizeof(struct job));
	if(newjob == NULL) {
		err("thpool_add_work(): Could not allocate memory for new job\n");
		return -1;
	}

	/* add function and argument */
	newjob->function = function_p;
	newjob->arg = arg_p;

	/* add job to queue */
	jobqueue_push(&thpool_p->jobqueue, newjob);

	return 0;
}

/* Wait until all jobs have finished */
void thpool_wait(thpool_* thpool_p) {
	pthread_mutex_lock(&thpool_p->thcount_lock);
	while(thpool_p->jobqueue.len || thpool_p->num_threads_working) {
		pthread_cond_wait(&thpool_p->threads_all_idle, &thpool_p->thcount_lock);
	}
	pthread_mutex_unlock(&thpool_p->thcount_lock);
}

/* Destroy the threadpool */
void thpool_destroy(thpool_* thpool_p) {
	/* No need to destory if it's NULL */
	if (thpool_p == NULL) {
		return;
	}

	const uint32_t threads_total = thpool_p->num_threads_alive;

	/* Make the threads leave their main loops. */
	threads_keepalive = 0;

	// Wake up the threads so that they exit and will become ready to be
	// destroyed.
	while(thpool_p->num_threads_alive) {
		bsem_post_all(thpool_p->jobqueue.has_jobs);
	}

	// Destroy the threads.
	for (size_t i = 0; i < threads_total; ++i) {
		ASSERT(pthread_join(thpool_p->threads[i]->pthread, NULL) == 0);
		thread_destroy(thpool_p->threads[i]);
	}
	rm_free(thpool_p->threads);

	/* Job queue cleanup */
	jobqueue_destroy(&thpool_p->jobqueue);

	rm_free(thpool_p);
}

int thpool_num_threads_working(thpool_* thpool_p) {
	return thpool_p->num_threads_working;
}

int thpool_num_threads(thpool_* thpool_p) {
	return thpool_p->num_threads_alive;
}

int thpool_get_thread_id(thpool_* thpool_p, pthread_t pthread) {
	for(int i = 0; i < thpool_p->num_threads_alive; i++) {
		thread *thread = thpool_p->threads[i];
		if(thread->pthread == pthread) return thread->id;
	}

	// Could not locate thread.
	return -1;
}

// return true if thread pool internal queue is full with pending work
bool thpool_queue_full(thpool_* thpool_p) {
	ASSERT(thpool_p != NULL);

	// test if there's enough room in thread pool queue
	return (thpool_p->jobqueue.len >= thpool_p->jobqueue.cap);
}

void thpool_set_jobqueue_cap
(
	thpool_* thpool_p,
	uint64_t val
) {
	ASSERT(thpool_p);
	thpool_p->jobqueue.cap = val;
}

uint64_t thpool_get_jobqueue_cap
(
	thpool_* thpool_p
) {
	ASSERT(thpool_p);
	return thpool_p->jobqueue.cap;
}

uint64_t thpool_get_jobqueue_len
(
	thpool_* thpool_p
) {
	ASSERT(thpool_p);
	return thpool_p->jobqueue.len;
}

// collects tasks matching given handler
void thpool_get_tasks
(
	threadpool thpool_p,      // thread pool
	void **tasks,             // array of tasks
	uint32_t *num_tasks,      // number of tasks collected
	void (*handler)(void *),  // handler function
	void (*match)(void*)      // [optional] executed on every match task
) {
	// validations
	ASSERT(tasks     != NULL);
	ASSERT(handler   != NULL);
	ASSERT(thpool_p  != NULL);
	ASSERT(num_tasks != NULL);

	jobqueue *jobqueue_p = &thpool_p->jobqueue;

	// lock job queue
	pthread_mutex_lock(&jobqueue_p->rwmutex);

	job *job = jobqueue_p->front;

	// iterate through job queue
	uint32_t k = 0;  // number of matched tasks
	for(uint32_t i = 0; i < jobqueue_p->len && i < *num_tasks; i++) {
		// check if job matches given handler
		if(job->function == handler) {
			tasks[k++] = job->arg;

			// execute match function if given
			if(match != NULL) {
				match(job->arg);
			}
		}

		// advance to next job
		job = job->prev;
	}

	// release lock
	pthread_mutex_unlock(&jobqueue_p->rwmutex);

	// set number of tasks collected
	*num_tasks = k;
}

/* ============================ THREAD ============================== */

/* Initialize a thread in the thread pool
 *
 * @param thread        address to the pointer of the thread to be created
 * @param id            id to be given to the thread
 * @return 0 on success, -1 otherwise.
 */
static int thread_init(thpool_* thpool_p, struct thread **thread_p, int id) {

	*thread_p = (struct thread *)rm_calloc(1, sizeof(struct thread));
	if(thread_p == NULL) {
		err("thread_init(): Could not allocate memory for thread\n");
		return -1;
	}

	(*thread_p)->thpool_p = thpool_p;
	(*thread_p)->id = id;

	pthread_create(&(*thread_p)->pthread, NULL, (void *)thread_do, (*thread_p));
	return 0;
}

/* Sets the calling thread on hold */
static void thread_hold(int sig_id) {
	(void)sig_id;
	threads_on_hold = 1;
	while(threads_on_hold) {
		sleep(1);
	}
}

/* What each thread is doing
*
* In principle this is an endless loop. The only time this loop gets interuppted is once
* thpool_destroy() is invoked or the program exits.
*
* @param  thread        thread that will run this function
* @return nothing
*/
static void *thread_do(struct thread *thread_p) {

	/* Set thread name for profiling and debuging */
	char thread_name[128] = {0};
	snprintf(thread_name, sizeof(thread_name), "thread-pool-%s-%d", 
		thread_p->thpool_p->name, thread_p->id);
	
#if defined(__linux__)
	/* Use prctl instead to prevent using _GNU_SOURCE flag and implicit declaration */
	prctl(PR_SET_NAME, thread_name);
#elif defined(__APPLE__) && defined(__MACH__)
	pthread_setname_np(thread_name);
#else
	err("thread_do(): pthread_setname_np is not supported on this system");
#endif

	/* Assure all threads have been created before starting serving */
	thpool_* thpool_p = thread_p->thpool_p;

	/* Mark thread as alive (initialized) */
	++thpool_p->num_threads_alive;

	while(threads_keepalive) {

		bsem_wait(thpool_p->jobqueue.has_jobs);

		if(threads_keepalive) {
			++thpool_p->num_threads_working;

			/* Read job from queue and execute it */
			void (*func_buff)(void *);
			void *arg_buff;
			job *job_p = jobqueue_pull(&thpool_p->jobqueue);
			if(job_p) {
				func_buff = job_p->function;
				arg_buff = job_p->arg;
				func_buff(arg_buff);
				rm_free(job_p);
			}

			--thpool_p->num_threads_working;
			pthread_mutex_lock(&thpool_p->thcount_lock);
			if (!thpool_p->num_threads_working) {
				pthread_cond_signal(&thpool_p->threads_all_idle);
			}
			pthread_mutex_unlock(&thpool_p->thcount_lock);
		}
	}

	--thpool_p->num_threads_alive;

	return NULL;
}

/* Frees a thread  */
static void thread_destroy(thread *thread_p) {
	rm_free(thread_p);
}

/* ============================ JOB QUEUE =========================== */

/* Initialize queue */
static int jobqueue_init(jobqueue *jobqueue_p) {
	jobqueue_p->len         =  0;
	jobqueue_p->front       =  NULL;
	jobqueue_p->rear        =  NULL;

	jobqueue_p->has_jobs = (struct bsem *)rm_calloc(1, sizeof(struct bsem));
	if(jobqueue_p->has_jobs == NULL) {
		return -1;
	}

	pthread_mutex_init(&(jobqueue_p->rwmutex), NULL);
	bsem_init(jobqueue_p->has_jobs, 0);

	jobqueue_p->cap = UINT64_MAX; // unlimited queue size

	return 0;
}

/* Clear the queue */
static void jobqueue_clear(jobqueue *jobqueue_p) {

	while(jobqueue_p->len) {
		rm_free(jobqueue_pull(jobqueue_p));
	}

	jobqueue_p->front = NULL;
	jobqueue_p->rear = NULL;
	bsem_reset(jobqueue_p->has_jobs);
	jobqueue_p->len = 0;
}

/* Add (allocated) job to queue */
static void jobqueue_push(jobqueue *jobqueue_p, struct job *newjob) {
	newjob->prev = NULL;

	pthread_mutex_lock(&jobqueue_p->rwmutex);

	switch(jobqueue_p->len) {
		case 0: /* no jobs in queue */
			jobqueue_p->front = newjob;
			jobqueue_p->rear = newjob;
			break;
		default: /* jobs in queue */
			jobqueue_p->rear->prev = newjob;
			jobqueue_p->rear = newjob;
	}

	jobqueue_p->len++;
	bsem_post(jobqueue_p->has_jobs);

	pthread_mutex_unlock(&jobqueue_p->rwmutex);
}

/* Get first job from queue(removes it from queue)
 *
 * Notice: Caller MUST hold a mutex
 */
static struct job *jobqueue_pull(jobqueue *jobqueue_p) {

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	job *job_p = jobqueue_p->front;

	switch(jobqueue_p->len) {

	case 0: /* if no jobs in queue */
		break;

	case 1: /* if one job in queue */
		jobqueue_p->front = NULL;
		jobqueue_p->rear = NULL;
		jobqueue_p->len = 0;
		break;

	default: /* if >1 jobs in queue */
		jobqueue_p->front = job_p->prev;
		jobqueue_p->len--;
		/* more than one job in queue -> post it */
		bsem_post(jobqueue_p->has_jobs);
	}

	pthread_mutex_unlock(&jobqueue_p->rwmutex);
	return job_p;
}

/* Free all queue resources back to the system */
static void jobqueue_destroy(jobqueue *jobqueue_p) {
	jobqueue_clear(jobqueue_p);
	rm_free(jobqueue_p->has_jobs);
}

/* ======================== SYNCHRONISATION ========================= */

/* Init semaphore to 1 or 0 */
static void bsem_init(bsem *bsem_p, int value) {
	if(value < 0 || value > 1) {
		err("bsem_init(): Binary semaphore can take only values 1 or 0");
		exit(1);
	}
	pthread_mutex_init(&(bsem_p->mutex), NULL);
	pthread_cond_init(&(bsem_p->cond), NULL);
	bsem_p->v = value;
}

/* Reset semaphore to 0 */
static void bsem_reset(bsem *bsem_p) {
	bsem_init(bsem_p, 0);
}

/* Post to at least one thread */
static void bsem_post(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_signal(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}

/* Post to all threads */
static void bsem_post_all(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_broadcast(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}

/* Wait on semaphore until semaphore has value 0 */
static void bsem_wait(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	while(bsem_p->v != 1) {
		pthread_cond_wait(&bsem_p->cond, &bsem_p->mutex);
	}
	bsem_p->v = 0;
	pthread_mutex_unlock(&bsem_p->mutex);
}
