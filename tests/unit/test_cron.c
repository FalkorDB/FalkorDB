/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/cron/cron.h"
#include "src/util/rmalloc.h"
#include "src/util/simple_timer.h"

#include <assert.h>
#include <sys/select.h>
#include <stdatomic.h>

static void setup();
static void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define TEST_CASE_TIMEOUT_SECONDS 3
// Just some random bytes we write/read from the pipes which signal that
// a task has either is already running or has completed.
#define RUNNING_SIGNAL_VALUE 105
#define COMPLETED_SIGNAL_VALUE 123
#define INVALID_SIGNAL_VALUE 100

volatile int X = 1;

static int mssleep(uint ms) {
	struct timespec req;

	req.tv_sec = ms / 1000;
	req.tv_nsec = (ms % 1000) * 1000000;

	return nanosleep(&req, NULL);
}

static struct timeval _get_test_timeout_value() {
	struct timeval timeout = {};

	timeout.tv_sec = TEST_CASE_TIMEOUT_SECONDS;
	timeout.tv_usec = 0;

	return timeout;
}

void add_task(void *pdata) {
	int *Y = (int*)pdata;
	X += *Y;
}

typedef void* TaskData;
typedef void (*TaskFunction)(TaskData);

typedef struct AddTaskData {
	TaskFunction task_function;
	TaskData task_data;
	int pipe_read_fd;
	int pipe_write_fd;
	atomic_bool has_started;
	atomic_bool has_completed;
} AddTaskData;

static AddTaskData _AddTaskData_New
(
	TaskFunction task_function,
	TaskData task_data
) {
	int pipe_fds[2] = {0, 0};

	TEST_ASSERT(pipe(pipe_fds) == 0);

	AddTaskData data = {
		.task_function = task_function,
		.task_data = task_data,
		.pipe_read_fd = pipe_fds[0],
		.pipe_write_fd = pipe_fds[1],
		.has_started = false,
		.has_completed = false
	};

	return data;
}

static inline bool _AddTaskData_HasStarted
(
	const AddTaskData data
) {
	return data.has_started;
}

static inline bool _AddTaskData_HasCompleted
(
	const AddTaskData data
) {
	return data.has_completed;
}

static void _AddTaskData_SignaliseValue
(
	AddTaskData data,
	const unsigned char signal_value
) {
	static const size_t SIGNAL_SIZE = sizeof(signal_value);
	const int written = write(
		data.pipe_write_fd,
		(const void *)&signal_value,
		SIGNAL_SIZE
	);
	assert(written == SIGNAL_SIZE);
}

static void _AddTaskData_SignalRunning
(
	AddTaskData data
) {
	_AddTaskData_SignaliseValue(data, RUNNING_SIGNAL_VALUE);
}

static void _AddTaskData_SignalCompleted
(
	AddTaskData data
) {
	_AddTaskData_SignaliseValue(data, COMPLETED_SIGNAL_VALUE);
}

static void _AddTaskData_Execute
(
	void *userdata
) {
	AddTaskData *data = (AddTaskData*) userdata;

	data->has_started = true;
	_AddTaskData_SignalRunning(*data);
	data->task_function(data->task_data);
	data->has_completed = true;
	_AddTaskData_SignalCompleted(*data);
}

static void _AddTaskData_WaitUntilSignalisedValue
(
	const AddTaskData data,
	const unsigned char signalised_value
) {
	struct timeval timeout = _get_test_timeout_value();
	unsigned char read_signal_data = INVALID_SIGNAL_VALUE;

	// Prepare the read file descriptors by including our signal pipe.
	fd_set read_fds;
	FD_ZERO(&read_fds);
	FD_SET(data.pipe_read_fd, &read_fds);

	while (read_signal_data != signalised_value) {
		const int result = select(FD_SETSIZE, &read_fds, NULL, NULL, &timeout);
		// Check that there wasn't an error of "select".
		TEST_ASSERT(result != -1);
		// Check that we didn't encounter the timeout scenario and
		// there is data available for reading, meaning the AddTask task
		// has successfully completed.
		TEST_ASSERT(result > 0);
		// Check that the read channel is available for reading and has unread
		// data.
		TEST_ASSERT(FD_ISSET(data.pipe_read_fd, &read_fds));
		// Wait until the signal expected is received.
		assert(read(data.pipe_read_fd, (void *)&read_signal_data, 1) == 1);
	}
}

static void _AddTaskData_WaitForRunning
(
	const AddTaskData data
) {
	if (data.has_started) {
		return;
	}

	_AddTaskData_WaitUntilSignalisedValue(data, RUNNING_SIGNAL_VALUE);
}

static void _AddTaskData_WaitForCompletion
(
	const AddTaskData data
) {
	if (data.has_completed) {
		return;
	}

	_AddTaskData_WaitUntilSignalisedValue(data, COMPLETED_SIGNAL_VALUE);
}

static void _AddTaskData_Wait
(
	const AddTaskData data
) {
	_AddTaskData_WaitForRunning(data);
	_AddTaskData_WaitForCompletion(data);
}

static void _AddTaskData_Free(const AddTaskData data) {
	close(data.pipe_read_fd);
	close(data.pipe_write_fd);
}

static void mul_task(void *pdata) {
	int *Y = (int*)pdata;
	X *= *Y;
}

static void long_running_task(void *pdata) {
	// sleep for 'n' milliseconds
	int *ms = (int*)pdata;
	mssleep(*ms);
}

static void setup() {
	// Use the malloc family for allocations
	Alloc_Reset();
	Cron_Start();
}

static void tearDown() {
	Cron_Stop();
}

static void test_cronExec() {
	// Add two tasks to CRON
	// one adds 2 to X
	// second multiply X by 2

	X = 1;
	int Y = 2;
	int Z = 2;

	// Introduce tasks
	// X = 1.
	// X *= Y, X = 2
	// X += Z, X = 4
	// if CRON executes the tasks in the wrong order
	// X += Z, X = 3
	// X *= Y, X = 6
	AddTaskData add_task_data = _AddTaskData_New(add_task, (void *)&Z);
	AddTaskData mul_task_data = _AddTaskData_New(mul_task, (void *)&Y);

	Cron_AddTask(15, _AddTaskData_Execute, NULL, &add_task_data);
	Cron_AddTask(5,  _AddTaskData_Execute, NULL, &mul_task_data);

	_AddTaskData_Wait(add_task_data);
	_AddTaskData_Wait(mul_task_data);

	_AddTaskData_Free(add_task_data);
	_AddTaskData_Free(mul_task_data);

	// verify X = (X * 2) + 2
	TEST_ASSERT(X == 4);
}

static void test_cronAbort() {
	// reset X = 1
	// issue a task X += 2
	// abort task
	// validate X = 1

	X = 1;
	int Y = 2;
	AddTaskData data = _AddTaskData_New(add_task, (void *)&Y);

	// issue task X += 2
	CronTaskHandle task_handle = Cron_AddTask(15, _AddTaskData_Execute, NULL,
			&data);

	// abort task
	Cron_AbortTask(task_handle);

	// If we get here within the 0-15+ms interval, the cron thread
	// shouldn't have yet begun the task execution.
	TEST_ASSERT(!_AddTaskData_HasStarted(data));

	_AddTaskData_Free(data);

	// task should have been aborted prior to its execution
	// expecting X = 1
	TEST_ASSERT(X == 1);
}

static void test_cronLateAbort() {
	// reset X = 1
	// issue a task X += 2
	// abort task AFTER task been performed
	// validate X = 3

	X = 1;
	int Y = 2;
	AddTaskData data = _AddTaskData_New(add_task, (void *)&Y);

	// issue task X += 2
	CronTaskHandle task_handle = Cron_AddTask(15, _AddTaskData_Execute, NULL,
			&data);

	_AddTaskData_Wait(data);
	_AddTaskData_Free(data);

	// task should have been executed, expecting X = 1
	TEST_ASSERT(X == 3);

	// abort task, should note hang/crash
	Cron_AbortTask(task_handle);
}

static void test_MultiAbort() {
	// reset X = 1
	// issue a task X += 2
	// abort task multiple times
	// validate X = 1

	X = 1;
	int Y = 2;
	AddTaskData data = _AddTaskData_New(add_task, (void *)&Y);

	// issue task X += 2
	CronTaskHandle task_handle = Cron_AddTask(15, _AddTaskData_Execute, NULL,
			&data);

	// abort task multiple times, should not crash or hang
	for(int i = 0; i < 20; i++) {
		Cron_AbortTask(task_handle);
	}

	if (_AddTaskData_HasStarted(data)) {
		_AddTaskData_WaitForCompletion(data);
	}

	_AddTaskData_Free(data);

	// task should have been aborted prior to its execution
	// expecting X = 1
	TEST_ASSERT(X == 1);
}

static void test_abortNoneExistingTask() {
	// reset X = 1
	// issue a task X += 2
	// abort none existing task
	// validate X = 3

	X = 1;
	int Y = 2;
	AddTaskData data = _AddTaskData_New(add_task, (void*)&Y);

	// issue task X += 2
	CronTaskHandle task_handle = Cron_AddTask(15, _AddTaskData_Execute, NULL,
			&data);
	CronTaskHandle none_existing_task_handle = task_handle + 1;

	// abort task, should not crash hang
	Cron_AbortTask(none_existing_task_handle);

	_AddTaskData_Wait(data);
	_AddTaskData_Free(data);

	// task should have been executed
	// expecting X = 3
	TEST_ASSERT(X == 3);
}

static void test_AbortRunningTask() {
	// issue a long running task ~100ms
	// issue abort 20ms into execution
	// validate call to Cron_AbortTask returns after task compelted
	int ms = 100;
	AddTaskData data = _AddTaskData_New(long_running_task, (void*)&ms);
	double tic[2];
	simple_tic(tic);

	// issue a long running task, task will sleep for 100 'ms'
	CronTaskHandle task_handle = Cron_AddTask(0, _AddTaskData_Execute, NULL,
			&data);

	_AddTaskData_WaitForRunning(data);

	// the task should be already running
	// abort the task, call should return until the task completed.
	TEST_CHECK(!Cron_AbortTask(task_handle));

	// expecting Cron_AbortTask to return after task completed
	TEST_CHECK(simple_toc(tic) >= 0.100);

	_AddTaskData_Free(data);
}

TEST_LIST = {
	{"cronExec", test_cronExec},
	{"cronAbort", test_cronAbort},
	{"cronLateAbort", test_cronLateAbort},
	{"MultiAbort", test_MultiAbort},
	{"abortNoneExistingTask", test_abortNoneExistingTask},
	{"AbortRunningTask", test_AbortRunningTask},
	{NULL, NULL}
};

