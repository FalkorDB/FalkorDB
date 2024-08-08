
#include <stdio.h>
#include <stdlib.h>

#include "readies/cetara/diag/gdb.h"

#if (defined(DEBUG) || defined(_DEBUG)) && !defined(NDEBUG)

#ifdef __linux__

#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/stat.h>

#elif defined(__APPLE__)

#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/sysctl.h>

#endif

///////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __linux__

static inline bool _via_gdb_simple()
{
    const int status_fd = open("/proc/self/status", O_RDONLY);
    if (status_fd == -1)
        return false;

    char buf[4096];
    const ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);
    if (num_read <= 0)
        return false;

    buf[num_read] = '\0';
    char tracer_pid[] = "TracerPid:";
    const char *tracer_pid_p = strstr(buf, tracer_pid);
    if (!tracer_pid_p)
        return false;

    for (const char *p = tracer_pid_p + sizeof(tracer_pid) - 1; p <= buf + num_read; ++p)
    {
        if (isspace(*p))
            continue;
        return isdigit(*p) && *p != '0';
    }

    return false;
}

static inline bool _via_gdb()
{
    if (_via_gdb_simple())
        return true;

    int pid;
    int from_child[2] = {-1, -1};

    if (pipe(from_child) < 0) {
        fprintf(stderr, "Debugger check failed: Error opening internal pipe: %s", strerror(errno));
        return false;
    }

    pid = fork();
    if (pid == -1) {
        fprintf(stderr, "Debugger check failed: Error forking: %s", strerror(errno));
        return false;
    }

    if (pid == 0) // child
    {
        uint8_t ret = 0;
        int ppid = getppid();

        close(from_child[0]); // close parent's side

        if (ptrace(PTRACE_ATTACH, ppid, NULL, NULL) == 0) 
        {
            waitpid(ppid, NULL, 0); // wait for the parent to stop
            write(from_child[1], &ret, sizeof(ret)); // tell the parent what happened

            ptrace(PTRACE_DETACH, ppid, NULL, NULL);
            _exit(0);
        }

        ret = 1;
        write(from_child[1], &ret, sizeof(ret)); // tell the parent what happened
        _exit(0);
    }
    else // parent
    { 
        uint8_t ret = -1;

        // child writes a 1 if pattach failed else 0.
        // read may be interrupted by pattach, hence the loop.
        while (read(from_child[0], &ret, sizeof(ret)) < 0 && errno == EINTR)
            ;

        // ret not updated
        if (ret < 0)
            fprintf(stderr, "Debugger check failed: Error getting status from child: %s", strerror(errno));

        // close the pipes here, to avoid races with pattach (if we did it above)
        close(from_child[1]);
        close(from_child[0]);

        waitpid(pid, NULL, 0); // collect the status of the child

        return ret == 1;
    }
}

#elif defined(__APPLE__)

static inline bool _via_gdb()
{
    struct kinfo_proc info;

    // initialize the flags so that, if sysctl fails for some bizarre reason,
    // we get a predictable result
    info.kp_proc.p_flag = 0;

    // initialize mib, which tells sysctl the info we want, in this case 
    // we're looking for information about a specific process ID
    int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid() };

    // call sysctl
    size_t size = sizeof(info);
    int junk = sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);
    assert(!junk);

    // we're being debugged if the P_TRACED flag is set.
    return (info.kp_proc.p_flag & P_TRACED) != 0;
}

#endif

//---------------------------------------------------------------------------------------------

bool __via_gdb = false;

__attribute__((constructor))
static void initialize(void) 
{
    __via_gdb = _via_gdb();
}

///////////////////////////////////////////////////////////////////////////////////////////////

#endif // DEBUG
