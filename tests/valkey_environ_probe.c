/*
 * Reports whether a module loaded by this valkey-server sees a NULL `environ`.
 *
 * That is the condition #3493 needs: the server dlopen()s modules with
 * RTLD_DEEPBIND, so a module binds `environ` to libc's own definition, and the
 * executable copy-relocates `environ`, so that definition is never set. Without
 * either one, the module sees the live environment and an unfixed engine loads
 * fine, so the valkey-smoke CI job runs this first to show it can still tell a
 * fixed engine from an unfixed one.
 *
 * It prints one line and then refuses to load, which makes the server exit.
 *
 * Build: cc -shared -fPIC -o probe.so tests/valkey_environ_probe.c
 */
#include <stdio.h>

extern char **environ;

int RedisModule_OnLoad(void *ctx, void **argv, int argc) {
    (void)ctx;
    (void)argv;
    (void)argc;
    fprintf(stderr, "environ-probe: %s\n", environ == NULL ? "NULL" : "set");
    return 1; /* REDISMODULE_ERR */
}
