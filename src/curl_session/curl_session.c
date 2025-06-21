/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "curl.h"
#include "curl_session.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"

#include <poll.h>
#include <errno.h>
#include <fcntl.h>      // for fcntl(), O_NONBLOCK, F_SETFL, etc.
#include <unistd.h>
#include <pthread.h>



// curl download session
struct Opaque_CurlSession {
	CURL *handle;         // curl handle
	int fd;               // stream file discriptor
	FILE *stream;         // output stream
	volatile bool abort;  // download abort flag
	pthread_t thread;     // download thread
};

// aborts the download
static void _Curl_Abort
(
	CurlSession session  // session handle
) {
	ASSERT(session != NULL);

	// set abort flag
	session->abort = true;
}

// Curl write callback
// called by libcurl when data is downloaded
// this function writes the received data into a file descriptor (e.g. a pipe)
// used by another thread or process to consume the data
static size_t _curl_write_cb
(
    void *ptr,     // pointer to the data to be written
    size_t size,   // size of each element
    size_t nmemb,  // number of elements
    void *pdata    // pointer to CurlSession containing the target fd
) {
    CurlSession session = (CurlSession)pdata;
    char *buf = (char *)ptr;

    ssize_t total_written = 0;
    ssize_t remaining     = size * nmemb;

    struct pollfd pfd = {
        .fd     = session->fd,
        .events = POLLOUT
    };

    // attempt to write all data unless download is aborted
    while (!session->abort && remaining > 0) {
		// wait up to 5 seconds for the fd to be writable
        int ret = poll(&pfd, 1, 5000);

        if (ret > 0 && (pfd.revents & POLLOUT)) {
            ssize_t nbytes = write(session->fd, buf, remaining);

            if (nbytes < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // temporary condition, retry
                    continue;
                }

                // log and abort on unrecoverable error
                RedisModule_Log(NULL, "warning", "Write error: %s", strerror(errno));
                break;
            }

            // update buffer position and remaining byte count
            buf            += nbytes;
            remaining      -= nbytes;
            total_written  += nbytes;
        }
        else if (ret == 0) {
            // timeout, presumably read end is lagging behind
            RedisModule_Log(NULL, "warning", "Timeout while waiting to write");
            continue;
        }
        else {
            // poll() failed
            RedisModule_Log(NULL, "warning", "poll failed: %s", strerror(errno));
            break;
        }
    }

    // return number of bytes successfully written
    // if not all bytes were written, libcurl may treat this as a failure
	return (size_t)total_written;
}

// curl thread
// downloads file in a separate thread
// closes write end of pipe when download is complete
static void *_curl_thread
(
	void *pdata  // CurlSession
) {
	CurlSession session = (CurlSession)pdata;

	// start curl download
	CURLcode res = curl_easy_perform(session->handle);

	// close write end of pipe
	fclose(session->stream);
	session->stream = NULL;

	if(res != CURLE_OK && !session->abort) {
		// if download was not aborted, set thread exit code to 'res'
		return (void *)(intptr_t)res;
	}

	return (void*)(intptr_t)CURLE_OK;
}

// asynchonously download a file from the internet
// populates the stream with the downloaded file
// returns a session handle
CurlSession Curl_Download
(
	const char *url,  // URL to download
	FILE **stream     // output stream
) {
	ASSERT(url    != NULL);
	ASSERT(stream != NULL && *stream != NULL);

	CurlSession s = rm_calloc(1, sizeof(struct Opaque_CurlSession));

	// take ownership over the stream
	s->fd     = fileno(*stream);
	s->stream = *stream;

	// take ownership over the stream
	*stream = NULL;

	// disable buffering on the stream
	int res = setvbuf(s->stream, NULL, _IONBF, 0);  // use unbuffered mode
	assert(res == 0);

	fcntl(s->fd, F_SETFL, O_NONBLOCK);  // set non-blocking mode

	// create a new session
	CURL *curl = curl_easy_init();
	if(curl == NULL) {
		ErrorCtx_SetError("Error initializing curl");
		goto error;
	}

	// set curl options
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, s);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _curl_write_cb);
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);      // treat HTTP errors as failures
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);  // establish connection timeout

	s->handle = curl;

	// download file in a separate thread
	if(pthread_create(&(s->thread), NULL, _curl_thread, (void*)s) != 0) {
		ErrorCtx_SetError("Error creating thread");
		goto error;
	}

	return s;

error:
	fclose(s->stream);
	s->stream = NULL;

	Curl_Free(&s);

	return NULL;
}

// frees the session handle
void Curl_Free
(
	CurlSession *session  // session handle
) {
	ASSERT(session != NULL && *session != NULL);

	CurlSession _session = *session;

	//--------------------------------------------------------------------------
	// abort download
	//--------------------------------------------------------------------------

	if(_session->stream != NULL) {
		// abort download and close stream
		_Curl_Abort(_session);
	}

	// wait for download thread to exit, OK if thread already existed
	if(_session->thread != 0) {
		void *thread_result;
		pthread_join(_session->thread, &thread_result);
		ASSERT(_session->stream == NULL);
	}

	// free curl handle
	if(_session->handle != NULL) {
		curl_easy_cleanup(_session->handle);
	}

	// free session and set to NULL
	rm_free(_session);
	*session = NULL;
}

