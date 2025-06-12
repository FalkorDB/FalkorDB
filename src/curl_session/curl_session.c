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
#include <fcntl.h>      // for fcntl(), O_NONBLOCK, F_SETFL, etc.
#include <unistd.h>
#include <pthread.h>
#include <sys/errno.h>



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

// curl write callback
// called when curl downloads data
// writes data to stream
static size_t _curl_write_cb
(
	void *ptr,     // data to write
	size_t size,   // size of each element
	size_t nmemb,  // number of elements
	void *pdata    // CurlSession
) {
	CurlSession session = (CurlSession)pdata;

	//--------------------------------------------------------------------------
	// check for readiness
	//--------------------------------------------------------------------------

	ssize_t nbytes = 0;

	struct pollfd pfd = {
		.fd = session->fd,
		.events = POLLOUT
	};

	// as long as download was aborted
	while(!session->abort) {
		int ret = poll(&pfd, 1, 50000);  // wait up to 50 second

		if (ret > 0 && (pfd.revents & POLLOUT)) {
			nbytes = write(session->fd, ptr, size * nmemb);
			if (nbytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				// Try again later
				continue;
			}
			break;
		} else if (ret == 0) {
			RedisModule_Log(NULL, "warning", "Timeout while waiting to write\n");
			break;
		} else {
			RedisModule_Log(NULL, "warning", "poll failed: %s\n", strerror(errno));
			break;
		}
	}

	return nbytes;
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

	// disable buffering on the stream
	int res = setvbuf(s->stream, NULL, _IONBF, 0);  // use unbuffered mode
	ASSERT(res == 0);

	fcntl(s->fd, F_SETFL, O_NONBLOCK);  // set non-blocking mode

	// take ownership over the stream
	*stream = NULL;

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

	// Skip SSL certificate verification
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

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

		intptr_t result = (intptr_t)thread_result;
		if(result != CURLE_OK) {
			//ErrorCtx_SetError("Error downloading file, error_code: %ld", result);
		}
	}

	// free curl handle
	if(_session->handle != NULL) {
		curl_easy_cleanup(_session->handle);
	}

	// free session and set to NULL
	rm_free(_session);
	*session = NULL;
}

