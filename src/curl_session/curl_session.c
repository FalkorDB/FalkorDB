/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "curl_session.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"

#include <pthread.h>
#include <curl/curl.h>

// curl download session
struct Opaque_CurlSession {
	CURL *handle;  // curl handle
	FILE *stream;  // output stream
	bool abort;    // download abort flag
};

// curl write callback
// called when curl downloads data
// writes data to stream
static size_t _curl_write_cb
(
	void *ptr,     // data to write
	size_t size,   // size of each element
	size_t nmemb,  // number of elements
	void *pdata    // stream to write to
) {
	CurlSession session = (CurlSession)pdata;

	// check if download was aborted
	if(session->abort) {
		return 0;  // 0 indicates to curl that download was aborted
	}

	// write data to stream
	return fwrite(ptr, size, nmemb, session->stream);
}

// curl thread
// downloads file in a separate thread
// closes write end of pipe when download is complete
static void *_curl_thread
(
	void *pdata  // CSVReader
) {
	CurlSession session = (CurlSession)pdata;

	// start curl download
	CURLcode res = curl_easy_perform(session->handle);
	if(res != CURLE_OK) {
		if(!session->abort) {
			// if download was not aborted, raise an exception
			ErrorCtx_RaiseRuntimeException("Error downloading file");
		}
	}

	// close write end of pipe
	fclose(session->stream);
	session->stream = NULL;

	return NULL;
}

// asynchonously download a file from the internet
// populates the stream with the downloaded file
// returns a session handle
CurlSession Curl_Download
(
	const char *url,  // URL to download
	FILE *stream      // output stream
) {
	ASSERT(url    != NULL);
	ASSERT(stream != NULL);

	CurlSession session = rm_calloc(1, sizeof(struct Opaque_CurlSession));

	// create a new session
	CURL *curl = curl_easy_init();
	if(curl == NULL) {
		ErrorCtx_SetError("Error initializing curl");
		goto error;
	}

	// set curl options
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, session);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _curl_write_cb);

	session->handle = curl;
	session->stream = stream;

	// download file in a separate thread
	pthread_t thread;
	if(pthread_create(&thread, NULL, _curl_thread, (void*)session) != 0) {
		ErrorCtx_SetError("Error creating thread");
		goto error;
	}

	return session;

error:
	fclose(stream);
	Curl_Free(&session);
	return NULL;
}

// aborts the download
void Curl_Abort
(
	CurlSession session  // session handle
) {
	ASSERT(session != NULL);

	// set abort flag
	session->abort = true;
}

// frees the session handle
void Curl_Free
(
	CurlSession *session  // session handle
) {
	ASSERT(session != NULL && *session != NULL);

	CurlSession _session = *session;

	// download is still in progress
	if(_session->stream != NULL) {
		// abort download and close stream
		Curl_Abort(_session);
	}

	// busy wait for download thread to exit
	while(_session->stream != NULL);

	// free curl handle
	if(_session->handle != NULL) {
		curl_easy_cleanup(_session->handle);
	}

	// free session and set to NULL
	rm_free(_session);
	*session = NULL;
}

