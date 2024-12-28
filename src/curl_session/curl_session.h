/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

typedef struct Opaque_CurlSession *CurlSession;

// asynchonously download a file from the internet
// populates the stream with the downloaded file
// returns a session handle
CurlSession Curl_Download
(
	const char* url,  // URL to download
	FILE* stream      // output stream
);

// aborts the download
void Curl_Abort
(
	CurlSession session  // session handle
);

// frees the session handle
void Curl_Free
(
	CurlSession *session  // session handle
);

