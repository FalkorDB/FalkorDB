//! The archive that got linked is the one the bindings were generated from.

use std::{ffi::CStr, os::raw::c_char};

use lagraph_sys::{
    LAGRAPH_DATE, LAGRAPH_MSG_LEN, LAGRAPH_VERSION_MAJOR, LAGRAPH_VERSION_MINOR,
    LAGRAPH_VERSION_UPDATE, LAGraph_Version,
};

#[test]
fn linked_library_reports_the_bound_version() {
    let mut version = [0; 3];
    let mut date = [0 as c_char; LAGRAPH_MSG_LEN as usize];
    let mut msg = [0 as c_char; LAGRAPH_MSG_LEN as usize];
    // SAFETY: three ints, and two buffers of `LAGRAPH_MSG_LEN` bytes, which is
    // what `LAGraph_Version` documents for `version`, `date` and `msg`. It
    // needs no `LAGraph_Init`.
    let status =
        unsafe { LAGraph_Version(version.as_mut_ptr(), date.as_mut_ptr(), msg.as_mut_ptr()) };
    assert_eq!(status, 0);
    assert_eq!(
        version.map(|part| u32::try_from(part).expect("a version part is not negative")),
        [
            LAGRAPH_VERSION_MAJOR,
            LAGRAPH_VERSION_MINOR,
            LAGRAPH_VERSION_UPDATE
        ]
    );
    // SAFETY: `LAGraph_Version` wrote a NUL-terminated string.
    let date = unsafe { CStr::from_ptr(date.as_ptr()) };
    assert_eq!(date.to_bytes_with_nul(), LAGRAPH_DATE);
}
