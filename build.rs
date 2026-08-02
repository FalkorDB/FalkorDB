fn main() {
    let major: u32 = std::env::var("CARGO_PKG_VERSION_MAJOR")
        .unwrap()
        .parse()
        .unwrap();
    let minor: u32 = std::env::var("CARGO_PKG_VERSION_MINOR")
        .unwrap()
        .parse()
        .unwrap();
    let patch: u32 = std::env::var("CARGO_PKG_VERSION_PATCH")
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        minor < 100 && patch < 100,
        "minor and patch must be < 100 for the MAJOR*10000 + MINOR*100 + PATCH encoding"
    );
    let version_int = major
        .checked_mul(10000)
        .and_then(|v| v.checked_add(minor * 100))
        .and_then(|v| v.checked_add(patch))
        .expect("version encoding overflowed u32");
    assert!(
        i32::try_from(version_int).is_ok(),
        "encoded version {version_int} exceeds i32::MAX; redis_module! `version:` takes i32"
    );
    println!("cargo:rustc-env=FALKORDB_VERSION_INT={version_int}");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
