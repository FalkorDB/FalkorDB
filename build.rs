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
    let version_int = major * 10000 + minor * 100 + patch;
    println!("cargo:rustc-env=FALKORDB_VERSION_INT={version_int}");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
