class Falkordb < Formula
  desc "Ultra-fast, multi-tenant Graph Database for LLMs"
  homepage "https://www.falkordb.com/"
  url "https://github.com/FalkorDB/FalkorDB/archive/refs/tags/v4.14.5.tar.gz"
  sha256 "288adef0e914c5bdd9ed3d102134dd32847b680e1deccd352d3fae72bfc53ffe"
  license "SSPL-1.0"
  head "https://github.com/FalkorDB/FalkorDB.git", branch: "master", using: :git

  depends_on "autoconf" => :build
  depends_on "automake" => :build
  depends_on "cmake" => :build
  depends_on "libtool" => :build
  depends_on "m4" => :build
  depends_on "peg" => :build
  depends_on "python@3.13" => :build
  depends_on "gcc" # For OpenMP support
  depends_on "libomp"
  depends_on "redis"

  on_macos do
    depends_on "make" => :build
  end

  # Submodule resources - these are required dependencies that are git submodules
  resource "rax" do
    url "https://github.com/antirez/rax.git",
        revision: "ba4529f6c836c9ff1296cde12b8557329f5530b7"
  end

  resource "xxHash" do
    url "https://github.com/Cyan4973/xxHash.git",
        revision: "bbb27a5efb85b92a0486cf361a8635715a53f6ba"
  end

  resource "RediSearch" do
    url "https://github.com/FalkorDB/RediSearch.git",
        revision: "297be5d4b6d5885f5f80b9fff66bddb248e38f70"
  end

  resource "utf8proc" do
    url "https://github.com/JuliaStrings/utf8proc.git",
        revision: "1cb28a66ca79a0845e99433fd1056257456cef8b"
  end

  resource "oniguruma" do
    url "https://github.com/kkos/oniguruma.git",
        revision: "28ee452bc98501b9b82e088948044fb810f2833c"
  end

  resource "libcsv" do
    url "https://github.com/rgamble/libcsv.git",
        revision: "b1d5212831842ee5869d99bc208a21837e4037d5"
  end

  resource "libcurl" do
    url "https://github.com/curl/curl.git",
        revision: "57495c64871d18905a0941db9196ef90bafe9a29"
  end

  resource "LAGraph" do
    url "https://github.com/GraphBLAS/LAGraph.git",
        revision: "50cd0c8f51a8925e3fae29f23e1d7b4d42893bee"
  end

  # Nested submodules within RediSearch
  resource "VectorSimilarity" do
    url "https://github.com/RedisLabsModules/VectorSimilarity.git",
        revision: "98d6a4d25bb7467ddbdce78155cc461e071bed44"
  end

  resource "googletest" do
    url "https://github.com/google/googletest.git",
        revision: "e2239ee6043f73722e7aa812a459f54a28552929"
  end

  resource "hiredis" do
    url "https://github.com/redis/hiredis.git",
        revision: "c14775b4e48334e0262c9f168887578f4a368b5d"
  end

  resource "libuv" do
    url "https://github.com/libuv/libuv.git",
        revision: "0c1fa696aa502eb749c2c4735005f41ba00a27b8"
  end

  resource "readies" do
    url "https://github.com/RedisLabsModules/readies.git",
        revision: "9f19bb2d56f5c60e842e758d8d1396b527031ea3"
  end

  def install
    # Extract top-level submodule resources into deps directory
    %w[rax xxHash RediSearch utf8proc oniguruma libcsv libcurl LAGraph].each do |dep|
      resource(dep).stage do
        (buildpath/"deps"/dep).install Dir["*"]
      end
    end

    # Extract nested RediSearch submodules
    %w[VectorSimilarity googletest hiredis libuv readies].each do |dep|
      resource(dep).stage do
        (buildpath/"deps/RediSearch/deps"/dep).install Dir["*"]
      end
    end

    ENV["LIBOMP"] = Formula["libomp"].opt_prefix
    ENV.append "LDFLAGS", "-L#{Formula["libomp"].opt_lib}"
    ENV.append "CPPFLAGS", "-I#{Formula["libomp"].opt_include}"

    # Use GNU make on macOS
    make_cmd = if OS.mac?
      "#{Formula["make"].opt_bin}/gmake"
    else
      "make"
    end

    system make_cmd
    
    # Find the built .so file (path varies by architecture)
    so_file = Dir.glob("bin/**/falkordb.so").first
    raise "Could not find falkordb.so" unless so_file
    
    lib.install so_file
  end

  def caveats
    <<~EOS
      To use FalkorDB with Redis, add the following to your redis.conf:
        loadmodule #{opt_lib}/falkordb.so

      Or start Redis with:
        redis-server --loadmodule #{opt_lib}/falkordb.so
    EOS
  end

  test do
    system "redis-server", "--version"
    assert_predicate lib/"falkordb.so", :exist?
  end
end
