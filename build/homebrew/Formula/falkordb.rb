class Falkordb < Formula
  desc "Ultra-fast, multi-tenant Graph Database for LLMs"
  homepage "https://www.falkordb.com/"
  url "https://github.com/FalkorDB/FalkorDB/archive/refs/tags/v4.14.5.tar.gz"
  sha256 "288adef0e914c5bdd9ed3d102134dd32847b680e1deccd352d3fae72bfc53ffe"
  license "SSPL-1.0"
  head "https://github.com/FalkorDB/FalkorDB.git", branch: "master"

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

  def install
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
