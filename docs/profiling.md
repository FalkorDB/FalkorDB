# Profiling using samply

> See also the `profile` agent skill (`.claude/skills/profile/SKILL.md`),
> which wraps this flow plus the benchmark suite; the `coverage` skill covers
> code-coverage reports.

To profile the code to find performance bottlenecks we use [samply](https://github.com/mstange/samply).

1. Install samply. On macOS, run `samply setup` once too, or samply hangs
   when it attaches to a running process
   ```
   cargo install --locked samply
   samply setup
   ```

2. Build the code in release mode with debug info. `[profile.profiling]` in
   `Cargo.toml` is release plus symbols; a plain `--release` build has none
   ```
   cargo build --profile profiling
   ```

3. Run the database under samply (`libfalkordb.so` on Linux)
   ```
   samply record redis-server --loadmodule ./target/profiling/libfalkordb.dylib
   ```

4. Run your query against the database

5. Stop the database and samply will open the Firefox Profiler UI in your browser
