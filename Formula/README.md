# FalkorDB Homebrew Formula

This directory contains the Homebrew formula for FalkorDB, an ultra-fast, multi-tenant Graph Database.

## For Users

### Installation

To install FalkorDB using Homebrew, you have two options:

#### Option 1: Install from this repository (once the tap is published)

```bash
brew tap falkordb/falkordb
brew install falkordb
```

#### Option 2: Install directly from this formula

```bash
brew install Formula/falkordb.rb
```

### Usage

After installation, you can use FalkorDB with Redis:

```bash
# Start Redis with FalkorDB module loaded
redis-server --loadmodule $(brew --prefix)/lib/falkordb.so
```

Or add to your `redis.conf`:

```
loadmodule /opt/homebrew/lib/falkordb.so
```

## For Maintainers

### Prerequisites

- A GitHub repository for the Homebrew tap: `FalkorDB/homebrew-falkordb` (optional but recommended)
- GitHub token with appropriate permissions for the tap repository

### Setting up the Homebrew Tap

1. Create a new repository named `homebrew-falkordb` under the FalkorDB organization
2. Initialize it with a `Formula/` directory
3. Set up the following GitHub secrets in the main FalkorDB repository:
   - `HOMEBREW_TAP_TOKEN`: Personal access token with write access to the tap repository
4. Set the following GitHub variable:
   - `HOMEBREW_TAP_ENABLED`: Set to `true` to enable automatic tap updates

### Automatic Formula Updates

The `.github/workflows/homebrew.yml` workflow automatically:

1. Triggers on new releases
2. Downloads the release tarball
3. Calculates the SHA256 checksum
4. Updates the formula with the new version and checksum
5. Creates a pull request in the main repository
6. (If tap is enabled) Pushes the updated formula to the tap repository

### Manual Formula Updates

To manually update the formula:

1. Update the version in the `url` field
2. Calculate and update the SHA256:
   ```bash
   curl -L -o release.tar.gz "https://github.com/FalkorDB/FalkorDB/archive/refs/tags/vX.Y.Z.tar.gz"
   shasum -a 256 release.tar.gz
   ```
3. Update the `sha256` field with the calculated hash

### Testing the Formula

#### Quick Test (Recommended)

Use the provided test script:

```bash
./scripts/test-homebrew-formula.sh
```

This interactive script will:
- Validate the formula syntax
- Check dependencies
- Help you install or build FalkorDB
- Run brew audit

#### Manual Testing

To test the formula manually:

```bash
brew install --build-from-source Formula/falkordb.rb
brew test falkordb
```

### Troubleshooting

#### Build Issues

- Ensure all dependencies are installed: `brew install cmake m4 automake peg libtool autoconf`
- Check that OpenMP is available: `brew install libomp`
- For GCC-related issues: `brew install gcc`

#### Runtime Issues

- Verify Redis is installed: `brew install redis`
- Check Redis version (requires 7.4+): `redis-server --version`
- Ensure the module file exists: `ls -l $(brew --prefix)/lib/falkordb.so`

## Contributing

If you encounter issues with the Homebrew formula, please open an issue in the main FalkorDB repository.

## License

The formula follows the same license as FalkorDB (SSPL-1.0).
