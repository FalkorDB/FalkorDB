# Homebrew Tap Setup Guide

This guide explains how to set up and maintain the FalkorDB Homebrew tap.

## Initial Setup

### 1. Create the Tap Repository

Create a new repository named `homebrew-falkordb` under the FalkorDB organization:

```bash
# Create the repository on GitHub (can be done via UI or CLI)
gh repo create FalkorDB/homebrew-falkordb --public --description "Homebrew tap for FalkorDB"

# Clone and initialize the repository
git clone https://github.com/FalkorDB/homebrew-falkordb.git
cd homebrew-falkordb

# Create the Formula directory
mkdir Formula

# Copy the initial formula
cp ../FalkorDB/build/homebrew/Formula/falkordb.rb Formula/

# Create a README
cat > README.md << 'EOF'
# Homebrew FalkorDB

Official Homebrew tap for FalkorDB.

## Installation

```bash
brew tap falkordb/falkordb
brew install falkordb
```

## Usage

After installation, start Redis with FalkorDB:

```bash
redis-server --loadmodule $(brew --prefix)/lib/falkordb.so
```

For more information, visit [FalkorDB](https://www.falkordb.com/).
EOF

# Commit and push
git add .
git commit -m "Initial commit: Add FalkorDB formula"
git push origin main
```

### 2. Configure GitHub Secrets

In the main FalkorDB repository, configure the following secrets:

1. Go to Settings → Secrets and variables → Actions
2. Add a new repository secret:
   - Name: `HOMEBREW_TAP_TOKEN`
   - Value: A GitHub Personal Access Token with `repo` scope that has write access to the `homebrew-falkordb` repository

To create a Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name like "Homebrew Tap Update Token"
4. Select the `repo` scope
5. Click "Generate token"
6. Copy the token and add it as the secret

### 3. Enable the Tap Updates

In the main FalkorDB repository:

1. Go to Settings → Secrets and variables → Actions → Variables
2. Add a new repository variable:
   - Name: `HOMEBREW_TAP_ENABLED`
   - Value: `true`

## How It Works

### Automatic Updates

The `.github/workflows/homebrew.yml` workflow automatically updates the formula when:

1. A new release is published
2. The workflow is manually triggered with a version number

The workflow:
1. Downloads the release tarball
2. Calculates the SHA256 checksum
3. Updates the formula in the main repository (creates a PR)
4. If `HOMEBREW_TAP_ENABLED` is true, pushes the updated formula to the tap repository

### Manual Updates

To manually update the formula:

```bash
# In the main FalkorDB repository
cd build/homebrew/Formula

# Update the version and SHA256 in falkordb.rb
# Then test locally
brew install --build-from-source ./falkordb.rb
brew test falkordb

# Copy to the tap repository
cp falkordb.rb ../../../homebrew-falkordb/Formula/

# Commit and push in the tap repository
cd ../../../homebrew-falkordb
git add Formula/falkordb.rb
git commit -m "Update FalkorDB to version X.Y.Z"
git push
```

## Testing

### Test the Formula Locally

```bash
# Install from the formula file
brew install --build-from-source build/homebrew/Formula/falkordb.rb

# Run the test
brew test falkordb

# Verify the installation
ls -l $(brew --prefix)/lib/falkordb.so

# Test with Redis
redis-server --loadmodule $(brew --prefix)/lib/falkordb.so &
redis-cli GRAPH.QUERY test "CREATE (:Person {name: 'Alice'})"
redis-cli GRAPH.QUERY test "MATCH (p:Person) RETURN p.name"
redis-cli SHUTDOWN
```

### Test the Tap

```bash
# Add the tap
brew tap falkordb/falkordb

# Install from the tap
brew install falkordb

# Verify
brew info falkordb
```

## Troubleshooting

### Common Issues

1. **SHA256 mismatch**: Ensure the SHA256 hash in the formula matches the actual release tarball
   ```bash
   curl -L -o test.tar.gz "https://github.com/FalkorDB/FalkorDB/archive/refs/tags/vX.Y.Z.tar.gz"
   shasum -a 256 test.tar.gz
   ```

2. **Build failures**: Check that all dependencies are correctly specified in the formula

3. **Module loading fails**: Ensure Redis is version 7.4 or higher and OpenMP is installed

4. **Token permissions**: Verify the `HOMEBREW_TAP_TOKEN` has write access to the tap repository

### Getting Help

- Check the [Homebrew documentation](https://docs.brew.sh/)
- Review the [Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- Open an issue in the FalkorDB repository

## Maintenance

### Regular Tasks

1. **Monitor releases**: Ensure the workflow successfully updates the formula for each release
2. **Review PRs**: Check and merge pull requests created by the workflow
3. **Test updates**: Periodically test the formula to ensure it still builds correctly
4. **Update dependencies**: Keep dependency versions up-to-date in the formula

### Version Updates

The workflow handles version updates automatically. Manual intervention is only needed if:
- The build process changes significantly
- New dependencies are added
- The installation path changes

## Publishing to Homebrew Core (Optional)

If you want to publish FalkorDB to the official Homebrew repository:

1. Ensure the formula meets all [Homebrew core requirements](https://docs.brew.sh/Acceptable-Formulae)
2. Test thoroughly on multiple macOS versions
3. Follow the [contribution guidelines](https://github.com/Homebrew/homebrew-core/blob/master/CONTRIBUTING.md)
4. Submit a PR to [homebrew-core](https://github.com/Homebrew/homebrew-core)

Note: This is a more involved process and requires ongoing maintenance.
