# Homebrew Setup Checklist

This checklist will guide you through the one-time setup process for publishing FalkorDB via Homebrew.

## Prerequisites

- [ ] GitHub organization admin access for FalkorDB
- [ ] Ability to create GitHub repositories
- [ ] Ability to create GitHub secrets and variables
- [ ] Access to a macOS machine for testing (optional but recommended)

## Step 1: Create the Tap Repository

- [ ] Create a new public repository: `FalkorDB/homebrew-falkordb`
  - Go to https://github.com/organizations/FalkorDB/repositories/new
  - Repository name: `homebrew-falkordb`
  - Description: "Homebrew tap for FalkorDB"
  - Visibility: Public
  - Initialize with README: Yes
  - Click "Create repository"

- [ ] Initialize the tap repository structure
  ```bash
  git clone https://github.com/FalkorDB/homebrew-falkordb.git
  cd homebrew-falkordb
  mkdir Formula
  cp ../FalkorDB/Formula/falkordb.rb Formula/
  git add Formula/falkordb.rb
  git commit -m "Add FalkorDB formula"
  git push
  ```

## Step 2: Create GitHub Personal Access Token

- [ ] Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- [ ] Click "Generate new token (classic)"
- [ ] Token settings:
  - Note: `Homebrew Tap Update Token`
  - Expiration: Set to "No expiration" or a long duration
  - Scopes: Select `repo` (Full control of private repositories)
- [ ] Click "Generate token"
- [ ] **IMPORTANT**: Copy the token immediately (you won't see it again)

## Step 3: Configure Secrets in Main Repository

- [ ] Go to the main FalkorDB repository
- [ ] Navigate to Settings → Secrets and variables → Actions
- [ ] Click "New repository secret"
  - Name: `HOMEBREW_TAP_TOKEN`
  - Value: Paste the token you created in Step 2
  - Click "Add secret"

## Step 4: Enable Tap Updates

- [ ] In the same Settings → Secrets and variables → Actions page
- [ ] Click on the "Variables" tab
- [ ] Click "New repository variable"
  - Name: `HOMEBREW_TAP_ENABLED`
  - Value: `true`
  - Click "Add variable"

## Step 5: Test the Setup

- [ ] Trigger a test workflow run:
  - Go to Actions → "Update Homebrew Formula"
  - Click "Run workflow"
  - Enter the latest version (e.g., `v4.14.5`)
  - Click "Run workflow"

- [ ] Verify the workflow completes successfully
- [ ] Check that a PR was created in the main repository
- [ ] Verify the formula was updated in the tap repository

## Step 6: Test Installation (Optional but Recommended)

If you have access to a macOS machine:

- [ ] Test installing from the tap:
  ```bash
  brew tap falkordb/falkordb
  brew install falkordb
  ```

- [ ] Verify the installation:
  ```bash
  redis-server --loadmodule $(brew --prefix)/lib/falkordb.so &
  redis-cli GRAPH.QUERY test "CREATE (:Person {name: 'Test'})"
  redis-cli GRAPH.QUERY test "MATCH (p:Person) RETURN p.name"
  redis-cli SHUTDOWN
  ```

## Step 7: Update Documentation

- [ ] Add a note to the main README about Homebrew availability
- [ ] Update installation documentation on the website
- [ ] Announce the new installation method in the community

## Verification Checklist

- [ ] The tap repository (`homebrew-falkordb`) exists and is public
- [ ] The formula file is in the correct location: `Formula/falkordb.rb`
- [ ] The `HOMEBREW_TAP_TOKEN` secret is configured
- [ ] The `HOMEBREW_TAP_ENABLED` variable is set to `true`
- [ ] The workflow runs successfully on new releases
- [ ] The formula can be installed using `brew install`
- [ ] Redis can load the FalkorDB module successfully

## Troubleshooting

### Workflow fails with authentication error
- Check that the `HOMEBREW_TAP_TOKEN` has the correct permissions
- Verify the token hasn't expired
- Ensure the token has access to the `homebrew-falkordb` repository

### Formula installation fails
- Check the formula syntax: `ruby -c Formula/falkordb.rb`
- Verify all dependencies are available
- Review build logs for specific errors
- Test locally using `./scripts/test-homebrew-formula.sh`

### SHA256 mismatch error
- The tarball might have changed
- Re-download and recalculate: `curl -L -o test.tar.gz "URL" && shasum -a 256 test.tar.gz`
- Update the formula with the correct hash

### Redis can't load the module
- Verify Redis version is 7.4 or higher
- Check that OpenMP is installed: `brew install libomp`
- Verify the module path is correct

## Maintenance

### Regular Tasks
- Monitor workflow runs on new releases
- Review and merge automated PRs
- Test formula updates periodically
- Keep documentation up-to-date

### When Things Change
- If build process changes: Update the formula's `install` method
- If dependencies change: Update the `depends_on` declarations
- If Redis version requirement changes: Update the formula description and docs

## Next Steps

After completing this checklist:
1. Monitor the next release to ensure automation works
2. Gather feedback from users installing via Homebrew
3. Consider submitting to Homebrew core for wider distribution

## Questions?

- Check the detailed guide: `.github/HOMEBREW_TAP_SETUP.md`
- Review the formula README: `Formula/README.md`
- Open an issue in the FalkorDB repository
