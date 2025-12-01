# Test script to verify quick_commit functionality
Write-Host "==== test_quick_commit.ps1 ===="
Write-Host "Current git status:"
git status
Write-Host ""

# Check for changes
$changes = git status --porcelain
if ([string]::IsNullOrEmpty($changes)) {
    Write-Host "No changes to commit. Nothing to do."
    exit 0
}

# Stage all changes
Write-Host "Staging all changes..."
git add -A

# Check if there are staged changes
git diff --cached --quiet > $null 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "No staged changes to commit after git add. Nothing to do."
    exit 0
}

# Commit message
$commitMsg = "Test commit from PowerShell script"

# Commit changes
Write-Host "Committing changes..."
git commit -m "$commitMsg"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Git commit failed."
    exit 1
}

# Push changes
Write-Host "Pushing changes..."
git push
if ($LASTEXITCODE -ne 0) {
    Write-Host "Git push failed."
    exit 1
}

Write-Host "Commit and push completed successfully."