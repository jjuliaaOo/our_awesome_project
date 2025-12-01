:: File: scripts/quick_commit.bat
:: Usage:
::   scripts\quick_commit.bat
::   scripts\quick_commit.bat "your commit message"
:: Description:
::   Stages all changes, prompts for a commit message if needed, creates a commit, and pushes to the current branch.

@echo off
setlocal enabledelayedexpansion

echo ==== quick_commit.bat ====
echo Current git status:
git status
echo.

:: Check for changes
for /f %%i in ('git status --porcelain ^| find /c /v ""') do set COUNT=%%i

if "%COUNT%"=="0" (
    echo No changes to commit. Nothing to do.
    endlocal
    exit /b 0
)

:: Stage all changes
git add -A

:: Check if there are staged changes
git diff --cached --quiet >nul 2>&1
if %errorlevel% EQU 0 (
    echo No staged changes to commit after git add. Nothing to do.
    endlocal
    exit /b 0
)

:: Determine commit message
set COMMIT_MSG=%*
if "%COMMIT_MSG%"=="" (
    echo Enter commit message:
    set /p COMMIT_MSG="> "
)
if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=auto-commit
)

:: Commit changes
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo Git commit failed.
    endlocal
    exit /b 1
)

:: Push changes
git push
if errorlevel 1 (
    echo Git push failed.
    endlocal
    exit /b 1
)

echo Commit and push completed successfully.
endlocal
exit /b 0