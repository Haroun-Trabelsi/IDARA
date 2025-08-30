# PowerShell script to make shell scripts executable on Windows
# This script sets the execution policy and prepares scripts for WSL/Linux environments

Write-Host "Making deployment scripts executable..." -ForegroundColor Green

# Get all shell scripts in the current directory
$scripts = Get-ChildItem -Path "." -Filter "*.sh"

foreach ($script in $scripts) {
    Write-Host "Processing: $($script.Name)" -ForegroundColor Yellow
    
    # Ensure Unix line endings (LF instead of CRLF)
    $content = Get-Content $script.FullName -Raw
    $content = $content -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($script.FullName, $content)
    
    Write-Host "  ✓ Fixed line endings" -ForegroundColor Green
}

Write-Host "`nAll scripts processed!" -ForegroundColor Green
Write-Host "Note: On Linux/WSL, run 'chmod +x *.sh' to make scripts executable" -ForegroundColor Cyan
