# ============================================================
# Auto-detect and commit notebooks in each weekX folder
# Usage: Run from C:\Users\msb80\Documents\zTh
#   .\commit_notebooks.ps1
# ============================================================

Set-Location "C:\Users\msb80\Documents\zTh"

$committed = 0
$folders = @("week1","week2","week3","week4","week5","week6","week6.5","week7","week7.5","week8","week9","week10")

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        Write-Host "  [$folder] Folder not found, skipping" -ForegroundColor Yellow
        continue
    }

    # Check for any new or modified files in this week folder
    $changes = git status --porcelain -- $folder
    
    if ($changes) {
        Write-Host "`n  [$folder] Changes detected:" -ForegroundColor Cyan
        $changes | ForEach-Object { Write-Host "    $_" }
        
        # Count notebooks specifically
        $nbCount = (Get-ChildItem -Path $folder -Recurse -Filter "*.ipynb" -File).Count
        
        git add "$folder/*"
        git commit -m "${folder}: Add/update notebooks ($nbCount notebook(s) total)"
        $committed++
        
        Write-Host "  [$folder] Committed!" -ForegroundColor Green
    } else {
        Write-Host "  [$folder] No changes" -ForegroundColor DarkGray
    }
}

if ($committed -gt 0) {
    Write-Host "`n Pushing all commits to GitHub..." -ForegroundColor Cyan
    git push origin main
    Write-Host " Done! $committed week(s) committed and pushed." -ForegroundColor Green
} else {
    Write-Host "`n No changes detected in any week folder." -ForegroundColor Yellow
}
