# Build Script for Modular Embeddings Presentation
# Usage: .\build.ps1 [-Clean] [-Fast] [-Output <filename>]

param(
    [switch]$Clean = $false,
    [switch]$Fast = $false,
    [string]$Output = "embeddings_presentation"
)

# Configuration
$MainFile = "main.tex"
$BuildDir = "build"
$FinalOutput = "$Output.pdf"

# Colors for output
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Create build directory if needed
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
    Write-Info "Created build directory: $BuildDir"
}

# Clean if requested
if ($Clean) {
    Write-Info "Cleaning temporary files..."
    Remove-Item -Path "$BuildDir\*" -Include "*.aux", "*.log", "*.nav", "*.out", "*.snm", "*.toc", "*.vrb", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.synctex.gz" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "*" -Include "*.aux", "*.log", "*.nav", "*.out", "*.snm", "*.toc", "*.vrb", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.synctex.gz" -Force -ErrorAction SilentlyContinue
    Write-Success "Cleanup complete"
}

# Compile presentation
Write-Info "Starting LaTeX compilation..."

if ($Fast) {
    # Single pass for fast preview
    Write-Info "Fast mode: Single compilation pass"
    $result = pdflatex -interaction=nonstopmode -output-directory=$BuildDir -jobname=$Output $MainFile 2>&1
} else {
    # Full compilation with multiple passes
    Write-Info "Full mode: Multiple compilation passes"
    
    # First pass
    Write-Info "Pass 1/3..."
    $result1 = pdflatex -interaction=nonstopmode -output-directory=$BuildDir -jobname=$Output $MainFile 2>&1
    
    # Second pass (for references)
    Write-Info "Pass 2/3..."
    $result2 = pdflatex -interaction=nonstopmode -output-directory=$BuildDir -jobname=$Output $MainFile 2>&1
    
    # Third pass (final)
    Write-Info "Pass 3/3..."
    $result3 = pdflatex -interaction=nonstopmode -output-directory=$BuildDir -jobname=$Output $MainFile 2>&1
}

# Check for successful compilation
if (Test-Path "$BuildDir\$FinalOutput") {
    # Copy to main directory
    Copy-Item -Path "$BuildDir\$FinalOutput" -Destination $FinalOutput -Force
    Write-Success "✓ Compilation successful!"
    Write-Success "Output: $FinalOutput"
    
    # Get file info
    $fileInfo = Get-Item $FinalOutput
    Write-Info "File size: $([math]::Round($fileInfo.Length / 1MB, 2)) MB"
    
    # Count pages (if pdfinfo is available)
    if (Get-Command pdfinfo -ErrorAction SilentlyContinue) {
        $pages = (pdfinfo $FinalOutput | Select-String "Pages:").ToString().Split()[1]
        Write-Info "Total pages: $pages"
    }
    
    # Prompt to open
    $response = Read-Host "Open PDF? (Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Start-Process $FinalOutput
    }
} else {
    Write-Error "✗ Compilation failed!"
    Write-Error "Check $BuildDir\$Output.log for details"
    
    # Show last few errors from log
    if (Test-Path "$BuildDir\$Output.log") {
        Write-Info "`nLast errors from log:"
        Get-Content "$BuildDir\$Output.log" | Select-String -Pattern "Error|Warning" | Select-Object -Last 5
    }
    
    exit 1
}

# Optional: Show warnings
$showWarnings = Read-Host "Show warnings? (Y/N)"
if ($showWarnings -eq 'Y' -or $showWarnings -eq 'y') {
    Write-Info "`nWarnings from compilation:"
    Get-Content "$BuildDir\$Output.log" | Select-String -Pattern "Warning" | Select-Object -Unique
}