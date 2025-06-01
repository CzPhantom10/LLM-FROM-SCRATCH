# PowerShell setup script for LLM from scratch project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LLM FROM SCRATCH - QUICK SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n1. Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "myenv") {
    Write-Host "✓ Virtual environment exists. Activating..." -ForegroundColor Green
    & "myenv\Scripts\Activate.ps1"
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv myenv
    & "myenv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment created and activated." -ForegroundColor Green
}

Write-Host "`n3. Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n4. Testing installation..." -ForegroundColor Yellow
$torchTest = python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1
$numpyTest = python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $torchTest" -ForegroundColor Green
    Write-Host "✓ $numpyTest" -ForegroundColor Green
} else {
    Write-Host "✗ Package import failed. Check installation." -ForegroundColor Red
}

Write-Host "`n5. Running quick examples..." -ForegroundColor Yellow
python examples.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nQuick commands to try:" -ForegroundColor Yellow
Write-Host "  python data_loader.py                    # Download data" -ForegroundColor White
Write-Host "  python train.py --model_type mlp         # Train MLP" -ForegroundColor White
Write-Host "  python train.py --model_type transformer # Train Transformer" -ForegroundColor White
Write-Host "  python generate.py --model_path models/best_model.pt --interactive" -ForegroundColor White
Write-Host "  python demo.py                           # Quick demo" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
