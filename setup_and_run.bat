@echo off
REM Quick setup script for LLM from scratch project
REM This script helps beginners get started quickly

echo ========================================
echo LLM FROM SCRATCH - QUICK SETUP
echo ========================================

echo.
echo 1. Activating virtual environment...
if exist "myenv\Scripts\activate.bat" (
    call myenv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Creating one...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Virtual environment created and activated.
)

echo.
echo 2. Installing dependencies...
pip install -r requirements.txt

echo.
echo 3. Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo.
echo 4. Running quick demo...
python demo.py

echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Quick commands to try:
echo   - python data_loader.py          (Download Shakespeare data)
echo   - python train.py --model_type mlp      (Train MLP model)
echo   - python train.py --model_type transformer  (Train Transformer)
echo   - python generate.py --model_path models/best_model.pt --interactive
echo   - python examples.py             (See more examples)
echo.

pause
