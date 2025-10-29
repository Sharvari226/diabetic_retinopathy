@echo off
cd /d "%~dp0"
echo ======================================================
echo  DIABETIC RETINOPATHY TRAINING AND EVALUATION STARTED
echo ======================================================

echo.
echo [1/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/4] Training model on APTOS 2019 dataset...
python -m src.train config.yaml

echo.
echo [3/4] Evaluating model on Messidor-2 dataset...
python -m src.evaluate --cfg config.yaml --ckpt .\checkpoints\best.ckpt

echo.
echo [4/4] Generating Grad-CAM visualizations (optional)...
python -m src.visualize

echo.
echo ======================================================
echo ✅  ALL STEPS COMPLETED SUCCESSFULLY!
echo Model saved in the 'checkpoints' folder.
echo ======================================================

pause

