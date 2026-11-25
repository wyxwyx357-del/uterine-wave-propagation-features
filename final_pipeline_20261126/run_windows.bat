@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_windows.bat D:\study\final_manifest.csv D:\study\final_outputs [conda_env]
  exit /b 2
)
if "%~2"=="" (
  echo Usage: run_windows.bat D:\study\final_manifest.csv D:\study\final_outputs [conda_env]
  exit /b 2
)
set ENVNAME=%~3
if "%ENVNAME%"=="" set ENVNAME=frequency_calculation
conda run -n %ENVNAME% python final_pipeline_20261126\verify_environment.py || exit /b %errorlevel%
conda run -n %ENVNAME% python final_pipeline_20261126\run_final_batch_full.py --manifest "%~1" --output "%~2" --resume --r2-workers 1
