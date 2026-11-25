param(
  [Parameter(Mandatory=$true)][string]$Manifest,
  [Parameter(Mandatory=$true)][string]$Output,
  [string]$EnvName = "frequency_calculation",
  [int]$R2Workers = 1
)
$ErrorActionPreference = "Stop"
Write-Host "Using conda environment: $EnvName"
conda run -n $EnvName python final_pipeline_20261126\verify_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed" }
conda run -n $EnvName python final_pipeline_20261126\run_final_batch_full.py --manifest $Manifest --output $Output --resume --r2-workers $R2Workers
if ($LASTEXITCODE -ne 0) { throw "Formal batch failed" }
