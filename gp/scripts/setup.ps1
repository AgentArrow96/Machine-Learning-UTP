param([switch]$SkipTorch)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TMP = Join-Path $projectRoot '.tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed.' }
}
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $pythonExe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Environment tooling installation failed.' }
if (-not $SkipTorch) {
    & $pythonExe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
}
& $pythonExe -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed.' }
$locked = & $pythonExe -m pip freeze --exclude-editable
@('--extra-index-url https://download.pytorch.org/whl/cu128') + $locked + @('-e .') | Set-Content -Encoding utf8 -LiteralPath requirements.lock.txt
& $pythonExe -m floodnet doctor --gpu-smoke
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed. U-Net will not fall back to CPU.' }
