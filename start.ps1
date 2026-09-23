param(
    [int]$Port = 8788,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$pythonPath = Join-Path $root $Python
$models = Join-Path $root "models\laya-multilingual-npu\runtime"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}
if (-not (Test-Path (Join-Path $models "onnx\laya_s128_m8_qnn_ctx.onnx"))) {
    throw "Compiled QNN contexts not found. Run: python .\prepare_multilingual_npu.py"
}

& $pythonPath (Join-Path $root "snapdragon_server.py") --models $models --port $Port

