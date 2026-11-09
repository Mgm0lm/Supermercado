param(
    [Parameter(Mandatory)] [string] $MediaPipeDll,
    [Parameter(Mandatory)] [string] $MediaPipeNotices,
    [Parameter(Mandatory)] [string] $OnnxRuntimeDir,
    [Parameter(Mandatory)] [string] $VcRuntimeDir,
    [Parameter(Mandatory)] [string] $OcrWheel,
    [Parameter(Mandatory)] [string] $OcrLicense,
    [string] $Output = '.release-work'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'target/release/caixa-visao-computacional.exe'
$outputDir = (New-Item -Force -ItemType Directory -Path $Output).FullName
$stage = Join-Path $outputDir 'CaixaVisaoComputacional-Rust'
$models = Join-Path $stage 'modelo'
$ocrModels = Join-Path $models 'ocr'
$notices = Join-Path $stage 'avisos'
New-Item -Force -ItemType Directory -Path $ocrModels,$notices | Out-Null

Copy-Item -LiteralPath $exe -Destination (Join-Path $stage 'CaixaVisaoComputacional.exe')
Copy-Item -LiteralPath $MediaPipeDll -Destination (Join-Path $stage 'libmediapipe.dll')
Copy-Item -LiteralPath (Join-Path $OnnxRuntimeDir 'capi/onnxruntime.dll') -Destination $stage
foreach ($runtime in 'msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll') {
    Copy-Item -LiteralPath (Join-Path $VcRuntimeDir $runtime) -Destination $stage
}
foreach ($file in 'best.onnx','hand_landmarker.task') {
    Copy-Item -LiteralPath (Join-Path $root "Programa/modelo/$file") -Destination $models
}
Copy-Item -LiteralPath (Join-Path $root 'rust/gesture_mlp.json') -Destination $models
foreach ($file in 'ocr_keywords.json','config_override.example.json') {
    Copy-Item -LiteralPath (Join-Path $root "Programa/$file") -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $MediaPipeNotices 'LICENSE') -Destination (Join-Path $notices 'MediaPipe-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $MediaPipeNotices 'NOTICE') -Destination (Join-Path $notices 'MediaPipe-NOTICE.txt')
Copy-Item -LiteralPath (Join-Path $OnnxRuntimeDir 'LICENSE') -Destination (Join-Path $notices 'ONNXRuntime-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $OnnxRuntimeDir 'ThirdPartyNotices.txt') -Destination (Join-Path $notices 'ONNXRuntime-ThirdPartyNotices.txt')
Copy-Item -LiteralPath $OcrLicense -Destination (Join-Path $notices 'RapidOCR-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'rust/terceiros.txt') -Destination $notices

$wheelPath = (Resolve-Path -LiteralPath $OcrWheel).Path
$expectedWheel = '971D7D5F223A7A808662229DF1EF69893809D8457D834E6373D3854BC1782CBF'
if ((Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash -ne $expectedWheel) {
    throw 'Use o wheel rapidocr-onnxruntime 1.4.4 indicado na documentação.'
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$wheel = [IO.Compression.ZipFile]::OpenRead($wheelPath)
try {
    foreach ($kind in 'det','rec') {
        $entry = $wheel.GetEntry("rapidocr_onnxruntime/models/ch_PP-OCRv4_${kind}_infer.onnx")
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $ocrModels "$kind.onnx"), $true)
    }
} finally {
    $wheel.Dispose()
}

$assetName = 'CaixaVisaoComputacional-Windows-Rust-x64.zip'
$asset = Join-Path $outputDir $assetName
$files = @(
    'CaixaVisaoComputacional.exe', 'libmediapipe.dll', 'onnxruntime.dll',
    'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
    'modelo/best.onnx', 'modelo/hand_landmarker.task', 'modelo/gesture_mlp.json',
    'modelo/ocr/det.onnx', 'modelo/ocr/rec.onnx',
    'ocr_keywords.json', 'config_override.example.json',
    'avisos/MediaPipe-LICENSE.txt', 'avisos/MediaPipe-NOTICE.txt',
    'avisos/ONNXRuntime-LICENSE.txt', 'avisos/ONNXRuntime-ThirdPartyNotices.txt',
    'avisos/RapidOCR-LICENSE.txt', 'avisos/terceiros.txt'
)
& tar -a -cf $asset -C $stage @files
if ($LASTEXITCODE -ne 0) { throw "Falha ao criar $asset" }
$sha = (Get-FileHash -LiteralPath $asset -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$asset.sha256" -Value "$sha  $assetName" -Encoding ascii
Get-Item -LiteralPath $asset
