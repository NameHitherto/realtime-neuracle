$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$guiRoot = Join-Path $projectRoot "GUI"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Project virtual environment is missing. Run .\00_setup_environment.ps1 first."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js before starting the GUI."
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo was not found. Install the Rust toolchain before starting the Tauri GUI."
}

Push-Location -LiteralPath $guiRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $guiRoot "node_modules"))) {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
    }
    npm run tauri dev
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
