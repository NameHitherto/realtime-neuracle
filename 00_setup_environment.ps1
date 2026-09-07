param(
    [string]$PythonPath = "",
    [switch]$SkipGui,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements_realtime.lock.txt"
$guiRoot = Join-Path $projectRoot "GUI"

function Test-Python310([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate) -or
        -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    try {
        $version = & $Candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return $LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.10"
    }
    catch {
        return $false
    }
}

function Find-Python310 {
    $candidates = [System.Collections.Generic.List[string]]::new()

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $launcherPython = & $launcher.Source -3.10 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $launcherPython) {
                $candidates.Add($launcherPython.Trim())
            }
        }
        catch {}
    }

    $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"))
    $candidates.Add("C:\Program Files\Python310\python.exe")

    Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Source) { $candidates.Add($_.Source) }
    }

    $condaExe = $null
    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
        $condaExe = $env:CONDA_EXE
    }
    else {
        $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
        if ($condaCommand) { $condaExe = $condaCommand.Source }
    }
    if ($condaExe) {
        try {
            $condaInfo = (& $condaExe info --json 2>$null) | ConvertFrom-Json
            foreach ($condaEnvironment in $condaInfo.envs) {
                $candidates.Add((Join-Path $condaEnvironment "python.exe"))
            }
        }
        catch {}
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Python310 $candidate) { return $candidate }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
    throw "Dependency lock file was not found: $lockFile"
}
if ((Test-Path -LiteralPath $venvPython -PathType Leaf) -and
    -not (Test-Python310 $venvPython)) {
    throw "The existing .venv is not a valid Python 3.10 environment. Rename or remove .venv, then run this script again."
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (-not $PythonPath) { $PythonPath = Find-Python310 }
    if (-not (Test-Python310 $PythonPath)) {
        throw "Python 3.10 was not found. Install Python 3.10 or pass -PythonPath with its python.exe path."
    }
    Write-Host "Creating project virtual environment: $venvRoot"
    Write-Host "Using Python: $PythonPath"
    & $PythonPath -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Installing locked realtime dependencies..."
& $venvPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $venvPython -m pip install --disable-pip-version-check -r $lockFile
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed." }

if (-not $SkipGui) {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) {
        throw "npm was not found. Install Node.js 20 or newer, then run this script again."
    }
    $cargoCommand = Get-Command cargo.exe -ErrorAction SilentlyContinue
    if (-not $cargoCommand) {
        throw "Cargo was not found. Install the Rust MSVC toolchain, then run this script again."
    }
    Write-Host "Installing locked GUI dependencies..."
    Push-Location -LiteralPath $guiRoot
    try {
        & $npmCommand.Source ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
        if (-not $SkipChecks) {
            & $npmCommand.Source run build
            if ($LASTEXITCODE -ne 0) { throw "GUI frontend build failed." }
        }
    }
    finally {
        Pop-Location
    }
    if (-not $SkipChecks) {
        & $cargoCommand.Source check --manifest-path (Join-Path $guiRoot "src-tauri\Cargo.toml")
        if ($LASTEXITCODE -ne 0) { throw "Tauri Rust check failed." }
    }
}

if (-not $SkipChecks) {
    & $venvPython (Join-Path $projectRoot "01_check_environment.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Environment setup completed successfully."
exit 0
