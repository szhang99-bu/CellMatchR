[CmdletBinding()]
param(
    [ValidateSet('Auto', 'CPU')]
    [string]$Device = 'Auto'
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'CellMatchR Windows setup requires 64-bit Windows on an x64 processor.'
}

$repo = Split-Path -Parent $PSScriptRoot
# Keep the cache path short for packages with deeply nested Windows paths.
$runtime = Join-Path "$env:SystemDrive\" 'CellMatchR'
if ($env:CELLMATCHR_WINDOWS_RUNTIME) {
    $runtime = [IO.Path]::GetFullPath($env:CELLMATCHR_WINDOWS_RUNTIME)
}
$mamba = Join-Path $runtime 'Library\bin\micromamba.exe'
$prefix = Join-Path $runtime 'CellMatchR-win-env'
$env:MAMBA_ROOT_PREFIX = $runtime
$env:CONDA_PKGS_DIRS = Join-Path $runtime 'pkgs'
$env:PIP_CONSTRAINT = Join-Path $repo 'scripts\linux-pip-constraints.txt'
$env:QT_API = 'pyqt5'
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with code $LASTEXITCODE."
    }
}

function Get-VerifiedArchive {
    param([string]$Name, [string]$Sha256)
    $path = Join-Path $runtime $Name
    if (!(Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path).Hash -ne $Sha256) {
        $url = "https://conda.anaconda.org/conda-forge/win-64/$Name"
        Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '2', '--output', "$path.download", $url)
        if ((Get-FileHash -LiteralPath "$path.download").Hash -ne $Sha256) {
            throw "Checksum mismatch for $Name."
        }
        Move-Item -LiteralPath "$path.download" -Destination $path -Force
    }
    return $path
}

if (!(Test-Path -LiteralPath $runtime)) {
    $directory = New-Item -ItemType Directory -Path $runtime
    $acl = Get-Acl -LiteralPath $directory.FullName
    $ownerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @($ownerSid, 'S-1-5-18', 'S-1-5-32-544')) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($sid),
            'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $directory.FullName -AclObject $acl
}

if (!(Test-Path -LiteralPath $mamba) -or
    !(Test-Path -LiteralPath (Join-Path (Split-Path -Parent $mamba) 'msvcp140_atomic_wait.dll'))) {
    $mambaArchive = Get-VerifiedArchive 'micromamba-2.10.0-0.tar.bz2' `
        '6283f25f8142606d86e81a43c7103b5d4f1d5c2027e3f9d7326eff23b33930d8'
    Invoke-Checked 'tar.exe' @('-xf', $mambaArchive, '-C', $runtime, 'Library/bin/micromamba.exe')
    $vcArchive = Get-VerifiedArchive 'vc14_runtime-14.51.36247-hb06f545_44.conda' `
        '8221fe78686f3b11586a7585f902f1403c923fb3a5040cff59a16db27782db8d'
    $vcPayloadName = 'pkg-vc14_runtime-14.51.36247-hb06f545_44.tar.zst'
    Invoke-Checked 'tar.exe' @('-xf', $vcArchive, '-C', $runtime, $vcPayloadName)
    Invoke-Checked 'tar.exe' @('-xf', (Join-Path $runtime $vcPayloadName), '-C', $runtime, 'Library/bin')
}

$operation = 'create'
if (Test-Path -LiteralPath (Join-Path $prefix 'conda-meta\history')) {
    $operation = 'install'
}
$arguments = @('--no-rc', $operation, '--yes', '--prefix', $prefix,
    '--file', (Join-Path $repo 'environment-windows.yml'))
if ($Device -eq 'CPU') {
    $arguments += @('pytorch=2.10.0=*cpu*', 'torchvision=0.25.0=*cpu*', 'onnxruntime=1.22.2=*cpu*')
}

Invoke-Checked $mamba $arguments
Invoke-Checked $mamba @('--no-rc', 'run', '--prefix', $prefix, 'python', '-m', 'pip', 'check')
Invoke-Checked $mamba @('--no-rc', 'run', '--prefix', $prefix, 'python', '-c',
    "import cv2, micro_sam.automatic_segmentation, tkinter, torch; from cellpose import models; from qtpy import QtWidgets; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')")
Write-Host 'Setup complete. Open run-windows.cmd to start CellMatchR.'
