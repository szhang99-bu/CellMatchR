[CmdletBinding()]
param(
    [ValidateSet('Auto', 'CPU')]
    [string]$Device = 'Auto',
    [switch]$ShortcutOnly
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'CellMatchR Windows setup requires 64-bit Windows on an x64 processor.'
}

$repo = Split-Path -Parent $PSScriptRoot

function New-LauncherShortcut {
    $launcher = Join-Path $repo 'run-windows.cmd'
    $icon = Join-Path $PSScriptRoot 'CellMatchR.ico'
    if (!(Test-Path -LiteralPath $launcher -PathType Leaf) -or
        !(Test-Path -LiteralPath $icon -PathType Leaf)) {
        throw 'Keep run-windows.cmd and scripts\CellMatchR.ico in the CellMatchR folder.'
    }
    $path = Join-Path $repo 'Launch CellMatchR (Windows).lnk'
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $launcher
    $shortcut.WorkingDirectory = $repo
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = 'Open CellMatchR on Windows'
    $shortcut.Save()
    Write-Host "Created: $path"
}

if ($ShortcutOnly) {
    New-LauncherShortcut
    return
}

# Keep the default runtime near the root of the drive.
$runtime = Join-Path "$env:SystemDrive\" 'CellMatchR'
if ($env:CELLMATCHR_WINDOWS_RUNTIME) {
    $runtime = [IO.Path]::GetFullPath($env:CELLMATCHR_WINDOWS_RUNTIME)
}
$mamba = Join-Path $runtime 'Library\bin\micromamba.exe'
$prefix = Join-Path $runtime 'CellMatchR-win-env'
$env:MAMBA_ROOT_PREFIX = $runtime
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

function New-PrivateDirectory {
    param([string]$Path, [string]$OwnerSid)
    $directory = New-Item -ItemType Directory -Path $Path
    $acl = Get-Acl -LiteralPath $directory.FullName
    $acl.SetOwner([Security.Principal.SecurityIdentifier]::new($OwnerSid))
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @($OwnerSid, 'S-1-5-18', 'S-1-5-32-544')) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($sid),
            'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $directory.FullName -AclObject $acl
}

function Get-PackageCachePath {
    param([string]$Runtime, [string]$OwnerSid)
    $normal = Join-Path $Runtime 'pkgs'
    # Micromamba's channel and package paths can add over 230 characters.
    $cachePathBudget = 25
    if ($normal.Length -le $cachePathBudget) {
        return $normal
    }
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $identity = $OwnerSid + '|' + $Runtime.TrimEnd('\').ToUpperInvariant()
        $digest = [BitConverter]::ToString($sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity)))
    } finally {
        $sha256.Dispose()
    }
    $suffix = $digest.Replace('-', '').Substring(0, 12).ToLowerInvariant()
    $candidates = @(Join-Path "$env:SystemDrive\" "CMR-$suffix")
    if ($env:ProgramData) {
        $candidates += Join-Path $env:ProgramData ("CM" + $suffix.Substring(0, 8))
    }
    foreach ($short in $candidates) {
        if ($short.Length -gt $cachePathBudget) {
            continue
        }
        try {
            if (!(Test-Path -LiteralPath $short)) {
                New-PrivateDirectory -Path $short -OwnerSid $OwnerSid
            }
            $directory = Get-Item -LiteralPath $short -Force
            $acl = Get-Acl -LiteralPath $short
            if (!$directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $OwnerSid -or
                !$acl.AreAccessRulesProtected) {
                throw 'The short package cache is not a private directory owned by this user.'
            }
            return $short
        } catch {
            Write-Verbose "Could not use package cache '$short'. $($_.Exception.Message)"
        }
    }
    Write-Warning "Could not prepare a short package cache. Using '$normal'. If extraction fails, choose a shorter CELLMATCHR_WINDOWS_RUNTIME path."
    return $normal
}

$ownerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
if (!(Test-Path -LiteralPath $runtime)) {
    New-PrivateDirectory -Path $runtime -OwnerSid $ownerSid
}
$env:CONDA_PKGS_DIRS = Get-PackageCachePath -Runtime $runtime -OwnerSid $ownerSid
Write-Host "Package cache: $env:CONDA_PKGS_DIRS"
try {
    Set-Content -LiteralPath (Join-Path $runtime 'setup-paths.txt') -Encoding UTF8 -Value @(
        "Runtime: $runtime"
        "Environment: $prefix"
        "Package cache: $env:CONDA_PKGS_DIRS"
        'A short package cache helps avoid Windows path-length errors during setup.'
    )
} catch {
    Write-Warning "Could not write setup-paths.txt. $($_.Exception.Message)"
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
    "import cv2, micro_sam.automatic_segmentation, tkinter, torch, zarr; from cellpose import models; from qtpy import QtWidgets; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')")
try {
    New-LauncherShortcut
} catch {
    Write-Warning "Could not create the Windows shortcut. $($_.Exception.Message)"
}
Write-Host 'Setup complete. Open the Launch CellMatchR (Windows) shortcut or run-windows.cmd.'
