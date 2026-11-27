[CmdletBinding()]
param(
    [ValidateSet('Auto', 'CPU')]
    [string]$Device = 'Auto',
    [switch]$ShortcutOnly
)

$ErrorActionPreference = 'Stop'
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
        Write-Host "Downloading $Name"
        Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '2',
            '--connect-timeout', '15', '--speed-limit', '1024', '--speed-time', '45',
            '--output', "$path.download", $url)
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

function Confirm-WritableDirectory {
    param([string]$Path)
    $directory = Get-Item -LiteralPath $Path -Force
    if (!$directory.PSIsContainer) {
        throw "Expected a folder at '$Path'."
    }
    $probe = Join-Path $directory.FullName ('.cellmatchr-write-' + [Guid]::NewGuid().ToString('N'))
    [IO.File]::WriteAllText($probe, '')
    Remove-Item -LiteralPath $probe
}

function Enable-WindowsLongPaths {
    $key = 'SYSTEM\CurrentControlSet\Control\FileSystem'
    $property = 'LongPathsEnabled'
    $registryPath = "HKLM:\$key"
    try {
        $setting = Get-ItemProperty -LiteralPath $registryPath -Name $property -ErrorAction SilentlyContinue
        if ($setting.LongPathsEnabled -eq 1) {
            Write-Host 'Windows long paths are already enabled.'
            return
        }

        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = [Security.Principal.WindowsPrincipal]::new($identity)
        $system = [Environment]::SystemDirectory
        $reg = Join-Path $system 'reg.exe'
        $arguments = @('add', "HKLM\$key", '/v', $property, '/t', 'REG_DWORD', '/d', '1', '/f', '/reg:64')
        if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            Invoke-Checked $reg $arguments
        } else {
            if (![Environment]::UserInteractive) {
                throw 'Administrator approval requires an interactive Windows session.'
            }
            Write-Host 'Approve the Windows administrator prompt to enable long paths.'
            # Elevate only the fixed registry command. Package setup stays in the current account.
            $process = Start-Process -FilePath $reg -ArgumentList $arguments -WorkingDirectory $system `
                -Verb RunAs -Wait -PassThru
            if ($process.ExitCode -ne 0) {
                throw "Windows registry update exited with code $($process.ExitCode)."
            }
        }

        $setting = Get-ItemProperty -LiteralPath $registryPath -Name $property
        if ($setting.LongPathsEnabled -ne 1) {
            throw 'Windows did not retain the long-path setting.'
        }
        Write-Host 'Windows long paths are enabled. If a path-length error persists, restart Windows and rerun setup.'
    } catch {
        Write-Warning "Could not enable Windows long paths. Continuing with package cache '$env:CONDA_PKGS_DIRS'. $($_.Exception.Message)"
    }
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
            Confirm-WritableDirectory $short
            return $short
        } catch {
            Write-Verbose "Could not use package cache '$short'. $($_.Exception.Message)"
        }
    }
    Write-Warning "Could not prepare a short package cache. Using '$normal'. If extraction fails, choose a shorter CELLMATCHR_WINDOWS_RUNTIME path."
    return $normal
}

if ($ShortcutOnly) {
    New-LauncherShortcut
    return
}

$log = Join-Path ([IO.Path]::GetTempPath()) ('CellMatchR-setup-{0:yyyyMMdd-HHmmss}-{1}.log' -f (Get-Date), $PID)
$logging = $false
try {
    Start-Transcript -LiteralPath $log | Out-Null
    $logging = $true
    Write-Host "Setup log: $log"
} catch {
    Write-Warning "Could not open setup log '$log'. $($_.Exception.Message)"
}

$stage = 'Checking Windows and the runtime folder'
try {
    if ($env:OS -ne 'Windows_NT' -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        throw 'CellMatchR Windows setup requires 64-bit Windows on an x64 processor.'
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
    Write-Host "Runtime: $runtime"

    $ownerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if (!(Test-Path -LiteralPath $runtime)) {
        New-PrivateDirectory -Path $runtime -OwnerSid $ownerSid
    }
    Confirm-WritableDirectory $runtime

    $stage = 'Preparing the micromamba package cache'
    $env:CONDA_PKGS_DIRS = Get-PackageCachePath -Runtime $runtime -OwnerSid $ownerSid
    Write-Host "Package cache: $env:CONDA_PKGS_DIRS"
    if (!(Test-Path -LiteralPath $env:CONDA_PKGS_DIRS)) {
        New-Item -ItemType Directory -Path $env:CONDA_PKGS_DIRS | Out-Null
    }
    Confirm-WritableDirectory $env:CONDA_PKGS_DIRS
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

    Enable-WindowsLongPaths

    $stage = 'Installing the micromamba bootstrap'
    if (!(Test-Path -LiteralPath $mamba) -or
        !(Test-Path -LiteralPath (Join-Path (Split-Path -Parent $mamba) 'msvcp140_atomic_wait.dll'))) {
        $mambaArchive = Get-VerifiedArchive 'micromamba-2.10.0-0.tar.bz2' `
            '6283f25f8142606d86e81a43c7103b5d4f1d5c2027e3f9d7326eff23b33930d8'
        Write-Host 'Extracting micromamba.exe'
        Invoke-Checked 'tar.exe' @('-xf', $mambaArchive, '-C', $runtime, 'Library/bin/micromamba.exe')
        $vcArchive = Get-VerifiedArchive 'vc14_runtime-14.51.36247-hb06f545_44.conda' `
            '8221fe78686f3b11586a7585f902f1403c923fb3a5040cff59a16db27782db8d'
        $vcPayloadName = 'pkg-vc14_runtime-14.51.36247-hb06f545_44.tar.zst'
        Write-Host 'Extracting the Visual C++ runtime'
        Invoke-Checked 'tar.exe' @('-xf', $vcArchive, '-C', $runtime, $vcPayloadName)
        Invoke-Checked 'tar.exe' @('-xf', (Join-Path $runtime $vcPayloadName), '-C', $runtime, 'Library/bin')
    }
    $stage = 'Checking the micromamba executable'
    Write-Host $stage
    Invoke-Checked $mamba @('--version')

    $stage = 'Resolving, downloading, and extracting the Python environment'
    Write-Host "$stage. The first setup can take several minutes."
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
    $stage = 'Checking the installed Python environment'
    Write-Host $stage
    Invoke-Checked $mamba @('--no-rc', 'run', '--prefix', $prefix, 'python', '-m', 'pip', 'check')
    Invoke-Checked $mamba @('--no-rc', 'run', '--prefix', $prefix, 'python', '-c',
        "import cv2, micro_sam.automatic_segmentation, tkinter, torch, zarr; from cellpose import models; from qtpy import QtWidgets; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')")
    try {
        New-LauncherShortcut
    } catch {
        Write-Warning "Could not create the Windows shortcut. $($_.Exception.Message)"
    }
    Write-Host 'Setup complete. Open the Launch CellMatchR (Windows) shortcut or run-windows.cmd.'
} catch {
    Write-Host "Setup failed while $($stage.ToLowerInvariant()): $($_.Exception.Message)" -ForegroundColor Red
    if ($logging) {
        Write-Host "Full setup log: $log"
    }
    throw
} finally {
    if ($logging) {
        Stop-Transcript | Out-Null
    }
}
