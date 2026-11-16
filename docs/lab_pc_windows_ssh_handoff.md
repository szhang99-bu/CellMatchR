# Lab PC Windows 11: 5 Gb Ethernet, SSH, and Remote Desktop handoff

**Prepared:** 16 November 2026. **Status:** Windows 11 is installed; Windows networking, SSH, and Remote Desktop steps below have not yet been performed or verified.

## Goal and known network layout

The lab PC dual boots Ubuntu and Windows 11. Only the operating system currently booted can answer SSH. The Mac already has a direct 5 Gb Ethernet connection on `en9` with `192.168.77.2/24`; its existing `labpc` SSH shortcut connects to Ubuntu at `192.168.77.1` as `shiyuzhang`. Keep that Linux shortcut. Give the Windows 5 Gb adapter the same PC-side address, `192.168.77.1/24`, because Ubuntu and Windows cannot run simultaneously. Use a separate `labpc-win` SSH shortcut and separate host-key identity for Windows.

This direct link needs **no gateway or DNS server**. Keep Wi-Fi or another network connection for internet access. Do not forward SSH port 22 or Remote Desktop port 3389 on a router. Use SSH for terminal work and file transfers, and Remote Desktop for the Windows graphical interface. Switching operating systems still requires a reboot and selection in GRUB or the firmware menu.

## On Windows: identify and configure the 5 Gb adapter

1. Connect the same Ethernet cable used for the Ubuntu direct link. Open **PowerShell as Administrator**. Inspect adapters and addresses:

   ```powershell
   Get-NetAdapter | Format-Table ifIndex,Name,InterfaceDescription,Status,LinkSpeed -Auto
   Get-NetIPAddress -AddressFamily IPv4 | Format-Table InterfaceIndex,InterfaceAlias,IPAddress,PrefixLength -Auto
   ```

2. Identify the **connected 5 Gb** adapter by its name, description and link speed. Do not change Wi-Fi or any other Ethernet adapter. If the driver is absent or link speed is not 5 Gbps, fix the driver/cable first; SSH can still work at a lower negotiated speed.

3. Replace `<5GB_IFINDEX>` with that adapter's `ifIndex`, then inspect it. If it already has `192.168.77.1/24`, skip the `New-NetIPAddress` line. If it has an unexpected manually assigned address, stop and inspect rather than overwrite it.

   ```powershell
   Get-NetIPConfiguration -InterfaceIndex <5GB_IFINDEX>
   New-NetIPAddress -InterfaceIndex <5GB_IFINDEX> -IPAddress 192.168.77.1 -PrefixLength 24
   Get-NetIPConfiguration -InterfaceIndex <5GB_IFINDEX>
   ```

   No `-DefaultGateway` is intentional. `24` means subnet mask `255.255.255.0`. The adapter may show as an unidentified or Public network; that is fine for the narrowly scoped firewall rule below.

## On Windows: enable OpenSSH Server

In **Administrator PowerShell**:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
# Run the next line only if the state above is NotPresent:
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
Get-Service sshd
whoami
```

The service should say `Running`. Record the username after the backslash in `whoami` for the Mac SSH shortcut; the old Windows username may not have survived the reinstall. Windows Hello PIN is not an SSH password. If this is a Microsoft Entra work account, use a local Windows account for SSH instead.

### Allow SSH only over the Mac-to-PC cable

OpenSSH installation may create a broad inbound firewall rule. Create the narrow direct-link rule **first**, then disable the broad rule:

```powershell
New-NetFirewallRule -Name 'LabPC-SSH-Mac-5Gb' -DisplayName 'SSH from Mac over 5 Gb link' `
  -Enabled True -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 `
  -LocalAddress 192.168.77.1 -RemoteAddress 192.168.77.2 -Profile Any

Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue |
  Disable-NetFirewallRule

Get-NetFirewallRule -Name 'LabPC-SSH-Mac-5Gb' | Format-Table Name,Enabled,Direction,Action -Auto
Get-NetTCPConnection -LocalPort 22 -State Listen -ErrorAction SilentlyContinue
```

If the named narrow rule already exists, inspect it instead of running `New-NetFirewallRule` again. Leave Windows Firewall enabled. This rule allows SSH only when the PC address is `192.168.77.1` and the connecting Mac address is `192.168.77.2`.

## Install the Mac's **public** SSH key on Windows

On the **Mac**, display its existing public key:

```sh
cat ~/.ssh/id_ed25519.pub
```

Transfer that single line to Windows by a trusted clipboard, message to yourself, or a USB stick. The line begins `ssh-ed25519`. **Never copy or share `~/.ssh/id_ed25519` without `.pub`**; that is the private key.

Use the branch below that matches the Windows account used for SSH:

**Administrator account** (most Windows first-user accounts): in **Administrator PowerShell**, run `notepad "$env:ProgramData\ssh\administrators_authorized_keys"`. Paste the Mac public key as one complete line, save, then run:

```powershell
icacls.exe "$env:ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
Restart-Service sshd
```

**Standard account:** while signed in as that account, open a normal PowerShell window, run `New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh"`, then `notepad "$env:USERPROFILE\.ssh\authorized_keys"`. Paste the same one-line Mac public key and save. The account's profile and `.ssh` files must not be writable by other ordinary users. Restart `sshd` from an Administrator PowerShell window.

For the administrator branch, Windows OpenSSH uses `C:\ProgramData\ssh\administrators_authorized_keys`, **not** the administrator's personal `.ssh\authorized_keys` file. If key login fails, check this location and its permissions before changing anything else.

Get the Windows server's host-key fingerprint in Administrator PowerShell:

```powershell
ssh-keygen.exe -lf "$env:ProgramData\ssh\ssh_host_ed25519_key.pub"
```

Record the complete `SHA256:...` fingerprint. It will be compared with the Mac's first-connection prompt.

## On the Mac: add a separate Windows SSH shortcut

Keep the existing `Host labpc` entry for Ubuntu. Add this separate block to `~/.ssh/config`, replacing `<WINDOWS_USER>` with the actual Windows username from `whoami`:

```sshconfig
Host labpc-win
    HostName 192.168.77.1
    User <WINDOWS_USER>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    HostKeyAlias labpc-win
```

Then, **while Windows is booted**, run on the Mac:

```sh
ssh labpc-win
```

Compare the displayed `SHA256:...` host-key fingerprint with the Windows value above before accepting. Once connected, run `whoami` and `hostname`. You can also test `scp` or `sftp`. **While Ubuntu is booted**, use the existing `ssh labpc` shortcut. The `HostKeyAlias` keeps the Windows host key separate from Ubuntu's even though both use `192.168.77.1`. A host-key warning is a reason to check which OS is running and its recorded fingerprint, not to disable host-key checks or delete entries blindly.

After key login is confirmed, password authentication can be disabled in `C:\ProgramData\ssh\sshd_config` if desired, followed by `sshd.exe -t` and `Restart-Service sshd`. Do this only after a second key-login test succeeds.

## Windows graphical access: Remote Desktop over the same cable

1. On Windows, open **Settings → System → About** and confirm the edition is **Windows 11 Pro**. Windows Home cannot accept built-in Remote Desktop connections. The old Windows installation had a Pro licence, but confirm the new edition before continuing.
2. Open **Settings → System → Remote Desktop**, turn **Remote Desktop** on, and confirm. Leave **Network Level Authentication** enabled. An administrator account is allowed by default; for a standard account, use **Select users that can remotely access this PC** to add it. The account needs a real password; a Windows Hello PIN alone will not work for this connection.
3. In **Administrator PowerShell**, create rules limited to the Mac's direct-link address, then disable the broader built-in Remote Desktop inbound rules that the Settings toggle may have enabled:

   ```powershell
   New-NetFirewallRule -Name 'LabPC-RDP-Mac-5Gb-TCP' -DisplayName 'RDP from Mac over 5 Gb link (TCP)' `
     -Enabled True -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3389 `
     -LocalAddress 192.168.77.1 -RemoteAddress 192.168.77.2 -Profile Any

   New-NetFirewallRule -Name 'LabPC-RDP-Mac-5Gb-UDP' -DisplayName 'RDP from Mac over 5 Gb link (UDP)' `
     -Enabled True -Direction Inbound -Action Allow -Protocol UDP -LocalPort 3389 `
     -LocalAddress 192.168.77.1 -RemoteAddress 192.168.77.2 -Profile Any

   Get-NetFirewallRule -DisplayGroup 'Remote Desktop' -Direction Inbound |
     Disable-NetFirewallRule

   Get-NetFirewallRule -Name 'LabPC-RDP-Mac-5Gb-*' |
     Format-Table Name,Enabled,Direction,Action -Auto
   Get-Service TermService
   Get-NetTCPConnection -LocalPort 3389 -State Listen -ErrorAction SilentlyContinue
   ```

   If either named narrow rule already exists, inspect it instead of creating it again. Keep Windows Firewall on. Recheck that the broader built-in rules remain disabled after changing Remote Desktop settings or major Windows updates.
4. On the Mac, install/open **Windows App** from the Mac App Store. In its **Devices** tab, choose **+ → Add PC**. Set **PC Name** to `192.168.77.1`, and optionally set **Friendly Name** to `Lab PC Windows`. You do **not** need to sign in to Windows App to add a direct PC connection. Connect using the Windows account name and **account password**, not the PIN. If the username alone is not accepted, use the full `COMPUTERNAME\username` shown by Windows `whoami`.
5. Test while Windows is booted: the Mac should reach the Windows desktop through Windows App, and `ssh labpc-win` should still work. An RDP connection may lock the physical Windows console; that is expected. For an unattended PC, set **Settings → System → Power → Screen, sleep & hibernate timeouts** so the PC does not sleep while plugged in; the display may still turn off. Long GPU training jobs should run independently of the remote desktop window.

If Windows App cannot connect, first verify Windows is booted, the 5 Gb cable is linked, `192.168.77.1` is on the correct adapter, `TermService` is running, and TCP 3389 is listening. Keep the port restricted to this direct link; do not open it on the Wi-Fi or internet connection.

## Verification and handoff notes

Record these after setup for the next session. Do not put passwords or private keys here.

| Item | Result |
| --- | --- |
| Windows `whoami` | _to fill_ |
| 5 Gb adapter name, `ifIndex`, and `LinkSpeed` | _to fill_ |
| Windows 5 Gb address | _expect `192.168.77.1/24`_ |
| `sshd` service state | _expect `Running`, startup `Automatic`_ |
| Windows SSH host-key fingerprint | _to fill_ |
| Mac `ssh labpc-win` result | _to fill_ |
| Windows edition and Remote Desktop toggle | _expect Windows 11 Pro and On_ |
| Mac Windows App connection to `192.168.77.1` | _to fill_ |
| Ubuntu `ssh labpc` still works after reboot | _to fill_ |

The next session should verify both operating systems separately, because only one boots at a time. Do not change GRUB, UEFI boot entries, or EFI files as part of SSH setup. The Windows installation may have reused the Ubuntu drive's EFI partition; its location must be checked independently before any boot-file cleanup.

## Official references

- [Microsoft: install and enable OpenSSH Server on Windows 11](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?pivots=windows-11)
- [Microsoft: Windows OpenSSH key authentication and account-specific key files](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement)
- [Microsoft: Windows OpenSSH server configuration](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration)
- [Microsoft: `New-NetIPAddress`](https://learn.microsoft.com/en-us/powershell/module/nettcpip/new-netipaddress)
- [Microsoft: `New-NetFirewallRule`](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule)
- [Microsoft: Remote Desktop host requirements and setup](https://learn.microsoft.com/en-us/windows-server/remote/remote-desktop-services/remotepc/remote-desktop-allow-access)
- [Microsoft: connect to a remote PC from Windows App on macOS](https://learn.microsoft.com/en-us/windows-app/get-started-connect-devices-desktops-apps)
