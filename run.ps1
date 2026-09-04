# Start the web app from the install made by .\install.ps1, and open it.
#   .\run.ps1                          on http://127.0.0.1:5000
#   $env:PORT = 8080; .\run.ps1        on another port
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "No .venv\ yet -- run .\install.ps1 first." -ForegroundColor Red; exit 1 }
$port = if ($env:PORT) { $env:PORT } else { "5000" }
# Move off a busy port rather than opening the browser on something
# that isn't this app.
function PortBusy($p) {
  try { $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, [int]$p); $l.Start(); $l.Stop(); return $false } catch { return $true }
}
foreach ($cand in @($port, "5050", "8080", "8000", "8765")) {
  if (-not (PortBusy $cand)) {
    if ($cand -ne $port) { Write-Host "Port $port is in use -- using $cand instead." -ForegroundColor Yellow }
    $port = $cand; break
  }
}
$url = "http://127.0.0.1:$port"
# Open the browser once the server answers, without blocking the server.
Start-Job -ScriptBlock {
  param($u)
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try { Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { }
  }
} -ArgumentList $url | Out-Null
Write-Host "Creative Automation Pipeline -> $url   (Ctrl-C to stop)"
$env:PORT = $port
& ".venv\Scripts\python.exe" webapp.py
