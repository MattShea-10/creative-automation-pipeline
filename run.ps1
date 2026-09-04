# Start the web app from the install made by .\install.ps1, and open it.
#   .\run.ps1                          on http://127.0.0.1:5000
#   $env:PORT = 8080; .\run.ps1        on another port
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "No .venv\ yet -- run .\install.ps1 first." -ForegroundColor Red; exit 1 }
$port = if ($env:PORT) { $env:PORT } else { "5000" }
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
