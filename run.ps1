# Start the web app from the install made by .\install.ps1, and open it.
#   .\run.ps1                          on http://127.0.0.1:5000
#   $env:PORT = 8080; .\run.ps1        on another port
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "No .venv\ yet -- run .\install.ps1 first." -ForegroundColor Red; exit 1 }
# The stylesheet is compiled ahead of time (styles\*.css -> static\*.css,
# `npm run css`). Editing the source and forgetting to rebuild is silent
# otherwise: the app keeps serving the old CSS. Compared by hash, not by
# date -- the Tailwind CLI leaves the output file alone when a rebuild
# changes nothing, so its timestamp would accuse the source forever.
if (Test-Path "static\styles.sha256") {
  foreach ($line in Get-Content "static\styles.sha256") {
    if ($line -match '^(\w+)\s+(.+)$') {
      $recorded = $Matches[1]; $src = $Matches[2] -replace '/', '\'
      if (Test-Path $src) {
        $now = (Get-FileHash -Algorithm SHA256 -Path $src).Hash
        if ($now -ne $recorded.ToUpper()) {
          Write-Host "warn  $src changed since the stylesheet was built -- run 'npm run css' to rebuild it." -ForegroundColor Yellow
        }
      }
    }
  }
}
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
