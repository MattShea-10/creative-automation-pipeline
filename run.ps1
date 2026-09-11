# Start the web app from the install made by .\install.ps1, and open it.
#   .\run.ps1                          on http://127.0.0.1:5000
#   $env:PORT = 8080; .\run.ps1        on another port
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "No .venv\ yet -- run .\install.ps1 first." -ForegroundColor Red; exit 1 }
# Keep the compiled stylesheet in step with styles\*.css, so `npm run css`
# is not something anyone has to remember. Hashes rather than dates: the
# Tailwind CLI leaves an output file untouched when a rebuild changes
# nothing, so a date comparison would accuse the source forever. Nothing
# here stops the app starting -- at worst it warns and serves the
# stylesheet that is already built.
$cssStale = $false
if (Test-Path "static\styles.sha256") {
  foreach ($line in Get-Content "static\styles.sha256") {
    if ($line -match '^(\w+)\s+(.+)$') {
      $recorded = $Matches[1]; $src = $Matches[2] -replace '/', '\'
      if ((Test-Path $src) -and ((Get-FileHash -Algorithm SHA256 -Path $src).Hash -ne $recorded.ToUpper())) {
        $cssStale = $true
      }
    }
  }
}
if ($cssStale) {
  Write-Host "==> styles\ changed -- rebuilding the stylesheet" -ForegroundColor Cyan
  if (Get-Command npm -ErrorAction SilentlyContinue) {
    # try/catch as well as the exit code: a native command writing to
    # stderr can itself throw under $ErrorActionPreference = "Stop".
    $rebuilt = $false
    try {
      if (-not (Test-Path "node_modules")) {
        Write-Host "    (first time: installing the build tool with npm install)"
        & npm install --no-audit --no-fund --silent *> $null
      }
      & npm run css --silent *> $null
      $rebuilt = ($LASTEXITCODE -eq 0)
    } catch { $rebuilt = $false }
    if ($rebuilt) {
      Write-Host "ok   stylesheet rebuilt" -ForegroundColor Green
    } else {
      Write-Host "warn the rebuild failed -- run 'npm run css' to see why. Serving the stylesheet that is already built." -ForegroundColor Yellow
    }
  } else {
    Write-Host "warn Node is not installed, so the stylesheet cannot be rebuilt here. Serving the one that is already built." -ForegroundColor Yellow
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
