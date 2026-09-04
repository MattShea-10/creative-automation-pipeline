# One-shot local install for Windows (PowerShell 5.1 or 7+).
#
#   .\install.ps1     set up (safe to re-run; it only fills in what's missing)
#   .\run.ps1         start the web app and open it in your browser
#
# If PowerShell refuses to run scripts, allow them for this session first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What it does, in order: finds a Python 3.10+, makes a private virtual
# environment in .venv\, installs requirements.txt into it, creates .env
# from .env.example if you don't have one yet, and checks for the optional
# tesseract binary. Nothing is installed outside this folder.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host " ok  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "warn $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "error $m" -ForegroundColor Red; exit 1 }

# ---- 1. A Python new enough. The `py` launcher is the reliable way to
#         ask for a version on Windows; plain `python` may be the Store
#         stub that opens a shop window.
Say "Looking for Python 3.10 or newer"
$py = $null
foreach ($cand in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "py -3", "python3", "python")) {
  $parts = $cand.Split(" ")
  try {
    & $parts[0] @($parts[1..($parts.Length-1)] + @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)")) 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $parts; break }
  } catch { }
}
if (-not $py) { Die "No Python 3.10+ found. Install one from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and re-run." }
$ver = & $py[0] @($py[1..($py.Length-1)] + @("--version"))
Ok "$ver via '$($py -join ' ')'"

# ---- 2. Virtual environment.
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Say "Creating virtual environment in .venv\"
  & $py[0] @($py[1..($py.Length-1)] + @("-m", "venv", ".venv"))
} else {
  Say "Using the existing .venv\"
}
$vpy = ".venv\Scripts\python.exe"
Ok "$(& $vpy --version) in .venv\"

# ---- 3. Dependencies.
Say "Installing dependencies (this can take a few minutes the first time)"
& $vpy -m pip install --upgrade pip wheel | Out-Null
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Die "pip install failed -- see the messages above." }
Ok "requirements installed"

& $vpy -c "import aggdraw, skimage, psd_tools, PIL, flask"
if ($LASTEXITCODE -ne 0) { Die "A required package didn't import after install (see above). aggdraw needs a wheel for your Python version; if pip tried to build it from source, install the 'Microsoft C++ Build Tools' or use Python 3.12." }
Ok "aggdraw, scikit-image, psd-tools, Pillow and Flask all import"

# ---- 4. Secrets file, never overwritten.
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Ok "created .env from .env.example -- add IDEOGRAM_API_KEY there to use Ideogram (Pollinations works without a key)"
} else {
  Ok ".env already present, left as is"
}

# ---- 5. Optional OCR binary.
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
  Ok "tesseract found"
} else {
  Warn "tesseract not found -- optional. The text-in-image compliance check is skipped without it. Installer: https://github.com/UB-Mannheim/tesseract/wiki"
}

# ---- 6. Folders the app writes to.
foreach ($d in @("outputs", "downloads", "assets\generated_cache")) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-Host ""
Say "Done. Start it with:"
Write-Host ""
Write-Host "    .\run.ps1"
Write-Host ""
Write-Host "Then open http://127.0.0.1:5000 (run.ps1 opens it for you). Ctrl-C stops it."
Write-Host "Command-line pipeline: .venv\Scripts\python.exe -m src.main --brief briefs\sample_campaign.yaml"
