# Build the Windows package of the web app: a folder with
# CreativeAutomationPipeline.exe, everything it needs, and the user's
# editable files beside it -- zipped as CreativeAutomationPipeline-windows.zip.
#
#   .\install.ps1                 (once, if you haven't -- makes .venv\)
#   .\windows\build_exe.ps1
#
# Output: dist\CreativeAutomationPipeline\ and dist\CreativeAutomationPipeline-windows.zip.
# The same thing runs on GitHub's Windows runner -- see
# .github\workflows\windows-exe.yml -- so nobody needs a Windows PC to
# make one.
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $py -m pip install --quiet pyinstaller
if ($LASTEXITCODE -ne 0) { Write-Host "pip install pyinstaller failed" -ForegroundColor Red; exit 1 }

& $py -m PyInstaller --noconfirm --clean windows\CreativeAutomationPipeline.spec
if ($LASTEXITCODE -ne 0) { Write-Host "PyInstaller failed -- see above" -ForegroundColor Red; exit 1 }

$out = "dist\CreativeAutomationPipeline"
# The files a user is meant to see and edit sit beside the exe, not
# inside _internal\: the templates the app applies per size (and writes
# back to), brand assets, sample briefs, and the key file.
foreach ($d in @("default_templates", "assets\brand", "briefs")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $out $d) | Out-Null
  Copy-Item -Recurse -Force (Join-Path $d "*") (Join-Path $out $d)
}
Copy-Item -Force ".env.example" (Join-Path $out ".env.example")
Copy-Item -Force "windows\READ ME FIRST.txt" (Join-Path $out "READ ME FIRST.txt")
foreach ($d in @("outputs", "downloads", "assets\generated_cache")) { New-Item -ItemType Directory -Force -Path (Join-Path $out $d) | Out-Null }

$zip = "dist\CreativeAutomationPipeline-windows.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $out -DestinationPath $zip
Write-Host ""
Write-Host "Built $zip" -ForegroundColor Green
Write-Host "Unzip it anywhere, put the Ideogram key in .env, double-click CreativeAutomationPipeline.exe."
