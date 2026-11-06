$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
Push-Location $repo
try {
    $reportAsset = (Join-Path $repo 'energy_studio/report_deck.mjs') + ';energy_studio'
    $independentAsset = (Join-Path $repo 'energy_studio/independent_deck.mjs') + ';energy_studio'
    & $python -m PyInstaller --noconfirm --onedir --windowed --name GRINS_Energy_Studio --distpath output/app/v0_5 --workpath tmp/energy_studio_build_v0_5 --specpath tmp --add-data $reportAsset --add-data $independentAsset --hidden-import matplotlib.backends.backend_svg --exclude-module scipy --exclude-module statsmodels --exclude-module sympy run_energy_studio.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
    & $python run_energy_studio.py --write-independent-config output/app/v0_5/GRINS_Energy_Studio/default_config.json
    & $python run_energy_studio.py --write-benchmark-profiles output/app/v0_5/GRINS_Energy_Studio/example_independent_profiles.json
    Copy-Item -LiteralPath energy_studio/README.md -Destination output/app/v0_5/GRINS_Energy_Studio/README.md
    Copy-Item -LiteralPath energy_studio/HYBRID_METHOD.md -Destination output/app/v0_5/GRINS_Energy_Studio/HYBRID_METHOD.md
    Copy-Item -LiteralPath energy_studio/INDEPENDENT_BENCHMARK.md -Destination output/app/v0_5/GRINS_Energy_Studio/INDEPENDENT_BENCHMARK.md
    Write-Output 'Built output/app/v0_5/GRINS_Energy_Studio/GRINS_Energy_Studio.exe'
} finally { Pop-Location }
