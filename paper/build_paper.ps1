param()

$ErrorActionPreference = "Stop"

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePdf = Join-Path $paperDir "build\main.pdf"
$namedPdf = Join-Path $paperDir "When_Accuracy_Hides_Path_Dependence_Criterion_Fidelity_in_LLM_Judges_Ali_Uyar.pdf"

Push-Location $paperDir
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

    if (-not (Test-Path -LiteralPath $sourcePdf)) {
        throw "Expected build output not found: $sourcePdf"
    }

    Copy-Item -LiteralPath $sourcePdf -Destination $namedPdf -Force
    Write-Host "Refreshed:" $namedPdf
}
finally {
    Pop-Location
}
