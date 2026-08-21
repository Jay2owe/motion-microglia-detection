param(
    [string]$Run = "r01_base",
    [string]$Stem = "95_A3",
    [string]$ReviewOnlyFrom = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
if ($ReviewOnlyFrom) {
    python "$projectRoot\code\pipeline.py" --run $Run --stem $Stem --review-only-from $ReviewOnlyFrom
} else {
    python "$projectRoot\code\pipeline.py" --run $Run --stem $Stem
}
if ($LASTEXITCODE -ne 0) {
    throw "Motion pipeline failed with exit code $LASTEXITCODE"
}
