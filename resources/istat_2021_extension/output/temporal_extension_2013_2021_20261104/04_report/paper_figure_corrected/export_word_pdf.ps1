$ErrorActionPreference = 'Stop'
$taskDocx = Join-Path $PSScriptRoot 'Household_archetypes_2013_2021_corrected_report.docx'
$taskPdf = Join-Path $PSScriptRoot 'Household_archetypes_2013_2021_corrected_report.pdf'
$taskWord = $null
$taskDocument = $null
try {
  $taskWord = New-Object -ComObject Word.Application
  $taskWord.Visible = $false
  $taskWord.DisplayAlerts = 0
  $taskWord.AutomationSecurity = 3
  $taskDocument = $taskWord.Documents.Open($taskDocx, $false, $true, $false)
  $taskDocument.Repaginate()
  $taskDocument.Fields.Update() | Out-Null
  $taskPages = $taskDocument.ComputeStatistics(2)
  $taskDocument.ExportAsFixedFormat($taskPdf,17,$false,0,0,1,1,0,$true,$true,1,$true,$true,$false)
  @{ renderer='Microsoft Word native PDF export'; pages=$taskPages; pdf=$taskPdf } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'word_export.json') -Encoding utf8
  Write-Output "Exported $taskPages pages"
}
finally {
  if ($null -ne $taskDocument) { $taskDocument.Close(0); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskDocument) }
  if ($null -ne $taskWord) { $taskWord.Quit(0); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskWord) }
}


