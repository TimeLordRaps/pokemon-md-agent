# start_mgba.ps1 - Launch mGBA and wait for Lua server connection
# Idempotent: checks if running, launches if not, waits for TCP:8888

param()

# Get process name from MGBAX path (remove extension for Get-Process)
$mgbaProcessName = [System.IO.Path]::GetFileNameWithoutExtension($env:MGBAX)

# Check if mGBA is already running
$mgbaProcess = Get-Process -Name $mgbaProcessName -ErrorAction SilentlyContinue
if (-not $mgbaProcess) {
    Write-Host "mGBA not running, launching $env:MGBAX with $env:PMD_ROM"
    Start-Process -FilePath $env:MGBAX -ArgumentList "`"$env:PMD_ROM`""
    Start-Sleep -Seconds 2  # Brief wait for process to start
}

# Wait up to 20 seconds for TCP port 8888 to be listening
$timeout = 20
$startTime = Get-Date
$connected = $false

do {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("localhost", 8888)
        $connected = $true
        $tcpClient.Close()
        Write-Host "mGBA Lua server detected on port 8888"
    } catch {
        Start-Sleep -Seconds 1
    }
} while (-not $connected -and ((Get-Date) - $startTime).TotalSeconds -lt $timeout)

if (-not $connected) {
    Write-Host "ERROR: mGBA Lua server not detected on port 8888 after $timeout seconds"
    Write-Host "Please manually open Tools → Scripting in mGBA and load: $env:MGBALUA"
    exit 1
}

# Export environment variables for downstream scripts
$env:PMD_ROM = $env:PMD_ROM
$env:PMD_SAVE = $env:PMD_SAVE
$env:MGBALUA = $env:MGBALUA
$env:MGBAX = $env:MGBAX

Write-Host "mGBA setup complete - ready for agent operations"