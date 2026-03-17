# Kaggle MCP Installation Script for Windows
# Run with administrator privileges

# Check if running with administrator privileges
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs to be run as Administrator. Please restart it with admin privileges."
    Start-Sleep -Seconds 5
    exit
}

Write-Host "Kaggle MCP Installation"
Write-Host "----------------------"
Write-Host "This script will install the Kaggle MCP server for Claude."
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version
    if (-not $pythonVersion) {
        throw "Python not found"
    }
} catch {
    Write-Host "Error: Python is required but not found or not in PATH."
    Write-Host "Please install Python and ensure it's in your PATH, then try again."
    Start-Sleep -Seconds 5
    exit
}

# Check pip installation
try {
    $pipVersion = pip --version
    if (-not $pipVersion) {
        throw "pip not found"
    }
} catch {
    Write-Host "Error: pip is required but not found or not in PATH."
    Write-Host "Please install pip and try again."
    Start-Sleep -Seconds 5
    exit
}

# Install Kaggle MCP
Write-Host "Installing Kaggle MCP..."
python -m pip install --upgrade "mcp[cli]" kaggle
python -m pip install --upgrade git+https://github.com/54yyyu/kaggle-mcp.git

# Run setup
Write-Host ""
Write-Host "Running setup..."
python -m kaggle_mcp.setup

Write-Host ""
Write-Host "Installation complete!"
Write-Host "You can now use Kaggle functions in Claude."
Write-Host "Remember to restart Claude Desktop for changes to take effect."
Start-Sleep -Seconds 5
