# BGE嵌入服务 - PowerShell启动脚本
# 使用方法: .\start_service.ps1 [选项]
# 选项:
#   -test          启动后自动运行测试
#   -cache         启动后测试缓存API
#   -port <端口>   指定服务端口（默认7860）
#   -host <主机>   指定服务主机（默认0.0.0.0）
#   -help          显示帮助信息

# Change to project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $ScriptDir)

param(
    [switch]$test,
    [switch]$cache,
    [string]$port = "7860",
    [string]$host = "0.0.0.0",
    [switch]$help
)

if ($help) {
    Write-Host "BGE嵌入服务 - PowerShell启动脚本" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使用方法: .\start_service.ps1 [选项]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "选项:" -ForegroundColor Green
    Write-Host "  -test          启动后自动运行测试"
    Write-Host "  -cache         启动后测试缓存API"
    Write-Host "  -port <端口>   指定服务端口（默认7860）"
    Write-Host "  -host <主机>   指定服务主机（默认0.0.0.0）"
    Write-Host "  -help          显示帮助信息"
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Yellow
    Write-Host "  .\start_service.ps1              # 默认启动"
    Write-Host "  .\start_service.ps1 -test        # 启动并测试"
    Write-Host "  .\start_service.ps1 -cache       # 启动并测试缓存"
    Write-Host "  .\start_service.ps1 -port 8080   # 指定端口启动"
    exit
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "BGE嵌入服务 - PowerShell启动脚本" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 设置环境变量
$env:BGE_PORT = $port
$env:BGE_HOST = $host

# 检查虚拟环境
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[错误] 虚拟环境不存在，请先创建虚拟环境" -ForegroundColor Red
    exit 1
}

# 激活虚拟环境
Write-Host "[1/3] 激活虚拟环境..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# 检查服务是否运行
Write-Host "[2/3] 检查服务状态..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -ErrorAction Stop
    Write-Host "[信息] 服务已在运行中" -ForegroundColor Green
    $serviceRunning = $true
} catch {
    Write-Host "[信息] 服务未运行，正在启动..." -ForegroundColor Yellow
    Write-Host "[3/3] 启动BGE嵌入服务..." -ForegroundColor Yellow
    Write-Host "[配置] 主机: $host, 端口: $port" -ForegroundColor Cyan
    
    # 启动服务
    $process = Start-Process -FilePath "python" -ArgumentList "cherry_bge_service.py" -PassThru -WindowStyle Normal
    
    # 等待服务启动
    Write-Host "[信息] 等待服务启动..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # 再次检查服务状态
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -ErrorAction Stop
        Write-Host "[成功] 服务启动成功" -ForegroundColor Green
        $serviceRunning = $true
    } catch {
        Write-Host "[错误] 服务启动失败" -ForegroundColor Red
        exit 1
    }
}

# 运行测试
if ($test) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "运行测试" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    python -m pytest tests/test_api.py -v
}

# 测试缓存API
if ($cache) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "测试缓存管理API" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "[1/4] 获取缓存统计信息..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://127.0.0.1:$port/cache/stats" | ConvertTo-Json -Depth 10
    Write-Host ""
    
    Write-Host "[2/4] 测试缓存效果 - 第一次请求..." -ForegroundColor Yellow
    $body = @{
        texts = @("测试缓存效果的文本")
        normalize = $true
    } | ConvertTo-Json -Depth 10
    Invoke-RestMethod -Uri "http://127.0.0.1:$port/embed" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 10
    Write-Host ""
    
    Write-Host "[3/4] 测试缓存效果 - 第二次请求..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://127.0.0.1:$port/embed" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 10
    Write-Host ""
    
    Write-Host "[4/4] 再次获取缓存统计信息..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://127.0.0.1:$port/cache/stats" | ConvertTo-Json -Depth 10
    Write-Host ""
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "完成" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "服务信息:" -ForegroundColor Green
Write-Host "  主机: $host" -ForegroundColor White
Write-Host "  端口: $port" -ForegroundColor White
Write-Host ""
Write-Host "API端点:" -ForegroundColor Green
Write-Host "  GET    /health              - 健康检查" -ForegroundColor White
Write-Host "  GET    /model-info          - 模型信息" -ForegroundColor White
Write-Host "  POST   /embed               - 主要嵌入端点" -ForegroundColor White
Write-Host "  POST   /embed_legacy        - 旧版API" -ForegroundColor White
Write-Host "  POST   /v1/embeddings       - OpenAI兼容API" -ForegroundColor White
Write-Host "  POST   /embeddings          - LangChain兼容API" -ForegroundColor White
Write-Host ""
Write-Host "缓存管理API:" -ForegroundColor Green
Write-Host "  GET    /cache/stats         - 获取缓存统计" -ForegroundColor White
Write-Host "  POST   /cache/clear         - 清空缓存" -ForegroundColor White
Write-Host "  POST   /cache/save          - 保存缓存" -ForegroundColor White
Write-Host "  POST   /cache/remove-expired - 移除过期缓存" -ForegroundColor White
Write-Host ""