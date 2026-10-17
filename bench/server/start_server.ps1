$env:CUDA_VISIBLE_DEVICES = 'GPU-a21cb855-8208-face-ba36-1b52373420a0'
$env:TEMP = 'D:\tmp'
$env:TMP = 'D:\tmp'
Set-Location D:\drama3\bench\server
$bin = 'D:\drama3\third_party\audio.cpp\build\windows-cuda-release\bin\audiocpp_server.exe'
$p = Start-Process -FilePath $bin -ArgumentList '--config','drama3-server.json','--log','--log-file','server.log' -RedirectStandardOutput server.out.txt -RedirectStandardError server.err.txt -WindowStyle Hidden -PassThru
$p.Id | Out-File server.pid -Encoding ascii
