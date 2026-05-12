@echo off

set PYTHON=C:\Users\ASIET-MEITY\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT=F:\test-bed\forcast-demand.py
set LOG=F:\test-bed\pipeline_log.txt

echo Running forcast-demand.py...

if not exist "%PYTHON%" (
    echo ERROR: Python not found >> "%LOG%"
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo ERROR: forcast-demand.py not found >> "%LOG%"
    exit /b 1
)

"%PYTHON%" -u "%SCRIPT%" >> "%LOG%" 2>&1

echo Script completed at %date% %time% >> "%LOG%"
exit /b 0
