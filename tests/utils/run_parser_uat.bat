@echo off
REM Batch file to run Binance Symbol Parser UAT tests
REM This will attempt to find and run Python

echo ========================================
echo Binance Symbol Parser UAT Test Runner
echo ========================================
echo.

REM Try different Python commands
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: python
    python test_binance_symbol_parser_uat.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: python3
    python3 test_binance_symbol_parser_uat.py
    goto :end
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found: py
    py test_binance_symbol_parser_uat.py
    goto :end
)

echo ERROR: Python not found in PATH
echo.
echo Please install Python or add it to your PATH, then run:
echo   python test_binance_symbol_parser_uat.py
echo.
pause

:end
echo.
pause
