@echo off
setlocal
cd /d "%~dp0"
echo Membuka File Explorer untuk memilih file Markdown...
python experiments\md_to_pdf.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pastikan anda sudah menginstall library yang dibutuhkan:
    echo pip install markdown xhtml2pdf
    echo.
    pause
)
endlocal
