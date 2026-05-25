@echo off
C:\Windows\System32\chcp.com 65001 > nul

set PYTHONUTF8=1
set PYSPARK_PYTHON=C:\Users\nguye\AppData\Local\Programs\Python\Python311\python.exe
set PYSPARK_DRIVER_PYTHON=C:\Users\nguye\AppData\Local\Programs\Python\Python311\python.exe

C:\Users\nguye\AppData\Local\Programs\Python\Python311\Scripts\spark-submit.cmd %*