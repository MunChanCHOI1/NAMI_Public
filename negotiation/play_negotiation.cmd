@echo off
chcp 65001 >nul
rem Double-click to open the negotiation playground (CPU keyword reference, not the Zoro model).
rem Extra options are passed through, e.g.  play_negotiation.cmd --backend zoro-http --url http://127.0.0.1:30001
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo 협상 체험을 시작합니다. 브라우저가 열리지 않으면 http://127.0.0.1:8766/play 를 여세요. 끝내려면 이 창에서 Ctrl+C.
"%PY%" "%~dp0run_play.py" %*
endlocal
