$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path $here
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$ops = Join-Path $here "joyomni_ops"
$cutlass = Join-Path $here "tmp\cutlass"
$cuda = if ($env:CUDA_HOME) { $env:CUDA_HOME } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" }
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uv = if ($uvCommand) { $uvCommand.Source } else { Join-Path $env:USERPROFILE ".local\bin\uv.exe" }

cmd.exe /c "`"$vcvars`" && set DISTUTILS_USE_SDK=1&& set CUDA_HOME=$cuda&& set JOYOMNI_OPS_CUDA_ARCHS=120a&& set JOYOMNI_OPS_CUTLASS_DIR=$cutlass&& `"$uv`" pip install --python `"$py`" --no-build-isolation `"$ops`""
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -c "import joyomni_ops; print('joyomni_ops ok')"
