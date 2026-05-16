Goal-net XPBD: Windows bat helpers
==================================

All scripts use `C:\Python312\python.exe` by default. Override with:

    set PY=C:\path\to\python.exe

before calling, or edit `_env.bat`.

Each script does `cd /d <repo_root>` internally, so you can launch them
from anywhere (double-click in Explorer, called from another folder, etc.).

Scripts
-------

_env.bat            Shared: locates Python interpreter and repo root.
                    Not meant to be called directly.

topology.bat        Print topology summary.
                        topology.bat
                        topology.bat my_params.json

generate.bat        Generate dataset on CUDA, writing raw frames.
                        generate.bat                     (default tmp_rerun_10, 10 samples)
                        generate.bat tmp_rerun_50 50
                        generate.bat my_run\01 100 42    (out, count, seed)

generate_cpu.bat    Same as generate.bat but on CPU.

view_spawn.bat      [Recommended on Windows] open the dataset in a NATIVE
                    rerun viewer window. Streams from raw\sample_*.json,
                    so it never touches a stale dataset.rrd.
                        view_spawn.bat
                        view_spawn.bat tmp_rerun_50\raw

view_web.bat        Serve the rerun WEB viewer (browser). Use only if you
                    really need to share over network / run headless.
                        view_web.bat
                        view_web.bat tmp_rerun_50\raw 19090
                    The console will print the exact URL to open
                    (with `?url=` properly URL-encoded and
                    `&hide_welcome_screen` appended).

view_save_rrd.bat   Persist the dataset to a single .rrd file you can
                    later open with the standalone rerun viewer.
                        view_save_rrd.bat
                        view_save_rrd.bat tmp_rerun_50\raw shared\run50.rrd

kill_rerun.bat      Kill stale processes holding rerun ports
                    (9090/9091/9876/19090/19091) + any leftover rerun.exe.

fresh_run.bat       One-shot: kill_rerun -> generate -> view_spawn.
                        fresh_run.bat
                        fresh_run.bat tmp_rerun_50 50 7

Typical workflow
----------------

First time / clean run:
    bat\fresh_run.bat

Re-open existing dataset:
    bat\view_spawn.bat tmp_rerun_10\raw

Bigger run:
    bat\generate.bat my_runs\big 200 1
    bat\view_spawn.bat my_runs\big\raw
