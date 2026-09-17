"""Docker supervisor: paper failure must not take down the training dashboard."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .diagnostics import public_diagnostic


def write_status(run, **values):
    path = run / "service.json"
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps({"updated_at": time.time(), **values}), encoding="utf-8")
    temporary.replace(path)


def worker_diagnostic(run, started_at):
    path = run / "error.json"
    try:
        if path.stat().st_mtime >= started_at:
            return public_diagnostic(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return public_diagnostic({})


def supervise(dashboard, worker, on_worker_exit, sleep=time.sleep):
    """Never restart a halted worker; stay alive as long as the UI is alive."""
    reported = False
    while True:
        code = dashboard.poll()
        if code is not None:
            return code or 1
        if not reported:
            code = worker.poll()
            if code is not None:
                on_worker_exit(code)
                reported = True
        sleep(0.25)


def stop_process(process):
    # Only groups created by this supervisor; trainers inherit the dashboard group.
    # A reaped child's PID can be reused during long-running dashboard sessions.
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def main():
    root = Path(os.environ.get("STONKFLY_ROOT", Path.cwd())).resolve()
    defaults = {"STONKFLY_DATA": str(root / "data"), "STONKFLY_RUN": str(root / "runs/paper"),
                "STONKFLY_VIZ_CACHE": str(root / "dashboard/cache"), "HOST": "0.0.0.0", "PORT": "8765"}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    run = Path(os.environ["STONKFLY_RUN"]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    processes = []

    def launch(args):
        process = subprocess.Popen([sys.executable, *args], cwd=root, start_new_session=True)
        processes.append(process)
        return process

    def shutdown(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print("[stonkfly] Iniciando dashboard y supervisor paper...", flush=True)
    try:
        write_status(run, state="preparing", message="Preparando/verificando el dataset; la UI está disponible.")
        dashboard = launch([str(root / "dashboard/server.py"), str(run)])
        data = Path(os.environ["STONKFLY_DATA"])
        prepared = all((data / name).is_file() for name in
                       ("graph.npz", "annotations.feather", "normalized/neurons.feather"))
        preparation = launch(["-m", "stonkfly", "verify" if prepared else "prepare"])
        while preparation.poll() is None:
            if dashboard.poll() is not None:
                return dashboard.returncode or 1
            time.sleep(0.25)
        if preparation.returncode:
            write_status(run, state="preparation_failed", exit_code=preparation.returncode,
                         message="Falló la preparación/verificación del dataset. Consulta el registro del contenedor.")
            print("[stonkfly] Dataset no disponible; la UI permanece activa. Paper no se inicia.", flush=True)
            return dashboard.wait()
        started = time.time()
        worker = launch(["-m", "stonkfly", "run", "--out", str(run)])
        write_status(run, state="paper_running", message="Proceso paper iniciado.")

        def stopped(code):
            diagnostic = worker_diagnostic(run, started)
            write_status(run, state="paper_stopped", exit_code=code, **diagnostic)
            print(f"[stonkfly] Paper terminó con código {code}. {diagnostic['message']}", flush=True)
            print("[stonkfly] Dashboard activo; entrenamiento en /training. Sin reinicio automático de paper.", flush=True)

        return supervise(dashboard, worker, stopped)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for process in reversed(processes):
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
