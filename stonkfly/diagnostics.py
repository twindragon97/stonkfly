"""Allowlisted public diagnostics: never echo account/SDK exception contents."""

SOURCE_CHANGED = "Run source/protocol changed; use a separate paper run or explicitly review migration"


def public_diagnostic(record):
    reason = record.get("reason") if isinstance(record, dict) else None
    if reason == SOURCE_CHANGED:
        return {"code": "source_changed", "message":
                "La sesión guardada pertenece a otra versión del código o del protocolo. "
                "Conserva esa sesión y elige una carpeta paper nueva con STONKFLY_RUN. "
                "El entrenamiento histórico sigue disponible."}
    if reason == "Checkpoint integrity mismatch":
        return {"code": "checkpoint_integrity", "message":
                "El checkpoint no supera la verificación de integridad. "
                "Revisa el estado guardado antes de reanudar paper. El entrenamiento sigue disponible."}
    return {"code": "worker_stopped", "message":
            "Paper se ha detenido. Consulta error.json en su carpeta de ejecución. "
            "La UI y el entrenamiento histórico siguen disponibles; no se reintentan operaciones."}
