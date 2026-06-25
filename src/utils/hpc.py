"""Resolución del número de workers de paralelismo, compartida por las fases
de transformación y entrenamiento."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolucionWorkers:
    """Workers solicitados, disponibles en la máquina y efectivamente usados."""

    solicitados: int
    disponibles: int
    efectivos: int

    @property
    def hubo_recorte(self) -> bool:
        return self.solicitados > self.disponibles

    def como_dict(self) -> dict:
        return {
            "workers_solicitados": self.solicitados,
            "workers_disponibles": self.disponibles,
            "workers_efectivos": self.efectivos,
            "hubo_recorte": self.hubo_recorte,
        }


def detectar_nucleos_disponibles() -> int:
    return os.cpu_count() or 1


def resolver_workers(solicitados: int) -> ResolucionWorkers:
    """Acota los workers solicitados a [1, núcleos disponibles] para no
    sobre-suscribir la CPU. ``-1`` equivale a todos los núcleos."""
    disponibles = detectar_nucleos_disponibles()
    if solicitados is None or solicitados == -1:
        solicitados = disponibles
    efectivos = max(1, min(int(solicitados), disponibles))
    return ResolucionWorkers(int(solicitados), disponibles, efectivos)
