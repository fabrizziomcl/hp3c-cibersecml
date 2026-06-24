"""Utilidades de cómputo de alto desempeño (HPC).

Centraliza la resolución del número de *workers* de paralelismo para que todas
las fases del pipeline (transformación y entrenamiento) usen el mismo criterio:
nunca solicitar más procesos que núcleos físicos disponibles. La función
expone tanto el valor solicitado como el efectivo y el detectado en la máquina,
de modo que esa información pueda registrarse en los logs y en el reporte JSON.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolucionWorkers:
    """Resultado de resolver cuántos *workers* usar.

    Atributos:
        solicitados: número de workers pedido por la configuración.
        disponibles: núcleos lógicos detectados en la máquina.
        efectivos: número de workers que realmente se usará
            (acotado entre 1 y ``disponibles``).
    """

    solicitados: int
    disponibles: int
    efectivos: int

    @property
    def hubo_recorte(self) -> bool:
        """Indica si se pidieron más workers de los disponibles."""
        return self.solicitados > self.disponibles

    def como_dict(self) -> dict:
        """Serializa la resolución para incluirla en el reporte JSON."""
        return {
            "workers_solicitados": self.solicitados,
            "workers_disponibles": self.disponibles,
            "workers_efectivos": self.efectivos,
            "hubo_recorte": self.hubo_recorte,
        }


def detectar_nucleos_disponibles() -> int:
    """Devuelve el número de núcleos lógicos de la máquina (mínimo 1)."""
    return os.cpu_count() or 1


def resolver_workers(solicitados: int) -> ResolucionWorkers:
    """Resuelve el número de *workers* a usar de forma segura.

    Acepta el valor de configuración ``solicitados`` y lo acota al rango
    ``[1, núcleos disponibles]``. Así, una configuración de 16 workers en una
    máquina (o *runner* de CI) con menos núcleos no sobre-suscribe la CPU, y el
    valor efectivo queda registrado para reportarlo con transparencia.

    Args:
        solicitados: número de workers pedido por la configuración. El valor
            especial ``-1`` se interpreta como "todos los núcleos disponibles"
            por compatibilidad, aunque la configuración por defecto usa 16.

    Returns:
        Una instancia de :class:`ResolucionWorkers` con los tres valores.
    """
    disponibles = detectar_nucleos_disponibles()
    if solicitados is None or solicitados == -1:
        solicitados = disponibles
    efectivos = max(1, min(int(solicitados), disponibles))
    return ResolucionWorkers(
        solicitados=int(solicitados),
        disponibles=disponibles,
        efectivos=efectivos,
    )
