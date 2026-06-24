import sys


def _formatear(error) -> str:
    """Formatea un error incluyendo archivo y línea donde ocurrió, si existe."""
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    nombre_archivo = exc_tb.tb_frame.f_code.co_filename
    return f"Error en [{nombre_archivo}] línea [{exc_tb.tb_lineno}] mensaje [{error}]"


class CustomException(Exception):
    """Excepción del proyecto que enriquece el mensaje con archivo y línea.

    El parámetro ``error_detail`` (habitualmente ``sys``) se acepta por
    compatibilidad con las llamadas existentes; la información de la traza se
    obtiene internamente vía ``sys.exc_info()``.
    """

    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)
        self.error_message = _formatear(error_message)

    def __str__(self) -> str:
        return self.error_message
