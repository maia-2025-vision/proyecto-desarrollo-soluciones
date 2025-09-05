class Settings:
    """Configuración principal del API."""

    HOST: str = "0.0.0.0"  # Acepta conexiones externas
    PORT: int = 8000  # Puerto por defecto
    RELOAD: bool = True  # Recarga automática (solo para desarrollo)


settings = Settings()
