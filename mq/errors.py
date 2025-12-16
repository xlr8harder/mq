class MQError(Exception):
    pass


class UserError(MQError):
    pass


class ConfigError(MQError):
    pass


class LLMError(MQError):
    def __init__(self, message: str, error_info: dict | None = None):
        super().__init__(message)
        self.error_info = error_info or {}

