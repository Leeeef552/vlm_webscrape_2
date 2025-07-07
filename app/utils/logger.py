import logging

class CustomFormatter(logging.Formatter):
    simple_fmt = '[%(asctime)s] [%(levelname)s] %(message)s'
    error_fmt = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'

    def format(self, record):
        if record.levelno >= logging.ERROR:
            self._style._fmt = self.error_fmt
        else:
            self._style._fmt = self.simple_fmt
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter(
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger = logging.getLogger("VLM WEBSCRAPING")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False  # Prevent double logging

# # Usage examples:
# logger.info("This is an info message")
# logger.error("This is an error message")
