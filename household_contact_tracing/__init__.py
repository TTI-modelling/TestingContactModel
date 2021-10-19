import pathlib
import sys

from loguru import logger


def set_up_logger():
    log_path = pathlib.Path("simulation.log")
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass

    # remove the default sink before adding new ones
    logger.remove()
    logger.add(sys.stderr, format="{level} - {message}", level="INFO")
    logger.add("simulation.log", format="{time:YYYY-MM-DD HH:mm:ss}: {level} - {message}",
               level="DEBUG", delay=True, enqueue = True)


set_up_logger()
