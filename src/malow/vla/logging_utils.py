import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.info_rank0 = logger.info
    logger.warning_rank0 = logger.warning
    return logger
