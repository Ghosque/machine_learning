import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    filename='data/logging.log',
                    filemode='a')
# 配置logger并设置等级为DEBUG
logger = logging.getLogger('logging_write')

logger.debug('Here has a bug')



