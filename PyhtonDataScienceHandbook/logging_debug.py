import logging
import datetime

# 配置logger并设置等级为DEBUG
logger = logging.getLogger('logging_debug')
logger.setLevel(logging.DEBUG)
# 配置控制台Handler并设置等级为DEBUG
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
# 将Handler加入logger
logger.addHandler(consoleHandler)

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.debug('{} This is a logging.debug'.format(time))



