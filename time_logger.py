import threading
import sys
import time
import logging
def setup_logger(name):
  formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
          datefmt='%Y-%m-%d %H:%M:%S')
  handler = logging.FileHandler('log.txt', mode='w')
  handler.setFormatter(formatter)
  screen_handler = logging.StreamHandler(stream=sys.stdout)
  screen_handler.setFormatter(formatter)
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler)
  logger.addHandler(screen_handler)
  return logger

def log_periodically(logger, message, interval, stop_event):
  while not stop_event.is_set():
    logger.info(message)
    time.sleep(interval)

def start_logger(message, interval=5, name="logger"):
  stop_event = threading.Event()
  logger = setup_logger(name)
  thread = threading.Thread(target=log_periodically, args=(logger, message, interval, stop_event))
  thread.start()
  def stop_logger():
    stop_event.set() 
    thread.join()
  return stop_logger
