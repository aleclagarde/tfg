from testing.main import make_api_call
from reports.reports import make_reports
import logging
import sys
import os

project_dir = os.path.abspath('.')
src_path = os.path.join(project_dir, 'src')
sys.path.append(src_path)

prune = [0, 0.1, 0.2, 0.3]

logging.info('Making api call...')
make_api_call()
logging.info('Testing completed.')

logging.info('Making reports...')
make_reports()
logging.info('Reports completed')
logging.info('EXECUTION COMPLETED')