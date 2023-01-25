from testing.main import make_api_call
from src.app import deploy
from reports.reports import make_reports
import logging

prune = [0, 0.1, 0.2, 0.3]

for prune_pct in prune:
    # Deploy the model
    logging.info('Deploying model...')
    deploy(prune_pct)
    logging.info('Model deployed.')
    logging.info('Making api call...')
    make_api_call(prune_pct)
    logging.info('Testing completed.')

logging.info('Making reports...')
make_reports()
logging.info('Reports completed')
logging.info('EXECUTION COMPLETED')
