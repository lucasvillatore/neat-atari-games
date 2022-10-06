import logging
import json
from logging import Formatter
from datetime import datetime


class FormatterLogger(Formatter):
    def __init__(self, *args, **kwargs):
        super(FormatterLogger, self).__init__()

    @classmethod
    def log_record_to_dict(cls, record):
        output = {
            'module': record.module,
            'global_event_timestamp': datetime.now().timestamp(),
            'level': record.levelname,
            'message': record.msg,
            'context': record.args,
            'path_name': record.pathname,
            'func_name': record.funcName,
            'line': record.lineno,
        }

        return output

    def format(self, record):
        def safe_str(object, *args, **kwargs):
            """Convert object to str, catching any errors raised."""
            try:
                return str(object, *args, **kwargs)
            except:
                return "<unprintable %s object>" % type(object).__name__

        return json.dumps(self.log_record_to_dict(record), default=safe_str, separators=(",", ":"))


def load() -> None:
    formatter = FormatterLogger()
    
    logger = logging.getLogger()
    logger.setLevel('INFO')
    handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)