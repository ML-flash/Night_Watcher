import signal
import logging

logger = logging.getLogger(__name__)

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.shutdown_requested = True
        logger.info(f"Shutdown requested (signal {signum})")

    def should_continue(self):
        return not self.shutdown_requested
