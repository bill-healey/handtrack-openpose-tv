import time


class Hysteresis:
    def __init__(self):
        self.state = None
        self.unchanged_since = time.time()
        self.consecutive_count = 0

    def reset(self):
        self.state = None
        self.unchanged_since = time.time()
        self.consecutive_count = 0

    def update_state(self, state):
        if state != self.state:
            self.state = state
            self.unchanged_since = time.time()
            self.consecutive_count = 0
        else:
            self.consecutive_count += 1

    def is_stable(self, state, secs=None, consecutive=None):
        if self.state != state:
            return False

        if secs is not None:
            if time.time() < self.unchanged_since + secs:
                return False

        if consecutive is not None:
            if self.consecutive_count < consecutive:
                return False

        return True
