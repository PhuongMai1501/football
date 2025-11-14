# pending
from collections import deque

class Ball_tracker:

    def __init__(self, tracker, lag=12):
        self.tracker = tracker
        self.lag = int(lag)

        self.frames_buf = deque()
        self.tracks_buf = deque()
        self.balls_buf  = deque()

    def push(self, frame, cur_tracks):
        self.frames_buf.append(frame)
        self.tracks_buf.append(cur_tracks)
        self.balls_buf.append(cur_tracks['ball'])

    def pop_ready(self):
        if len(self.balls_buf) < (self.lag + 1):
            return None

        balls_smoothed = self.tracker.interpolate_ball_positions(list(self.balls_buf))
        frame0  = self.frames_buf.popleft()
        tracks0 = self.tracks_buf.popleft()
        self.balls_buf.popleft()

        tracks0['ball'] = balls_smoothed[0]
        return frame0, tracks0

    def flush_all(self):

        while len(self.balls_buf) > 0:
            balls_smoothed = self.tracker.interpolate_ball_positions(list(self.balls_buf))
            frame0  = self.frames_buf.popleft()
            tracks0 = self.tracks_buf.popleft()
            self.balls_buf.popleft()
            tracks0['ball'] = balls_smoothed[0]
            yield frame0, tracks0
