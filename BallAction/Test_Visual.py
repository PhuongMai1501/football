from BallAction.src.ball_action import constants
from BallAction.src.utils import post_processing
from BallAction.scripts.ball_action.visualize import draw_graph
from collections import defaultdict
import numpy as np
import cv2

def load_video_predictions():
    raw_predictions_path = "models/ball_action/test_raw_predictions.npz"
    raw_predictions_npz = np.load(str(raw_predictions_path))
    frame_indexes = raw_predictions_npz["frame_indexes"]
    raw_predictions = raw_predictions_npz["raw_predictions"]
    video_prediction = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    for frame_index, prediction in zip(frame_indexes, raw_predictions):
        video_prediction[frame_index] = prediction

    video_pred_actions = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    for cls, cls_index in constants.class2target.items():
        action_frame_indexes, _ = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **constants.postprocess_params
        )
        for frame_index in action_frame_indexes:
            video_pred_actions[frame_index][cls_index] = 1.0
    return video_prediction, video_pred_actions

class BallActionSpot:
    def __init__(self):
        self.targets = {cls: [] for cls in constants.classes}
        self.predictions = {cls: [] for cls in constants.classes}
        self.pred_actions = {cls: [] for cls in constants.classes}
        self.video_prediction, self.video_pred_actions = load_video_predictions()

    def visualize_frame(self, frame, frame_idx):
        prediction = self.video_prediction[frame_idx]
        pred_action = self.video_pred_actions[frame_idx]

        for cls, cls_index in constants.class2target.items():
            self.predictions[cls].append(prediction[cls_index])
            self.targets[cls].append(0)
            self.pred_actions[cls].append(pred_action[cls_index])

        x = 50
        for cls, y in zip(constants.classes, (500, 700)):
            pass_graph = draw_graph(self.targets[cls], self.predictions[cls], self.pred_actions[cls])
            crop = frame[y: y + pass_graph.shape[0], x: x + pass_graph.shape[1]]
            cv2.addWeighted(pass_graph, 1., crop, 1., 0.0, crop)

        cv2.putText(frame, str(frame_idx), (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)
        
        return frame