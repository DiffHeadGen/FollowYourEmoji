from functools import cached_property
import cv2
import numpy as np
import imageio

from media_pipe.mp_utils import LMKExtractor
from media_pipe.draw_util import FaceMeshVisualizer
from tqdm import tqdm


class Tracker:
    @cached_property
    def lmk_extractor(self):
        return LMKExtractor()

    @cached_property
    def vis(self):
        return FaceMeshVisualizer(forehead_edge=False)

    def track(self, video_path, output_path):
        assert output_path.endswith(".npy"), "Output path must be a .npy file."
        frames = imageio.get_reader(video_path)
        face_results = []
        motions = []
        for frame in tqdm(frames, total=frames.count_frames(), desc="Tracking"):
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            face_result = self.lmk_extractor(frame_bgr)
            assert face_result is not None, "Can not detect a face in the reference image."
            face_result["width"] = frame_bgr.shape[1]
            face_result["height"] = frame_bgr.shape[0]

            face_results.append(face_result)
            lmks = face_result["lmks"].astype(np.float32)
            motion = self.vis.draw_landmarks((frame_bgr.shape[1], frame_bgr.shape[0]), lmks, normed=True)
            motions.append(motion)
        print(face_results[0]["lmks"].shape)
        np.save(output_path, face_results)
        print(output_path, "done")


def main():
    tracker = Tracker()
    tracker.track("data/video.mp4", "data/video.npy")

if __name__ == '__main__':
    main()