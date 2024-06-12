import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UvaDataset(Dataset):
    def __init__(self, raw_videos, label_videos):
        assert len(raw_videos) == len(label_videos), "Raw videos and label videos list must be of the same length"
        self.raw_videos = raw_videos
        self.label_videos = label_videos
        self.lengths = self.calculate_lengths()

    def calculate_lengths(self):
        lengths = []
        for video in self.raw_videos:
            cap = cv2.VideoCapture(video)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # lengths.append(max(0,length - 2))  # 5 frame
            lengths.append(max(0, length - 1))  # 3 frame
        return lengths

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, global_idx):

        video_idx, local_idx = self.map_global_index_to_video(global_idx)

        # for frame
        # raw_frames = self.load_frames(self.raw_videos[video_idx], local_idx, 5)  # 5 frame
        raw_frames = self.load_frames(self.raw_videos[video_idx], local_idx, 3)  # 3 frame
        # for label
        label_frame = self.load_frames(self.label_videos[video_idx], local_idx + 1, 1)  # middle frame in 3 frame
        return  np.asarray(raw_frames), label_frame[0]

    def map_global_index_to_video(self, global_idx):
        running_sum = 0
        for i, length in enumerate(self.lengths):
            if running_sum + length > global_idx:
                return i, global_idx - running_sum
            running_sum += length
        raise IndexError("Global index out of range")

    def load_frames(self, video, start_idx, num_frames):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.array(frame)
                GrayLevels = 255
                tmax = 256
                s_frame = np.floor((GrayLevels - frame.reshape(100, 120)) * tmax / GrayLevels).astype(int)
                frames.append(s_frame)
            else:
                break

        cap.release()
        return frames

