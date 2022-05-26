#!/usr/bin/python

import os
import h5py
import cv2
import numpy as np

from traffik.models import ResNet
from traffik.KTS import cpd_auto
from traffik.usersum import build_frame_binary_array, extract_action_frames
from tqdm import tqdm



class DatasetGenerator:

    # Take the i-th frame only if i % TRAIN_FRAME_FRACTION == 0.  
    TRAIN_FRAME_FRACTION = 15

    def __init__(self, sourcepath: str, savepath: str, h5filename='traffikds.h5'):
        """
        Args:
            sourcepath (str):   Path of the directory that contains videos and CSVs
            savepath (str):     Path where to store the hdf5 dataset file
        """
        assert os.path.isdir(sourcepath), "Source path must point to a directory."
        assert os.path.isdir(savepath), "Save path must point to a directory."
        self.sourcepath = sourcepath
        self.savepath   = savepath
        self.h5filepath = os.path.join(savepath, h5filename)
        self.resnet = ResNet()
        self.dataset = {}
        self.videolist = []
        self.csvlist = []
        self.h5file = h5py.File(self.h5filepath, 'w')
        self._set_video_list()
        self._create_hdf5_groups()


    def _set_video_list(self):
        """ Load videos and CSVs filenames into their lists. """
        dircontent = os.listdir(self.sourcepath)
        self.videolist = [ v for v in dircontent if '.mp4' in v ]
        self.csvlist   = [ c for c in dircontent if '.csv' in c ]
        self.videolist.sort()
        self.csvlist.sort()


    def _create_hdf5_groups(self):
        """ Create a group for each video in the HDF5 file. """
        for i, _ in enumerate(self.videolist):
            videokey = f'video_{i+1}'
            self.dataset[videokey] = {}
            self.h5file.create_group(videokey)


    def generate(self):
        """ Generate the dataset and store into HDF5 file. """
        for video_idx, video_filename in enumerate(tqdm(self.videolist)):
            print(f'processing video {video_filename}')
            videopath = os.path.join(self.sourcepath, video_filename)
            csvpath   = os.path.join(self.sourcepath, self.csvlist[video_idx])

            videocap = cv2.VideoCapture(videopath)
            nframes  = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps      = int(videocap.get(cv2.CAP_PROP_FPS))

            usersumm = build_frame_binary_array(nframes, extract_action_frames(csvpath))
            gtscore     = []
            picks       = []
            videofeat       = np.empty((0, 1024), float)
            videofeat_train = np.empty((0, 1024), float)

            for frameidx in tqdm(range(nframes - 1)):
                success, frame = videocap.read()
                assert success, "Cannot capture frame."
                frame_features = self._extract_features(frame)
                if frameidx % self.TRAIN_FRAME_FRACTION == 0:
                    picks.append(frameidx)
                    gtscore.append(self._compute_gtscore(frameidx, usersumm))
                    videofeat_train = np.vstack((videofeat_train, frame_features))
                videofeat = np.vstack((videofeat, frame_features))

            videocap.release()
            cps, nfpseg = self._get_change_points(videofeat, nframes, fps)
            videoname = f'video_{video_idx + 1}'
            self.h5file[videoname]['features'] = videofeat_train
            self.h5file[videoname]['picks'] = np.array(picks)
            self.h5file[videoname]['n_frames'] = nframes
            self.h5file[videoname]['fps'] = fps
            self.h5file[videoname]['change_points'] = cps
            self.h5file[videoname]['n_frame_per_seg'] = nfpseg
            self.h5file[videoname]['user_summary'] = usersumm
            self.h5file[videoname]['gtscore'] = gtscore

        self.h5file.close()


    def _compute_gtscore(self, frameidx, usersumm):
        """ Compute the Ground Truth score (gtscore) for the frame.
            Given n user summaries of the frame (1 if the frame considered
            important, 0 otherwise), let m be the number of user summaries
            that consider the frame imporant, then the gtscore is computed 
            as m / n. In our case, we have only 1 machine-generated "user"
            summary, so the ground truth score is 1 if the object is in the 
            scene, 0 otherwise. 
        """
        return usersumm[frameidx]


    def _extract_features(self, frame):
        """ Extract the frame features using the ResNet """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        return self.resnet(frame).cpu().data.numpy().flatten()


    def _get_change_points(self, videofeat, nframes, fps):
        """ Extract the changepoints from the video features """
        n = nframes / fps
        m = int(np.ceil(n/2.0))
        K = np.dot(videofeat, videofeat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [nframes-1]))
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]
            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))
        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(nframes)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))
        return change_points, n_frame_per_seg