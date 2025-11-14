import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self._norm_type = cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher(self._norm_type)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self._norm_type = cv2.NORM_L2
            self.matcher = cv2.BFMatcher(self._norm_type)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1,
                                       blockSize=3, useHarrisDetector=False, k=0.04)

        elif self.method in ['file', 'files']:
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')
            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)

        elif self.method.lower() == 'none':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown GMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

        self._IDENTITY_2x3 = np.eye(2, 3, dtype=np.float32)

    # ---------------- HELPER FUNCTIONS ---------------- #
    def _identity_affine(self):
        return self._IDENTITY_2x3.copy()

    # ----------- sửa lại hàm _prep_desc ----------- #
    def _prep_desc(self, desc):
        """Normalize descriptor dtype to match matcher; drop if invalid."""
        if desc is None:
            return None
        if not isinstance(desc, np.ndarray) or desc.ndim != 2 or desc.shape[0] < 2:
            return None
        # Dựa vào self._norm_type thay vì self.matcher.normType
        if self._norm_type in (cv2.NORM_HAMMING, cv2.NORM_HAMMING2):
            if desc.dtype != np.uint8:
                desc = desc.astype(np.uint8, copy=False)
        else:
            if desc.dtype != np.float32:
                desc = desc.astype(np.float32, copy=False)
        return desc

    def _compatible_desc(self, d1, d2):
        return (
            d1 is not None and d2 is not None and
            isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray) and
            d1.dtype == d2.dtype and d1.ndim == 2 and d2.ndim == 2 and
            d1.shape[1] == d2.shape[1] and d1.shape[0] >= 2 and d2.shape[0] >= 2
        )

    # ---------------- MAIN APPLY FUNCTIONS ---------------- #
    def apply(self, raw_frame, detections=None):
        if self.method in ['orb', 'sift']:
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return self._identity_affine()
        else:
            return self._identity_affine()

    # ---------------- ECC ---------------- #
    def applyEcc(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            print(f'Warning: findTransformECC failed: {e}. Set warp as identity')
            H = self._identity_affine()

        return H.astype(np.float32, copy=False)

    # ---------------- FEATURE MATCHING ---------------- #
    def applyFeaures(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = self._identity_affine()

        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height),
             int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)
        descriptors = self._prep_desc(descriptors)

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        prevD = self._prep_desc(self.prevDescriptors)
        currD = self._prep_desc(descriptors)
        if not self._compatible_desc(prevD, currD):
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        try:
            knnMatches = self.matcher.knnMatch(prevD, currD, k=2)
        except Exception as e:
            print(f"Warning: knnMatch failed: {e}. Using identity warp")
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        matches, spatialDistances = [], []
        maxSpatialDistance = 0.25 * np.array([width, height])
        ratio = 0.75

        for pair in knnMatches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                prev_pt = self.prevKeyPoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt
                sd = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
                if (abs(sd[0]) < maxSpatialDistance[0]) and (abs(sd[1]) < maxSpatialDistance[1]):
                    matches.append(m)
                    spatialDistances.append(sd)

        if len(matches) < 4:
            print("Warning: not enough good matches")
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        spatialDistances = np.asarray(spatialDistances, dtype=np.float32)
        mean_sd = np.mean(spatialDistances, axis=0)
        std_sd = np.std(spatialDistances, axis=0) + 1e-6
        mask_inlier = (np.abs(spatialDistances - mean_sd) < 2.5 * std_sd)

        prevPoints, currPoints = [], []
        for i, m in enumerate(matches):
            if mask_inlier[i, 0] and mask_inlier[i, 1]:
                prevPoints.append(self.prevKeyPoints[m.queryIdx].pt)
                currPoints.append(keypoints[m.trainIdx].pt)

        if len(prevPoints) >= 4:
            H_est, _ = cv2.estimateAffinePartial2D(np.array(prevPoints), np.array(currPoints), cv2.RANSAC)
            if H_est is not None:
                H = H_est.astype(np.float32)
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points for affine')

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    # ---------------- OPTICAL FLOW ---------------- #
    def applySparseOptFlow(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = self._identity_affine()

        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, self.prevKeyPoints, None)

        prevPoints, currPoints = [], []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        if len(prevPoints) >= 4:
            H_est, _ = cv2.estimateAffinePartial2D(np.array(prevPoints), np.array(currPoints), cv2.RANSAC)
            if H_est is not None:
                H = H_est.astype(np.float32)
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points (optflow)')

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        return H

    # ---------------- FILE-BASED ---------------- #
    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=np.float32)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])
        return H
