import cv2
import numpy as np
import time
import os


def normalize(frame):
    """
    Normalize a image

    :param: frame: frame we want to normalize / type: np.ndarray
    :return: norm_frame: normalized frame / type: np.ndarray
    """
    norm_frame = np.round((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype(np.uint8)
    return norm_frame


def get_metadata(cfg_path):
    """
    Get raw metadata from file

    :param: cfg_path: path where configuration file is stored / type: string
    :return: metadata: raw metadata / type: dict
             + filename
             + width
             + height
             + channels
             + fps
             + dtype
    """
    lines = []

    with open(cfg_path, 'r') as f:
        for ln in f:
            line = ln.rstrip().split(': ')
            lines.append(line)

    metadata = dict(zip(list(map(itemgetter(0), lines)),
                        list(map(itemgetter(1), lines))))

    metadata['width'] = int(metadata['width'])
    metadata['height'] = int(metadata['height'])
    metadata['fps'] = int(metadata['fps'])
    metadata['channels'] = int(metadata['channels'])

    return metadata


class RawVideoCapture:
    """
    Class to read raw frames from a binary file created with RawVideoWriter.
    """

    def __init__(self, file_root, frame_size, channels, dtype):
        """
        RawVideoCapture from raw video file created with RawVideoWriter

        :param: filename: Name of the file / type: str
        :param: frame_size: Size of the frame in (width, height) format / type: tuple[int, int]
        :param: channels: Number of channels (e.g., grayscale has 1 channel, RGB image has 3 channels) / type: int
        :param: dtype: Data type (e.g., np.uint8 for 8-bit channels, np.uint16 for 16-bit channels, etc.) /
                type: np.dtype
        """
        self.f = open(file_root, 'rb')
        self.frame_size = frame_size
        self.channels = channels
        self.dtype = np.dtype(dtype)

    def read(self):
        """
        Reads a frame from the raw file


        :return: ret: Evaluate if there was an exception occurred / type: bool
        :return: frame: Corresponding frame / type: np.ndarray
        """
        frame = self.f.read(self.frame_size[0] * self.frame_size[1] * self.channels * self.dtype.itemsize)
        try:
            if self.channels > 1:
                shape = (self.frame_size[1], self.frame_size[0], self.channels)
            else:
                shape = (self.frame_size[1], self.frame_size[0])
            frame = np.frombuffer(frame, dtype=self.dtype).reshape(shape)
        except ValueError:
            ret = False
            frame = None
        else:
            ret = True
        return ret, frame

    def release(self):
        self.f.close()

    def __del__(self):
        self.release()


class RawVideo:
    """
    Class to play  frames from a video.
    """

    def __init__(self, frames, metadata):
        """
        :param: frames: video frames of the raw file stored / type: np.ndarray
        :param: fps: frames per second of the video we want to play / type: int
        """
        self.frames = frames
        self.metadata = metadata

    def play(self, path):
        """
        Play a raw video

        :param: path: output path to store frames saved while playing / type: string
        :return: anything
        """
        fcount = 0
        n_save = 0
        freeze = False
        delay = 1000 / self.metadata['fps']
        first_frame = True

        os.chdir(path)

        window_name = "Video Raw Player"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while fcount < len(self.frames):
            time0 = time.time()

            if first_frame:
                cv2.resizeWindow(window_name, self.metadata['width'] * 2, self.metadata['height'] * 2)
                first_frame = False

            norm_frame = normalize(self.frames[fcount])
            cv2.imshow(window_name, norm_frame)

            if not freeze:
                fcount += 1

            key = cv2.waitKey(1)

            if key:
                if key == 100:
                    fcount += 1
                    freeze = True
                if key == 97:
                    fcount -= 1
                    freeze = True
                if key == 32:
                    if freeze:
                        freeze = False
                    else:
                        freeze = True
                if key == 115:
                    cv2.imwrite('frame' + str(n_save) + '.png', norm_frame)
                    n_save += 1
                if key == 27:
                    break

            key = None
            time1 = time.time()

            if (time1 - time0) * 1000 < delay:
                time.sleep((delay - 1000 * (time1 - time0)) / 1000)

            if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                break

        cv2.destroyAllWindows()

    def convert(self, path):
        """
        Convert a raw video into a mkv video file.

        :param: path: path: output path to store video after conversion / type: string
        :return: anything
        """
        os.chdir(path)
        vw = cv2.VideoWriter(filename='video_output.mkv',
                             fourcc=cv2.VideoWriter_fourcc('F','F','V','1'),
                             fps=self.metadata['fps'],
                             frameSize=(self.metadata['width'], self.metadata['height']),
                             isColor=False)

        for fcount in range(len(self.frames)):
            norm_frame = normalize(self.frames[fcount])
            vw.write(norm_frame)

        print('\nVideo conversion completed.')

        vw.release()






