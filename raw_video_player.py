import argparse
import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', type=str, choices=['play', 'convert'], required=True, help='Mode: play or convert input video')
    parser.add_argument('--input-source', '-ins', type=str, help='input video root')
    parser.add_argument('--cfg-source', '-cfgs', type=str, help='configuration file root')
    parser.add_argument('--output-source', '-outs', type=str, help='output video root')

    return parser.parse_args()


def main(args):
    """
    Main program execution. Raw metadata is obtained, raw video is stored in np.ndarray format and mode selected
    (play or convert) is performed
    """
    mode = args['mode']
    in_path = args['input_source']
    out_path = args['output_source']
    cfg_path = args['cfg_source']
    frames = []

    metadata = utils.get_metadata(cfg_path)
    cap = utils.RawVideoCapture(in_path, metadata)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(img)
    cap.release()

    raw = utils.RawVideo(frames, metadata, out_path, mode)
    raw()


if __name__ == '__main__':
    args = vars(parse_args())
    main(args)
