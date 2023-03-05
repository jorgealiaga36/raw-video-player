import argparse
import utils
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    parser_play = subparsers.add_parser('play', help='display video')
    parser_play.add_argument('--input-source', '-ins', type=str, help='input video root')
    parser_play.add_argument('--cfg-source', '-cfgs', type=str, help='configuration file root')
    parser_play.add_argument('--output-source', '-outs', type=str, help='output video root')

    parser_convert = subparsers.add_parser('convert', help='change video format')
    parser_convert.add_argument('--input-source', '-ins', type=str, help='bar help')
    parser_convert.add_argument('--cfg-source', '-cfgs', type=str, help='configuration file root')
    parser_convert.add_argument('--output-source', '-outs', type=str, help='output video root')

    return parser.parse_args()


def main(args):
    """
    Main program execution. Raw metadata is obtained, raw video is stored in np.ndarray format and action required
    (play or convert) is evaluated
    """
    in_path = args['input_source']
    out_path = args['output_source']
    cfg_path = args['cfg_source']
    frames = []

    metadata = utils.get_metadata(cfg_path)
    cap = utils.RawVideoCapture(in_path,
                                (metadata['width'], metadata['height']),
                                metadata['channels'],
                                metadata['dtype']
                                )
    while True:
        ret, img = cap.read()
        if ret:
            frames.append(img)
        else:
            break

    cap.release()
    raw = utils.RawVideo(frames, metadata)

    if args['mode'] == 'play':
        raw.play(out_path)
    if args['mode'] == 'convert':
        raw.convert(out_path)


if __name__ == '__main__':
    args = vars(parse_args())
    main(args)
