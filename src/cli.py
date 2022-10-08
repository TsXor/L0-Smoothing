import numpy as np
from PIL import Image
import argparse, pathlib

from L0_Smoothing import L0_Smoothing_accel as L0_Smoothing_accel
from L0_Smoothing import L0_Smoothing as L0_Smoothing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fin', metavar='FILE OR FOLDER', type=str,
                        help='input file or folder path')
    parser.add_argument('fout', metavar='FILE OR FOLDER', type=str,
                        help='output file or folder path')
    parser.add_argument('--noaccel', action='store_true',
                        help="disable OpenCL acceleration")
    parser.add_argument('--params', type=float, nargs='+', metavar='', default=[],
                        help='parameters for the algorithm')
    args = parser.parse_args()


    def PILopen(path):
        return np.asarray(Image.open(str(path)))

    def PILshow(img):
        Image.fromarray(img).show()

    def PILsave(img, path):
        Image.fromarray(img).save(str(path))

    def smooth(img):
        print('running smoothing')
        if args.noaccel:
            smoothed = L0_Smoothing(img, *args.params)
        else:
            smoothed = L0_Smoothing_accel(img, *args.params)
        return smoothed

    in_path = pathlib.Path(args.fin)
    out_path = pathlib.Path(args.fout)
    assert (in_path.is_dir() == out_path.is_dir()) or args.fout=='show', \
        'input and output should both be file or folder'

    if in_path.is_dir():
        paths = [*list(in_path.glob('*.jpg')), *list(in_path.glob('*.png'))]
    else:
        assert in_path.suffix in ('.jpg', '.png'), \
            'support jpg and png only'
        paths = [in_path]

    imgs = [PILopen(path) for path in paths]
    outs = [smooth(img) for img in imgs]

    if args.fout=='show':
        for out in outs:
            PILshow(out)
    else:
        if in_path.is_dir():
            opaths = [out_path/p.name for p in paths]
        else:
            opaths = [out_path]

        for opath, out in zip(opaths, outs):
            PILsave(out, opath)

if __name__ == '__main__':
    main()