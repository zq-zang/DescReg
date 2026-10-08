from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import gzsd_eval, fast_eval_recall


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--result-proposal', help='result file path',default=None)
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--num-seen',
        type=int,
        default=16,
        help='seen classes num')
    parser.add_argument(
        '--classwise', action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    print("evaluating GZSD performance...")
    gzsd_eval(args.result, test_dataset, args.iou_thr, args.num_seen, args.classwise)

    proposal_results = mmcv.load(args.result_proposal)
    if args.result_proposal is not None:
        fast_eval_recall(proposal_results, test_dataset, 1000, iou_thrs=0.5, cls_wise=True)

if __name__ == '__main__':
    main()
