from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

import motmetrics as mm
import numpy as np

logger.setLevel(logging.INFO)


def motMetricsEnhancedCalculator(gt, t):
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5
        )  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={
            "idf1": "IDF1",
            "idp": "IDP",
            "idr": "IDR",
            "recall": "Rcll",
            "precision": "Prcn",
            "num_objects": "GT",
            "mostly_tracked": "MT",
            "partially_tracked": "PT",
            "mostly_lost": "ML",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_switches": "IDsw",
            "num_fragmentations": "FM",
            "mota": "MOTA",
            "motp": "MOTP",
        },
    )
    print(strsummary)


def demo(opt):
    result_root = opt.output_root if opt.output_root != "" else "."
    mkdir_if_missing(result_root)

    logger.info("Starting tracking...")
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    video_filename = osp.split(opt.input_video)[-1]

    # ground truth file is in the same folder with _gt.txt extension
    gt_filename = video_filename.replace(".mp4", "_gt.txt")
    gt_file = osp.join(osp.split(opt.input_video)[0], gt_filename)
    if not osp.exists(gt_file):
        raise ValueError("Ground truth file not found: {}".format(gt_file))

    result_filename = os.path.join(result_root, video_filename.replace(".mp4", "_t.txt"))

    frame_rate = dataloader.frame_rate
    video_length = dataloader.__len__()

    frame_dir = None if opt.output_format == "text" else osp.join(result_root, "frame")
    eval_seq(
        opt,
        dataloader,
        "mot",
        result_filename,
        save_dir=frame_dir,
        show_image=False,
        frame_rate=frame_rate,
        use_cuda=opt.gpus != [-1],
    )

    # Calculate MOT metrics
    gt = np.loadtxt(gt_file, delimiter=",")
    t = np.loadtxt(result_filename, delimiter=",")

    # # rescale because we scale the video to 1920x1080 in the dataloader
    # t[:, 2] = t[:, 2] / 2
    # t[:, 4] = t[:, 4] / 2

    motMetricsEnhancedCalculator(gt, t)

    if opt.output_format == "video":
        output_video_path = osp.join(result_root, video_filename.replace(".mp4", "_result.mp4"))
        cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -r {} -b 5000k -c:v mpeg4 {}".format(
            osp.join(result_root, "frame"), frame_rate, output_video_path
        )
        os.system(cmd_str)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()
    demo(opt)
