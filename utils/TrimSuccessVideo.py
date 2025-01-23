import argparse
import os
import sys
from enum import Enum

import cv2
import ffmpeg
import numpy as np


class Color(Enum):
    GREEN = (0, 128, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)

    @classmethod
    def match_closest(cls, sub_frame):
        border_pixels = np.concatenate(
            (
                sub_frame[:2, :].reshape(-1, 3),  # top
                sub_frame[-2:, :].reshape(-1, 3),  # bottom
                sub_frame[:, :2].reshape(-1, 3),  # left
                sub_frame[:, -2:].reshape(-1, 3),  # right
            ),
            axis=0,
        )
        closest_color = min(  # find the closest color
            list(cls),  # convert class to a list
            key=lambda curr_color: np.sum(  # define key function for comparison
                np.abs(  # take absolute difference
                    np.mean(  # calculate mean of border pixels
                        border_pixels,  # pixels of the border
                        axis=0,  # calculate along the x-axis
                    )
                    - curr_color.value  # subtract current color value
                )
            ),
        )
        return closest_color


def main(
    input_video_path,
    output_video_path,
    n_row,
    n_col,
    trim_color_th,
    trim_ratio,
    max_video_width,
):

    # read
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"{cap.isOpened()=}"

    is_frame_read_successfully, frame = cap.read()
    assert is_frame_read_successfully, f"{is_frame_read_successfully=}"

    x_trim, y_trim, w_trim, h_trim = None, None, None, None

    # find green sub_frame
    h_frame, w_frame, _ = frame.shape
    for i_row in range(n_row):
        if all(s is not None for s in [x_trim, y_trim, w_trim, h_trim]):
            break

        for i_col in range(n_col):
            x_offset = i_col * w_frame // n_col
            y_offset = i_row * h_frame // n_row
            sub_frame = frame[
                y_offset : y_offset + (h_frame // n_row),
                x_offset : x_offset + (w_frame // n_col),
            ]
            closest_color = Color.match_closest(sub_frame)
            print(f"{(i_row, i_col, closest_color.name)=}")

            # check white
            assert closest_color != Color.WHITE, f"{(closest_color != Color.WHITE)=}"

            # check red
            if closest_color == Color.RED:
                continue

            # check green
            assert closest_color == Color.GREEN, f"{(closest_color == Color.GREEN)=}"
            # # trim
            dist_mask = np.where(
                np.mean(np.abs(sub_frame - Color.GREEN.value), axis=2) > trim_color_th
            )
            # # set trim box
            y_min, y_max, x_min, x_max = np.array(
                [func(dist_mask[ax]) for ax in [0, 1] for func in [min, max]]
            ) + [y_offset, y_offset, x_offset, x_offset]
            x_trim = x_min * (1 - trim_ratio[0]) + x_max * trim_ratio[0]
            y_trim = y_min * (1 - trim_ratio[1]) + y_max * trim_ratio[1]
            w_trim = x_min * (1 - trim_ratio[2]) + x_max * trim_ratio[2] - x_trim
            h_trim = y_min * (1 - trim_ratio[3]) + y_max * trim_ratio[3] - y_trim
            break
    cap.release()

    assert all(
        s is not None for s in [x_trim, y_trim, w_trim, h_trim]
    ), f"{[x_trim, y_trim, w_trim, h_trim]=}"

    # trim, scale
    print(f"{(h_frame, w_frame, x_trim, y_trim, w_trim, h_trim)=}")
    input_video = (
        ffmpeg.input(input_video_path)
        .crop(x_trim, y_trim, w_trim, h_trim)
        .filter("scale", max_video_width, -2)
    )

    # write
    if os.path.exists(output_video_path):
        backup_path = f"{output_video_path}~"
        os.rename(output_video_path, backup_path)
        print(f"renamed '{output_video_path}' -> '{backup_path}'")
    assert not os.path.exists(output_video_path)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    try:
        input_video.output(output_video_path).run(overwrite_output=False)
    except ffmpeg._run.Error:
        sys.stderr.write(f"{(input_video_path, output_video_path)}=")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_video_path", required=True, type=str)
    parser.add_argument("-o", "--output_video_path", default="output.mp4")
    parser.add_argument(
        "--n_row", default=2, type=int, help="Number of rows for input video splitting"
    )
    parser.add_argument(
        "--n_col",
        default=3,
        type=int,
        help="Number of columns for input video splitting",
    )
    parser.add_argument(
        "-c",
        "--trim_frame_to_subframe_color_threshold",
        metavar="COLOR_THRESHOLD",
        default=192,
        type=int,
        help="Color threshold for extracting subframe from frame",
    )
    parser.add_argument(
        "-r",
        "--trim_subframe_region_ratio",
        nargs=4,
        metavar=("LEFTTOP_X", "LEFTTOP_Y", "SIZE_X", "SIZE_Y"),
        default=[0, 0, 1, 0.5],
        type=float,
        help="Ratio for trimming region of subframe",
    )
    parser.add_argument("-w", "--max_video_width", default=640, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.input_video_path,
        args.output_video_path,
        args.n_row,
        args.n_col,
        args.trim_frame_to_subframe_color_threshold,
        args.trim_subframe_region_ratio,
        args.max_video_width,
    )
    print(f"{args=}")
    print(f"{args.output_video_path=}")
    print("Done.")
