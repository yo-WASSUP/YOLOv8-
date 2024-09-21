import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 多边形点
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR值
        "text_color": (255, 255, 255),  # 区域文本颜色
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(200, 250), (640, 250), (640, 750), (200, 750)]),  # 多边形点
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR值
        "text_color": (0, 0, 0),  # 区域文本颜色
    },
]


def mouse_callback(event, x, y, flags, param):
    """
    处理区域操作的鼠标事件。

    参数:
        event (int): 鼠标事件类型(例如cv2.EVENT_LBUTTONDOWN)。
        x (int): 鼠标指针的x坐标。
        y (int): 鼠标指针的y坐标。
        flags (int): OpenCV传递的其他标志。
        param: 传递给回调函数的其他参数(在此函数中未使用)。

    全局变量:
        current_region (dict): 表示当前选定区域的字典。

    鼠标事件:
        - LBUTTONDOWN: 为包含点击点的区域启动拖动。
        - MOUSEMOVE: 如果正在拖动，则移动选定区域。
        - LBUTTONUP: 结束选定区域的拖动。

    注释:
        - 此函数旨在用作OpenCV鼠标事件的回调函数。
        - 需要存在"counting_regions"列表和"Polygon"类。

    示例:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # 鼠标左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # 鼠标左键抬起事件
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    在视频中使用YOLOv8和ByteTrack运行区域计数。

    支持移动区域进行特定区域内的实时计数。
    支持多个区域计数。
    区域可以是多边形或矩形形状。

    参数:
        weights (str): 模型权重路径。
        source (str): 视频文件路径。
        device (str): 处理设备cpu, 0, 1
        view_img (bool): 显示结果。
        save_img (bool): 保存结果。
        exist_ok (bool): 覆盖现有文件。
        classes (list): 要检测和跟踪的类别
        line_thickness (int): 边界框厚度。
        track_thickness (int): 跟踪线厚度
        region_thickness (int): 区域厚度。
    """
    vid_frame_count = 0

    # 检查源路径
    if not Path(source).exists():
        raise FileNotFoundError(f"源路径'{source}'不存在。")

    # 设置模型
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # 提取类别名称
    names = model.model.names

    # 视频设置
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # 输出设置
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    # 遍历视频帧
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # 提取结果
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 边界框中心

                track = track_history[track_id]  # 绘制跟踪线
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # 检查目标是否在区域内
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1
            # 绘制区域(多边形/矩形)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # 重新初始化每个区域的计数
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
def parse_opt():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8m.pt", help="初始权重路径")
    parser.add_argument("--device", default="0", help="cuda设备, 如0或cpu")
    parser.add_argument("--source", type=str, default="street_people.mkv", help="视频文件路径")
    parser.add_argument("--view-img", type=bool, default="True", help="显示结果")
    parser.add_argument("--save-img", type=bool, default="True", help="保存结果")
    parser.add_argument("--exist-ok", type=bool, default="True", help="")
    parser.add_argument("--classes", nargs="+", type=int, default="0", help="按类别过滤: --classes 0, 或 --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="边界框厚度")
    parser.add_argument("--track-thickness", type=int, default=2, help="跟踪线厚度")
    parser.add_argument("--region-thickness", type=int, default=4, help="区域厚度")

    return parser.parse_args()
def main(opt):
    """主函数。"""
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)