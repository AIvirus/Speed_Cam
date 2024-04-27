import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

# Define source and target points for perspective transformation
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

# ViewTransformer class for perspective transformation
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


if __name__ == "__main__":
    # Define paths and thresholds
    source_video_path = 'data/vehicles.mp4'
    target_video_path = 'target/video.mp4'
    confidence_threshold = 0.3
    iou_threshold = 0.7

    # Get video information
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = YOLO("yolov8x.pt")

    # Initialize ByteTrack and other annotators
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=confidence_threshold)
    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Get video frames generator
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    # Initialize ViewTransformer and coordinates dictionary
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Adjust the size of the output display window
    display_width = 800
    display_height = 600

    # Open video sink
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Perform object detection
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            # Transform coordinates
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            # Store coordinates and calculate speed
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6  # Convert to km/h
                    labels.append(f"#{tracker_id} {int(speed)} km/h")


            # Annotate frame and write to sink
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)


            # Resize the frame for display
            resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
