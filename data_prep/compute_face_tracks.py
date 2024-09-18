import pandas as pd
import ast
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_file",
        type=str,
        default="../data/faces.csv",
        help="Path to the faces CSV file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="../data/faces/config.json",
        help="Path to the face detector config file",
    )
    parser.add_argument(
        "--shot_file",
        type=str,
        default="../data/shots.csv",
        help="Path to the shots CSV file",
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        default="video_ids.csv",
        help="Path to the video ids CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/face_tracks.json",
        help="Path to save the face tracks JSON file",
    )
    return parser.parse_args()


def assign_shot_id(row, shots_df):
    # For each face, find the shot where the face's time_seconds falls between Start Time (seconds) and End Time (seconds)
    matching_shot = shots_df[
        (shots_df["Start Time (seconds)"] <= row["time_seconds"])
        & (shots_df["End Time (seconds)"] >= row["time_seconds"])
    ]

    # If a matching shot is found, return the shot_id
    if not matching_shot.empty:
        return matching_shot.iloc[0]["shot_id"]

    return None


def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Bounding boxes should be in the format {'x': x, 'y': y, 'w': w, 'h': h}
    """
    x1_min, y1_min = box1["x"], box1["y"]
    x1_max, y1_max = x1_min + box1["w"], y1_min + box1["h"]

    x2_min, y2_min = box2["x"], box2["y"]
    x2_max, y2_max = x2_min + box2["w"], y2_min + box2["h"]

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Calculate IoU
    iou = inter_area / union_area
    return iou


def extract_face_tracks(grouped_faces, iou_threshold=0.5):
    """
    For each group of faces (i.e., faces in the same shot), extract face tracks.
    A face track is a sequence of consecutive faces with IoU > iou_threshold.
    """
    face_tracks = []
    for face in grouped_faces:
        matched = False
        for track in face_tracks:
            if (
                compute_iou(face["facial_area"], track[-1]["facial_area"])
                > iou_threshold
            ):
                track.append(face)
                matched = True
                break

        if not matched:
            face_tracks.append([face])

    return face_tracks


def main(args):
    face_file = args.face_file
    config_file = args.config_file
    shot_file = args.shot_file
    video_ids_file = args.video_ids
    output_path = args.output_path

    cfg = json.load(open(config_file))
    fps_faces = float(cfg["fps"])

    faces = pd.read_csv(face_file)
    shots = pd.read_csv(shot_file)
    video_ids = pd.read_csv(video_ids_file).video_id.tolist()

    all_face_tracks = {}
    for _, video_id in enumerate(video_ids):
        video_faces = faces[(faces["video_id"] == video_id)]
        video_shots = shots[(shots["video_id"] == video_id)]
        if len(video_faces) == 0:
            continue

        video_faces.loc[:, "time_seconds"] = video_faces.frame_number.apply(
            lambda x: int(x) / fps_faces
        )
        video_faces.loc[:, "shot_id"] = video_faces.apply(
            assign_shot_id, shots_df=video_shots, axis=1
        )
        grouped_by_shot = video_faces.groupby("shot_id")
        face_tracks_by_shot = {}
        for shot_id, faces_in_shot in grouped_by_shot:
            faces_in_shot = [
                {
                    "face_id": row["face_id"],
                    "facial_area": ast.literal_eval(row["facial_area"]),
                }
                for _, row in faces_in_shot.iterrows()
            ]
            if len(faces_in_shot) == 0:
                continue

            face_tracks_by_shot[shot_id] = extract_face_tracks(faces_in_shot)

        all_face_tracks[video_id] = face_tracks_by_shot

    # Save all face tracks
    with open(output_path, "w") as f:
        json.dump(all_face_tracks, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
