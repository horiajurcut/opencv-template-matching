from imutils.video import FileVideoStream
from imutils.video import FPS

import os
import time
import cv2
import youtube_dl
import numpy as np
import bitarray


IMAGE_TRAINING_SET_WEAPONS_PATH = "training_images/weapons/"


class Task:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def ready():
        return True

    def get(self):
        return self.data


def youtube_download_hook(download):
    """Progress hook called while downloading a Youtube video."""
    if download["status"] == "finished":
        process_youtube_video(download["filename"])


def process_video_frame(frame_details, weapon_templates):
    """Process each individual frame."""
    frame, width, height = frame_details

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top_left, top_right, bottom_left, bottom_right = split_frame(gray)

    character_bit_mask = []
    for character, weapon_template in weapon_templates.iteritems():
        x, y = template_matching(bottom_right, weapon_template)
        weapon, bw, bh = weapon_template

        if x is None or y is None:
            character_bit_mask.append(False)
            continue

        character_bit_mask.append(True)

        cv2.rectangle(
            frame,
            (x + int(width / 2), y + int(height / 2)),
            (x + int(width / 2) + bw, y + int(height / 2) + bh),
            (0, 0, 255), 2
        )

    return frame, character_bit_mask


def process_youtube_video(filename):
    """Process each frame from local video file."""
    weapon_templates = generate_weapons_template()
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    character_usage = []

    source_video = FileVideoStream(filename).start()
    source_width, source_height = (1280, 720)

    time.sleep(1.0)
    fps_counter = FPS().start()

    processed_video = cv2.VideoWriter()
    processed_video.open(
        os.path.splitext(filename)[0] + "_processed.mov",
        video_codec,
        60.0,
        (int(source_width), int(source_height)),
        True
    )

    while source_video.more():
        frame = source_video.read()

        output_frame, character_bit_mask = process_video_frame(
            (frame, source_width, source_height), weapon_templates)
        character_usage.append(bitarray.bitarray(character_bit_mask))

        processed_video.write(output_frame)
        # cv2.imshow("Character", output_frame)
        # cv2.waitKey(0)

        fps_counter.update()
    fps_counter.stop()

    print("[INFO] Elapsed time: {:.2f}".format(fps_counter.elapsed()))
    print("[INFO] Approximate FPS: {:.2f}".format(fps_counter.fps()))
    # print "Character Usage: ", character_usage

    source_video.stop()
    processed_video.release()
    cv2.destroyAllWindows()


def split_frame(frame):
    """Slice current frame into quadrants."""
    w, h = frame.shape[::-1]

    top_left = frame[0:(h / 2), 0:(w / 2)]
    top_right = frame[0:(h / 2), (w / 2):w]
    bottom_left = frame[(h / 2):h, 0:(w / 2)]
    bottom_right = frame[(h / 2):h, (w / 2):w]

    return top_left, top_right, bottom_left, bottom_right


def template_matching(image, template):
    """Match template image with current frame."""
    template_image, width, height = template

    result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    locations = np.where(result >= threshold)

    average_x = 0
    average_y = 0
    matches_found = zip(*locations[::-1])
    for pt in matches_found:
        average_x += pt[0]
        average_y += pt[1]

    if len(matches_found):
        average_x /= len(matches_found)
        average_y /= len(matches_found)
        cv2.rectangle(
            image,
            (average_x, average_y),
            (average_x + width, average_y + height),
            (0, 0, 255), 2
        )
        return average_x, average_y

    return None, None


def generate_weapons_template():
    """Create OpenCV template images from weapon files."""
    weapon_files = []
    weapon_templates = {}

    for (path, directories, files) in os.walk(IMAGE_TRAINING_SET_WEAPONS_PATH):
        weapon_files.extend(files)

    for weapon_file in weapon_files:
        if not weapon_file.endswith(".png"):
            continue

        weapon_template = cv2.imread(IMAGE_TRAINING_SET_WEAPONS_PATH + weapon_file, 0)
        width, height = weapon_template.shape[::-1]
        weapon_templates[os.path.splitext(weapon_file)[0]] = (weapon_template, width, height)

        # cv2.imshow(os.path.splitext(weapon_file)[0], weapon_template)

    return weapon_templates


def main():
    ydl_opts = {
        "format": "mp4",
        "progress_hooks": [youtube_download_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # ydl.download(["https://www.youtube.com/watch?v=2YvU1BNgdEU"])
        ydl.download(["https://www.youtube.com/watch?v=pe2cKw2UK3k"])


if __name__ == "__main__":
    main()
