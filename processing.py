from __future__ import unicode_literals

import os
import cv2
import youtube_dl
import numpy as np


IMAGE_TRAINING_SET_WEAPONS_PATH = "training_images/weapons/"


def youtube_download_hook(download):
    """Progress hook called while downloading a Youtube video."""
    if download["status"] == "finished":
        process_youtube_video(download["filename"])


def process_youtube_video(filename):
    """Process each frame from local video file."""
    weapon_templates = generate_weapons_template()
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        top_left, top_right, bottom_left, bottom_right = split_frame(gray)

        for character, weapon_template in weapon_templates.iteritems():
            template_matching(bottom_right, weapon_template)

        cv2.imshow("Character", bottom_right)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cap.release()
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

    for pt in zip(*locations[::-1]):
        cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)


def generate_weapons_template():
    """Create OpenCV template images from weapon files."""
    weapon_files = []
    weapon_templates = {}

    for (path, directories, files) in os.walk(IMAGE_TRAINING_SET_WEAPONS_PATH):
        weapon_files.extend(files)

    for weapon_file in weapon_files:
        weapon_template = cv2.imread(IMAGE_TRAINING_SET_WEAPONS_PATH + weapon_file, 0)
        width, height = weapon_template.shape[::-1]
        weapon_templates[os.path.splitext(weapon_file)[0]] = (weapon_template, width, height)

        cv2.imshow(os.path.splitext(weapon_file)[0], weapon_template)

    return weapon_templates


def main():
    ydl_opts = {
        "format": "mp4",
        "progress_hooks": [youtube_download_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(["https://www.youtube.com/watch?v=2YvU1BNgdEU"])
        # ydl.download(["https://www.youtube.com/watch?v=pe2cKw2UK3k"])


if __name__ == "__main__":
    main()
