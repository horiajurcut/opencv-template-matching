from __future__ import unicode_literals

import cv2
import youtube_dl
import numpy as np

template = cv2.imread('training_images/zarya_weapon.png', 0)


def youtube_download_hook(download):
    """Progress hook called while downloading a Youtube video."""
    if download['status'] == 'finished':
        process_youtube_video(download['filename'])


def process_youtube_video(filename):
    """Process each frame from local video file."""
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tl, tr, bl, br = split_frame(gray)
        enhanced_br = enhance_image(br)

        template_matching(br, enhanced_br)

        cv2.imshow('Character Normal', br)
        cv2.imshow('Character Enhanced', enhanced_br)

        if cv2.waitKey(0) & 0xFF == ord('q'):
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


def template_matching(image, enhanced_image):
    """Match template image with current frame."""
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(enhanced_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


def enhance_image(image):
    """Enhance greyscale image to better detect template."""
    # 0 = threshold, 255 = max_value
    th, eq = cv2.threshold(image, 130, 140, cv2.THRESH_BINARY);

    return eq


def main():
    ydl_opts = {
        'format': 'mp4',
        'progress_hooks': [youtube_download_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=2YvU1BNgdEU'])


if __name__ == "__main__":
    main()
