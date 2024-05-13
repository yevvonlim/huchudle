import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor

def videocap(args):
    filepath, frame_seconds, thread_num = args

    # Open the video file
    vidcap = cv2.VideoCapture(filepath)

    # Get video information
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # print("Video Information - Length: {}, Width: {}, Height: {}, FPS: {}".format(length, width, height, fps))

    # Get video name
    video_name = os.path.basename(filepath)[:-4]

    # Create output folder if not exists
    output_folder = 'output_frames_2'
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except OSError:
        print('Error: Creating directory. ' + output_folder)

    # Initialize variables
    frame_rate = int(fps * frame_seconds)  # Desired frames per second
    count = 0

    while vidcap.isOpened():
        # Move to the frame every specified number of seconds
        frame_number = int(count * frame_rate)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, image = vidcap.read()

        if ret:
            # Save the image
            frame_number_str = "{}{}".format(str(thread_num).zfill(6), str(count).zfill(4))  # Include thread number in the filename
            result = cv2.imwrite(os.path.join(output_folder, "{}.jpg".format(frame_number_str)), image)
            print('Saved frame:', frame_number_str)
            count += 1
            if count > 100:
                break
        else:
            break

    # Release video resources
    vidcap.release()
    return count - 1  # Return the number of processed frames

if __name__ == '__main__':
    start = time.time()

    total_results = 0
    num_threads = os.cpu_count()  # Get CPU core count
    print(num_threads)

    # Get all video files in the dataset folder
    video_files = [os.path.join('../datasets/celebav-hq/35666', filename) for filename in os.listdir('../datasets/celebav-hq/35666') if filename.endswith('.mp4')]

    # Frame interval setting (seconds)
    frame_seconds = 5 

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Process each video file
        args = [(filepath, frame_seconds, i+1) for i, filepath in enumerate(video_files)]  # Thread number starts from 1
        futures = [executor.submit(videocap, arg) for arg in args]

        # Get results when completed
        for future in futures:
            total_results += future.result()

    end = time.time()

    print("Total execution time: %.2f seconds" % (end - start))
    print("Total results: %s" % total_results)
