import cv2
import argparse
import time
import os


def main(output_path, camera_number, width, height, directory):
    # make directory

    new_path = os.path.join(output_path, directory)
    print(f"Saving videos in directory: {new_path}")
    os.makedirs(new_path)

    # Open the video capture for stereo camera
    cap = cv2.VideoCapture(camera_number)

    # Creating resolution
    resolution = (int(args.width), int(args.height))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Create video writers for left and right cameras
    left_writer = cv2.VideoWriter(os.path.join(new_path, str(int(time.time())) + '_left_camera_video.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (width // 2, height))
    right_writer = cv2.VideoWriter(os.path.join(new_path, str(int(time.time())) + '_right_camera_video.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (width // 2, height))

    count_till = 5

    print("Initialising camera...")
    for i in range(count_till):
        ret, frame = cap.read()
        print(f"Frame count: {i}")

    print(ret)

    while True:
        # Read frames from the stereo camera
        ret, frame = cap.read()

        if not ret:
            break

        # Split the frame into left and right images
        left_frame = frame[:, :width // 2, :]
        right_frame = frame[:, width // 2:, :]

        # Write frames to the corresponding video writers
        left_writer.write(left_frame)
        right_writer.write(right_frame)

        # # Display the frames
        cv2.imshow('Left Camera', left_frame)
        cv2.imshow('Right Camera', right_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writers
    cap.release()
    left_writer.release()
    right_writer.release()

    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stereo Camera Video Capture')
    parser.add_argument('-o','--output_video', default='', type=str, help='Path to store the videos')
    parser.add_argument('-c','--camera', default=4, type=int, help='Camera number')
    parser.add_argument('-width', "--width", default=2560, help="Width of the image resolution. Defaults to 640.")
    parser.add_argument('-height', "--height", default=720, help="Height of the image resolution. Defaults to 420.")
    parser.add_argument('-d', "--directory", help="Directory to store the left and right videos together")


    args = parser.parse_args()

    # Call the main function with command line arguments
    main(args.output_video, args.camera, args.width, args.height, args.directory)
