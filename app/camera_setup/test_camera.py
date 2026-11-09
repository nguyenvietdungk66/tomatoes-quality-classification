
import cv2
import time

from hikrobot_camera import HikrobotCamera


def main():

    camera = HikrobotCamera(camera_index=0)

    try:

        camera.open()
        camera.start_grabbing()

        print("\nPress Q to exit")

        frame_count = 0
        start_time = time.time()

        while True:

            frame = camera.get_frame(timeout_ms=1000)

            if frame is None:
                print("Frame timeout!")
                continue

            frame_count += 1

            elapsed = time.time() - start_time

            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Hikrobot Live Camera",
                cv2.resize(frame, (960, 640))
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:

        print(f"Error: {e}")

    finally:

        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
