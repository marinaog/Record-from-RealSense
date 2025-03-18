import pyrealsense2 as rs
import numpy as np
import cv2
import os

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.raw16, 30)

    try:
        pipeline_profile = pipeline.start(config)
        print("✅ Recording started! Press 'q' to stop.")

        # Create directories to save frames
        os.makedirs("depth", exist_ok=True)
        os.makedirs("color", exist_ok=True)

        frame_count = 0
        
        while True:
            frames = pipeline.wait_for_frames()
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("⚠️ Frame skipped (invalid depth/color data)")
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint16)

            print(f"✅ Captured Frame {frame_count}")

            # Save raw binary files
            depth_image.tofile(f"depth/frame_{frame_count}.raw")
            color_image.tofile(f"color/frame_{frame_count}.raw")

            # Optionally display the images
            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            color_vis = cv2.convertScaleAbs(color_image, alpha=0.03)
            
            cv2.imshow("Depth Stream", depth_vis)
            cv2.imshow("Color Stream", color_vis)
            
            frame_count += 1

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🔄 Recording stopped.")

if __name__ == "__main__":
    main()