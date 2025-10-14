import os
import shutil
import argparse
import cv2
from ultralytics import YOLO

def detect_and_save(model_path, image_path, dataset_base='dataset'):
    
    train_dir = os.path.join(dataset_base, 'train')
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    print(f"✅ Directories '{images_dir}' and '{labels_dir}' are ready.")

    # --- 2. Load the Pre-trained YOLO Model ---
    try:
        model = YOLO(model_path)
        print(f"✅ Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 3. Perform Inference on the Image ---
    print(f"🔍 Performing inference on '{image_path}'...")
    try:
        # verbose=False reduces the amount of console output from the YOLO model
        results = model(image_path, verbose=False)
    except Exception as e:
        print(f"❌ Error during model inference: {e}")
        return

    # The model returns a list of results; we process the first one for the single image.
    result = results[0]

    if len(result.boxes) == 0:
        print("ℹ️ No fire or smoke detected in the image.")
    else:
        print(f"🔥 Detected {len(result.boxes)} objects.")

    # --- 4. Save the Original Image to the Training Set ---
    image_filename = os.path.basename(image_path)
    dest_image_path = os.path.join(images_dir, image_filename)
    shutil.copy(image_path, dest_image_path)
    print(f"✅ Original image saved to: {dest_image_path}")

    # --- 5. Generate and Save the YOLO Label File ---
    # The label filename must match the image filename, but with a .txt extension.
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    dest_label_path = os.path.join(labels_dir, label_filename)

    with open(dest_label_path, 'w') as f:
        # The .boxes attribute contains detection data.
        # .xywhn gives normalized coordinates (x_center, y_center, width, height)
        # which is the exact format required for YOLO labels.
        boxes = result.boxes.xywhn
        # .cls gives the class ID for each detection.
        classes = result.boxes.cls

        for i in range(len(boxes)):
            class_id = int(classes[i])
            x_center, y_center, width, height = boxes[i]

            # Write the formatted line to the label file.
            # We use .6f to format the floats to 6 decimal places.
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    if len(result.boxes) > 0:
        print(f"✅ Labels saved to: {dest_label_path}")

    # --- 6. (Optional) Save Annotated Image for Verification ---
    # The .plot() method draws the bounding boxes on the image.
    annotated_image = result.plot()
    output_annotated_filename = f"annotated_{image_filename}"
    cv2.imwrite(output_annotated_filename, annotated_image)
    print(f"✅ Annotated image saved as '{output_annotated_filename}' for your review.")


if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="YOLO Fire & Smoke Detection and Training Data Generation")
    parser.add_argument('--model', type=str, default='dog.pt', help="Path to the YOLO model file (e.g., dog.pt)")
    parser.add_argument('--image', type=str, default='image1234.png', help="Path to the input image file")

    args = parser.parse_args()

    # Basic validation to ensure the model and image files exist.
    if not os.path.isfile(args.model):
        print(f"❌ Error: Model file not found at '{args.model}'")
    elif not os.path.isfile(args.image):
        print(f"❌ Error: Image file not found at '{args.image}'")
    else:
        detect_and_save(model_path=args.model, image_path=args.image)

