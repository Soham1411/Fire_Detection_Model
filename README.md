# Fire Detection using YOLOv10

## Project Overview

This project is focused on detecting fire in real-time using the YOLOv10 (You Only Look Once, version 10) model. The goal is to provide an efficient and accurate system for fire detection in video feeds or images, which can be integrated into surveillance systems for early fire warnings and prevention.

## Features
- **Real-time fire detection** using the advanced YOLOv10 object detection algorithm.
- **High accuracy and speed**, ideal for real-time surveillance and alert systems.
- **Customizable** model for different environments and fire characteristics.
- **Easy integration** with various cameras and video sources (e.g., CCTV, drone footage).
  
## Model Details

- **Model**: YOLOv10
- **Architecture**: The YOLO family architecture is known for dividing images into grid cells and predicting bounding boxes along with class probabilities directly.
- **Input size**: Configurable (default size: 640x640).
- **Training Dataset**: The model can be trained on custom fire detection datasets or public datasets containing fire and non-fire images.

## Prerequisites

### Hardware
- **GPU**: Recommended for faster training and inference (e.g., NVIDIA GTX/RTX series).
- **CPU**: Intel i5 or higher for basic testing.
  
### Software
- **Python 3.7+**
- **PyTorch or TensorFlow** (based on the YOLOv10 implementation used)
- **OpenCV** for video stream handling and image processing.
- **CUDA** for GPU support (if using GPU for inference).

### Dependencies
The required Python libraries can be installed using:

```bash
pip install -r requirements.txt
```

`requirements.txt` example:

```text
torch>=1.9.0
opencv-python
matplotlib
numpy
Pillow
tqdm
```

## Dataset

- You can use a custom fire dataset or download an open-source fire dataset.
- The images should be labeled for fire and non-fire regions using formats such as YOLO or COCO.
- Example of datasets:
  - [FiresNet](https://github.com/FiresNet)
  - [Fire and Smoke Dataset](https://universe.roboflow.com/-jwzpw/continuous_fire/dataset/6#)

## Training the Model

1. **Prepare the dataset**: Ensure your dataset is labeled properly in the YOLO format. 
2. **Configure model parameters**: Edit the `config.yaml` file to specify parameters such as input size, batch size, learning rate, and number of classes.
3. **Start training**:
4. **Evaluate the model** after training using validation data to ensure accuracy and performance.

## Inference

To run inference on an image or video stream:

```bash
from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source=0, imgsz=640, conf=0.6, show=True)
```

- `--source`: Path to the input image or video file, or specify `0` for a webcam stream.
- `--conf`: Confidence threshold (e.g., 0.5 for 50% confidence).

## Results

After detection, the output will be a video or image with bounding boxes around detected fire instances.

## Performance Evaluation

Evaluate the model using metrics like:
- **Precision**: How accurate are the fire detections.
- **Recall**: How well the model detects all fire instances.
- **FPS (Frames per Second)**: How fast the model can run for real-time detection.

## Customization

- **Model fine-tuning**: Adjust model parameters such as learning rate, batch size, and input resolution in the `config.yaml`.
- **Dataset augmentation**: Use augmentations like flipping, scaling, or brightness adjustment to improve model robustness in various scenarios.

## Future Work

- **Multi-Class Fire Detection**: Extending the model to detect other fire-related objects such as smoke.
- **Fire Intensity Estimation**: Implementing methods to estimate fire intensity or spread rate.
- **Edge Device Deployment**: Optimizing the model for deployment on edge devices like Raspberry Pi or NVIDIA Jetson for portable fire detection systems.

## References

- [YOLOv10 Paper](https://arxiv.org/abs/XXXX.XXXXX) (Add when the paper becomes available)
- [YOLO Repository](https://github.com/ultralytics/yolov10) (Link to YOLOv10 implementation)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
