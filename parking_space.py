import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO

def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available - falling back to CPU")
        return False
    print(f"CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
    return True

def draw_parking_polygons(frame, spots, occupied_spots):
    overlay = frame.copy()
    for idx, spot in enumerate(spots):
        points = np.array(spot["points"], np.int32)
        color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [points], True, color, 2)
        cv2.putText(frame, str(idx+1), tuple(points[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def main():
    # Verify CUDA availability
    use_cuda = check_cuda()
    
    # Load parking spots
    try:
        with open("bounding_boxes.json") as f:
            parking_data = json.load(f)
        parking_polygons = [np.array(spot["points"], np.int32) for spot in parking_data]
        print(f"Loaded {len(parking_data)} parking spots")
    except Exception as e:
        print(f"Error loading parking spots: {e}")
        return

    # Load model with appropriate device
    try:
        model = YOLO("yolov8n.pt")
        if use_cuda:
            model.to('cuda')
        model.fuse()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Video setup
    cap = cv2.VideoCapture("test1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer
    output_fps = min(20, fps) if fps > 0 else 20
    video_writer = cv2.VideoWriter("output.mp4", 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                 output_fps, 
                                 (original_w, original_h))
    
    # Main processing
    frame_count = 0
    prev_occupied = set()
    vehicle_classes = [2, 5, 7]  # car, bus, truck
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            frame_count += 1
            
            # Process detection every 2nd frame
            if frame_count % 2 == 0:
                # Run detection
                with torch.no_grad():
                    results = model(frame, 
                                  imgsz=640,
                                  verbose=False,
                                  device='cuda' if use_cuda else 'cpu')
                
                occupied_spots = set()
                for r in results:
                    boxes = r.boxes.cpu().numpy()
                    vehicle_indices = np.where(np.isin(boxes.cls, vehicle_classes))[0]
                    
                    for i in vehicle_indices:
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        center = (int((x1+x2)/2), int((y1+y2)/2))  # Ensure integer coordinates
                        
                        for idx, poly in enumerate(parking_polygons):
                            if cv2.pointPolygonTest(poly, center, False) >= 0:
                                occupied_spots.add(idx)
                                break
                
                prev_occupied = occupied_spots
            else:
                occupied_spots = prev_occupied
            
            # Draw UI
            text = f'Occupied: {len(occupied_spots)}/{len(parking_data)}'
            cv2.putText(frame, text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            frame = draw_parking_polygons(frame, parking_data, occupied_spots)
            
            # Display and save
            cv2.imshow('Parking Detection', frame)
            video_writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"Processing completed. Processed {frame_count} frames")

if _name_ == "_main_":
    # Install correct PyTorch if needed
    if not torch.cuda.is_available():
        print("\nWARNING: For best performance, install PyTorch with Jetson support:")
        print("$ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118")
    
    main(
