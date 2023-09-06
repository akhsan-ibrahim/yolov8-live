import cv2 #opencv
import argparse #argumen parser
from ultralytics import YOLO # object detection without boundary box
import supervision as sv # simpler cv library >>> supervision == 0.3.0
import numpy as np

# set area detection
ZONE_POLYGON = np.array([
  [0,0], # top left point
  [1280//4,0], # top right point
  [1250//4,720], # bottom right point
  [0, 720] # bottom left point
])

# Set frame resolution
def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
    "--webcam-resolution", #argument name
    default=[1280,720], #default resolution
    nargs=2, #argument count
    type=int #argument type
  )
  args = parser.parse_args() #parse arguments
  return args

def main():
  #initialize resolution
  args = parse_arguments()
  frame_width, frame_height = args.webcam_resolution

  #initialize camera --> >>>ls -la /dev/ | grep video
  cap = cv2.VideoCapture(1) #arguments = camera source
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

  #model YOLO version 8 large datasets (l)
  model = YOLO("yolov8l.pt")

  #boundary box setup
  box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
  )

  # set polygon zone w supervision
  zone = sv.PolygonZone(
    polygon=ZONE_POLYGON, # set polygon point
    frame_resolution_wh=tuple(args.webcam_resolution)
  )
  # define zone
  zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
  )

  #while cap is True == camera turned on
  while True:
    # frame of video capture
    _, frame = cap.read()

    # result of model execution with frame --> store in list
    result = model(frame)[0] # print to show class object
    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id == 0] # only person detections >>> class_id == 0 (person)
    # print(detections[:5])

    # set label name of object
    labels = [
      f"{model.model.names[class_id]} {confidence:.2f}"
      for _,confidence,class_id,_ # find confidence (accuracy) & class_id (name)
      in detections
    ]

    # set object detection box guidelines
    frame = box_annotator.annotate(
      scene=frame,
      detections=detections,
      labels=labels
    )

    # activate zone w detection
    zone.trigger(detections=detections) # count the objects in the zone >>> current_count
    frame = zone_annotator.annotate(scene=frame)

    # show frame
    cv2.imshow("yolov8 niice", frame) # (window title, frame showed)

    # print(frame.shape)
    # break
    if (cv2.waitKey(30) == 27): # wait 30 ms for press Esc (27 ASCII)
      break

if __name__ == "__main__":
  main()
