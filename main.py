import cv2 #opencv
import argparse #argumen parser
from ultralytics import YOLO # object detection without boundary box
import supervision as sv # simpler cv library >>> supervision == 0.3.0
import numpy as np

# set area detection
ZONE_POLYGON = np.array([
  [0,0], # top left point
  [0.5,0], # top right point
  [0.5,1], # bottom right point
  [0, 1] # bottom left point
])

# Set frame resolution
def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
    "--webcam-resolution", #argument name
    default=[640,480], #default resolution
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

  # set zone relative w frame 
  zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)

  # set polygon zone w supervision
  zone = sv.PolygonZone(
    polygon=zone_polygon, # set polygon point
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
    result = model(frame, agnostic_nms=True)[0] # print to show class object // agnostic = prevent double detection by get rid an object when 2 object oscillating
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
