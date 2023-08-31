import cv2

def main():
  cap = cv2.VideoCapture(1)

  while True:
    ret, frame = cap.read()
    cv2.imshow("yolov8", frame)

    if (cv2.waitKey(30) == 27): # wait 30 ms for press Esc (27 ASCII)
      break

if __name__ == "__main__":
  main()
