import cv2

# Reading the video frame by frame
video = cv2.VideoCapture("sample_video_2.mp4")  

# Pre trained car classifier
classifer_file_car = "cars.xml"

# Pre trained pedestrian classifier
classifer_file_pedestrian = "pedestrian.xml"

# Car classifier
cars_classifier = cv2.CascadeClassifier(classifer_file_car)

# pedestrian classifier
pedestrian_classifier = cv2.CascadeClassifier(classifer_file_pedestrian)

# Runs till we close it, and go frame by frame
while True:

    # Read the current frame of the video
    (read_successfull, frame) = video.read()

    # If read was successfull then execute
    if read_successfull:
        black_n_white_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting Coloured frame into black and white
    else:
        break

    # Detects the cars
    cars = cars_classifier.detectMultiScale(black_n_white_frame)

    # Detects the pedestrians
    pedestrians = pedestrian_classifier.detectMultiScale(black_n_white_frame)

    # Drawing the rectangles on cars
    for x,y,width,height in cars:
        cv2.putText(frame,"car",(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)

    # Drawing the rectangles on pedestrians
    for x,y,width,height in pedestrians:
        cv2.putText(frame,"people",(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255,0,0), 2)

    cv2.imshow("Dashcam",frame)

    cv2.waitKey(1)

print("Running")
video.release()
