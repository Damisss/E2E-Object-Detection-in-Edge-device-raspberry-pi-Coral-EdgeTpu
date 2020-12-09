#Usage python real_time_detection.py
# Both --input and __confidence are optional
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2
import numpy as np
import argparse
import datetime 

ap = argparse.ArgumentParser()
#parse arguments
ap.add_argument('-i', '--input', help='Path to input video')
ap.add_argument('-c', '--confidence', default=.6, type=float, help='Minimum proba to filter weak detections')
args = vars(ap.parse_args())

class Inference ():
    @staticmethod
    def start ():
        try:
            LABEL = 'Pistol'
            # Load model from disk
            net = DetectionEngine('model/weapons_detector_edgetpu.tflite')
            # set frame height and width to None
            H, W = None, None
             # Use camera if there si no input video
            if not args.get('input', False):
                cap = cv2.VideoCapture(1)
            else:
                cap = cv2.VideoCapture(args['input'])
                assert cap.isOpened(), "Can't open " + args['input']
            
            satratTime = datetime.datetime.now()
            # number of frame
            numFrame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                # stop while loop if is there is no frame available 
                if not ret:
                    break
                #stop while loop if q is pressed
                key = cv2.waitKey(1) & 0xFF 
                if key == ord('q'):
                    break
                
                if H is None or W is None:
                    H, W = frame.shape[:2]
                    
                # data preprocessing
                inFrame = cv2.resize(frame, (300, 300))
                inFrame = Image.fromarray(inFrame)
                
                # perform inference
                result = net.detect_with_image(inFrame, threshold=args['confidence'])
                
                #relative_coord=False
                for r in result:
                    confidence = r.score
                    #grab bounding box
                    bbox = r.bounding_box.flatten() * np.array([W, H, W, H])
                    bbox = bbox.astype('int')
                    xmin, ymin, xmax, ymax = bbox
                    #draw bbox
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    # put label on frame
                    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                    text = '{} {:.2f}%'.format(LABEL, confidence * 100)
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(text), (xmin, y), font, .5,  (0, 0, 255), 2)
                    
                #compute average frame per second
                numFrame += 1
                elaps = (datetime.datetime.now() - satratTime).total_seconds()
                fps = numFrame/elaps
                #put FPS on frame
                cv2.putText(frame, 'Average FPS: {:.2f}'.format(fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5,  (255, 0, 0), 2)     
                #show the frame
                cv2.imshow('Weapons', frame)
                  
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            raise e

if __name__=='__main__':
    Inference.start()
    
