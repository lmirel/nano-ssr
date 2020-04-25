#
#

# import the necessary packages

from threading import Thread
import cv2
#from datetime import datetime
import time

class TSRui:
    def __init__(self, **kwargs):
        self.stopped = False
        self.frame_list = []
        self.kFin = 0
        self.kFot = 0
    #
    def start(self):
        # start the thread to read frames from the video stream
        t = Thread (target=self.update, args= ())
        t.daemon = True
        t.start ()
        return self
    #
    def update(self):
        print("#i:start UI thread")
        while self.stopped == False:
            if len (self.frame_list) > 0:
                fs = self.frame_list.pop (0)
                fn = self.frame_list.pop (0)
                if fs is not None and fn is not None:
                    self.kFot = self.kFot + 1
                    #kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
                    #print("#i:frame save:process frame {} name {}".format (fs.shape, fn))
                    #cv2.imwrite (fn, fs)
                    #resize result, just in case..
                    #result = cv2.resize(result, (int(result.shape[0]/2), int(result.shape[1]/2)), interpolation = cv2.INTER_AREA)
                    cv2.namedWindow('result', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # draw black bars
                    """
                    wframe = np.zeros((720, 1280, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
                    ##result = cv2.resize(result, (int(result.shape[1]/2), int(result.shape[0]/2)), interpolation = cv2.INTER_AREA)
                    ##wframe = np.zeros((480, 640, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
                    sy, sx = (wframe.shape[0] - result.shape[0])//2, (wframe.shape[1] - result.shape[1])//2
                    wframe[sy:sy + result.shape[0], sx:sx + result.shape[1]] = result
                    #wframe[0:0 + result.shape[0], 0:0 + result.shape[1]] = result
                    cv2.imshow ('result', wframe)
                    """
                    cv2.imshow ('result', fs)
                    print("#i:frame save:process frame {} name {}".format (fs.shape, fn))
            #break
            time.sleep(0.0001)
        print ("#i:end UI thread")
    #
    def stop (self):
        # indicate that the thread should be stopped
        self.stopped = True
        print ("#i:stop UI thread {}>{}/{}".format (self.kFin, self.kFot, len(self.frame_list)))
    #
    def count (self):
        # indicate that the thread should be stopped
        return len(self.frame_list)
    #
    def save (self, frmfn, frm):
        # return the frame most recently read
        self.frame_list.append (frm)
        self.frame_list.append (frmfn)
        self.kFin = self.kFin + 1
        #kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFin)
        #print ("#i:frame save stored frame {}".format (frmfn))  # print ocr text from image
#
