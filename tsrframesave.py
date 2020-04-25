#
#

# import the necessary packages

from threading import Thread
import cv2
#from datetime import datetime
import time

b_th = 70 #black_threshold 

class TSRframeSave:
    def __init__(self, **kwargs):
        self.stopped = False
        self.frame_list = []
        self.speed = 0
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
        print("#i:start frame save thread")
        while self.stopped == False:
            if len (self.frame_list) > 0:
                fs = self.frame_list.pop (0)
                fn = self.frame_list.pop (0)
                if fs is not None and fn is not None:
                    self.kFot = self.kFot + 1
                    #kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
                    #print("#i:frame save:process frame {} name {}".format (fs.shape, fn))
                    cv2.imwrite (fn, fs)
            #break
            time.sleep(0.0001)
        print ("#i:end save frame thread")
    #
    def stop (self):
        # indicate that the thread should be stopped
        self.stopped = True
        print ("#i:stop save frame thread {}>{}/{}".format (self.kFin, self.kFot, len(self.frame_list)))
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
