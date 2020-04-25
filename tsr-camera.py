#!/usr/bin/python3
#
#./tsr-camera.py --model=resnet18_e34.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt --video=1
#./tsr-camera.py --model=resnet18_e34.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt --video=1 --display=1
# note: with --video=1, the frame rate drops by approx.10fps

#
import jetson.inference
import jetson.utils
#from jetcam.csi_camera import CSICamera
#from jetcam.usb_camera import USBCamera
#
import argparse
import sys
from datetime import datetime
#
import ctypes
import numpy as np
import cv2
from skimage import exposure
#
# pip3 install pyserial
import serial
#
from tsrvideosave import TSRvideoSave
from tsrframesave import TSRframeSave
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())
# args
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--display", type=int, default=0, help="render stream to DISPLAY")
parser.add_argument("--video", type=int, default=0, help="save stream to ./raw/ storage")
parser.add_argument("--file", type=str, help="filename of the video to process")
parser.add_argument("--nolearn", type=int, default=0, help="filename of the video to process")
parser.add_argument("--with_ui", type=int, default=1, help="show everything on the display")
# parse args
try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
#
try:
    ser = None #serial.Serial ('/dev/ttyTHS1', 9600, timeout=1)
except:
    print("")
    print ('!serial port NOT accessible')
    ser = None
    #sys.exit(0)
#
#ensure directories are created for learning

# open the serial port
if ser is not None and ser.isOpen ():
    print (ser.name + ' is open...')
    #
    st = 'v'
    ser.write (st.encode ())
    st = 'rrrr'
    ser.write (st.encode ())
#    st = '    '
#    ser.write (st.encode ())
#
def write_to_7seg (val):
    if ser is None:
        return
    #never access the serial twice for the same value
    if write_to_7seg._mval == val:
        return
    #
    write_to_7seg._mval = val
    if val == -1:
        st = '    '
    else:
        st = '{:4d}'.format (val)
    #
    ser.write (st.encode ())
write_to_7seg._mval = -2
#
# don't store signs and frames
nolearn = False
if opt.nolearn == 1:
  nolearn = True
if nolearn == False:
    # create directories
    print('check learning storage')
    import os
    if not os.path.exists('../raw'):
        print('creating {}'.format('../raw'))
        os.makedirs('../raw')
        os.makedirs('../raw/_unk')
    # create label dirs
    lines = [line.rstrip('\n') for line in open('./labels.txt')]
    for dirs in lines:
        #print('creating dir for label {}'.format('../raw/' + dirs))
        if not os.path.exists('../raw/' + dirs):
            print('creating {}'.format('../raw/' + dirs))
            os.makedirs('../raw/' + dirs)
    #
# save video stream from camera
save_video = False
if opt.video == 1:
    save_video = True
#use video file?
video_file = False
if opt.file is not None:
    video_file = True
    print("using video file from {:s}".format(opt.file))
    save_video = False
# use CSI/USB camera or gstCamera
csi_camera = True
# show frame from camera on display
show_display = False
if opt.display == 1:
    show_display = True
#--with-ui
show_ui = True
if opt.with_ui == 0:
    show_ui = False
# use ESC key to end the task
ESC = 27
show_fps = True
#
lFps_sec = 0  #current second
lFps_c = 0    #current fps
lFps_k = 0    #current frames
lFps_M = 0    #max fps
lFps_T = 0    #tot
lFps_rS = 0   #running seconds
cFk = 0       #frame count
#
cs_sec = 0
cs_spd = 0
# red circle radius
c_r_min = 8#8-720p #5 #10
c_r_max = 40#40-720p #25 #50

#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
kFot = 0    #count of saved frames
##
# load speed limit signs
sign_white = cv2.imread('./ref/all-speeds-white.jpg')
sign_black = cv2.imread('./ref/all-speeds-black.jpg')
#
def extract_sign(speed, dark):
    # all signs is: 1000x600px
    # idx = row * W + col; row = (idx - col) / W
    sidx = speed // 10 - 1
    scol = sidx % 5
    srow = (sidx - scol) // 5
    #print ('sign index %d, %d' % (srow, scol))
    if dark == True:
        return sign_black[srow*200:srow*200+200, scol*200:scol*200+200]
    return sign_white[srow*200:srow*200+200, scol*200:scol*200+200]
#POV definition
# source 1280x720
# of interest povs
# xx >= 640 - right half
# yy <= 360 - top half
# xx: 320..640..960
# yy: 180..360..540
### for right side classification of traffic posts
"""
#c_xx = (640 + 160) # horizontal mid-point
c_xx = (640 + 160 + 90) # horizontal mid-point
#c_yy = 360 #c_ry # vertical mid-point
c_yy = (360) #c_ry # vertical mid-point
#c_rx = int (360 / 2) # horizontal width (half): x-rx, x+rx
c_rx = int (240 / 2) # horizontal width (half): x-rx, x+rx
#c_ry = int (360 / 2) # vertical width (half):   y-ry, y+ry
c_ry = int ((360-90) / 2) # vertical width (half):   y-ry, y+ry
"""
## use percentage values: 0.0..1.0 - 0..100%
ctrx = 0.5  # POV horizontal center %: left to right
ctry = 0.4  # POV vertical center %: top to bottom
sizx = 0.5  # POV width %
sizy = 0.25 # POV heigth %
#sizy = 0.2
### for mid side/overhead classification of traffic signs
#c_xx = (640 + 160) # horizontal mid-point
c_xx = int(opt.width * ctrx) # horizontal mid-point
#c_yy = 360 #c_ry # vertical mid-point
c_yy = int(opt.height * ctry) #c_ry # vertical mid-point
#c_rx = int (360 / 2) # horizontal width (half): x-rx, x+rx
c_rx = int(opt.width * sizx / 2) # horizontal width (half): x-rx, x+rx
#c_ry = int (360 / 2) # vertical width (half):   y-ry, y+ry
c_ry = int(opt.height * sizy / 2) # vertical width (half):   y-ry, y+ry
#
def store_image(img, desc, kts, kout, confi, sp):
    #global c_xx, c_yy, c_rx, c_ry
    #print("sign at: {:d}x{:d}r{:d}".format(int(sp[0]+c_xx - c_rx), int(sp[1]+c_yy - c_ry), int(sp[2])))
    #
    if nolearn == True:
        return
    if img is None:
        return
    iname = None
    if confi == 0:
        if desc is None:
            # store frame - confi = 0
            iname = "./raw/_unk/img-{}_{}-frame.jpg".format (kts, kout)
        else:
            # store frame - confi = 0
            iname = "./raw/{}/img-{}_{}-frame.jpg".format (desc, kts, kout)
    else:
        if desc is None:
            # save sign for classification training
            iname = "./raw/_unk/img-{}_{}-ori-c{}.jpg".format (kts, kout, confi)
        else:
            # save relevant'ish sign
            iname = "./raw/{}/img-{}_{}-sign-c{}.jpg".format (desc, kts, kout, confi)
    if iname is not None:
        # save image/frame
        global tsr_fs
        tsr_fs.save (iname, img)
#
def img_subrange (img):
    #crop the image area containing the circle
    # subtract the interesting frame
    global c_xx, c_yy, c_rx, c_ry
    return img.copy()[c_yy - c_ry:c_yy + c_ry, c_xx - c_rx:c_xx + c_rx]
#
"""are you using one of the SSD-Mobilenet/Inception models? 
If so, try changing the class names that you want to ignore to 'void' 
(without the '') in the labels file (e.g. <jetson-inference>/data/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt). 
The classes with the name void will then be ignored during detection.
"""
def do_detect (cuda_mem, width, height):
    # detect objects in the image (with overlay)
    return detnet.Detect (cuda_mem, width, height, "box,labels,conf")
#
def do_ai (tsr_img, kTS, kFot, sub_img, dfy, cfy, sign_pos):
    width = tsr_img.shape[0]
    height = tsr_img.shape[1]
    confi = 0
    #cv2.imwrite (iname, final)
    #iname = "./raw/thd-image-{}_{}.png".format (kTS, kFot)
    tsr_imga = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson.utils.cudaFromNumpy (tsr_imga)
    # do object detection
    if dfy == True:
        detections = detnet.Detect (cuda_mem, width, height, "box,labels,conf")
        if len (detections) > 0:
            print("detected {:d} objects in image".format(len(detections)))
            iname = "./raw/_objs/img-{}_{}-cuda-o{}.jpg".format (kTS, kFot, 0)
            #jetson.utils.saveImageRGBA (iname, cuda_mem, width, height)
            #for detection in detections:
            #    print(detection)
            # print out timing info
            #net.PrintProfilerTimes()
            #print (cuda_mem)
        #
    # do classification
    if cfy == True:
        class_idx, confidence = imgnet.Classify (cuda_mem, width, height)
        confi = int (confidence * 1000)
        if class_idx >= 0:# and confi > 800: # or confidence * 100) > 60:
            # find the object description
            class_desc = imgnet.GetClassDesc (class_idx)
            #print ("found sign {:d} {:s} on {:d}".format (confi, class_desc, kFot))
            # save ROI
            if confi > 800: # over 99% confidence
                #save relevant'ish sign
                store_image(tsr_img, class_desc, kTS, kFot, confi, sign_pos)
                # save originating frame, for reference
                if sub_img is not None:
                    # store frame - confi = 0
                    store_image(sub_img, class_desc, kTS, kFot, 0, sign_pos)
            else:
                #save sign for classification training - desc None
                store_image(tsr_img, None, kTS, kFot, confi, sign_pos)
                # save originating frame, for reference
                if sub_img is not None:
                    # store frame - confi = 0
                    store_image(sub_img, None, kTS, kFot, 0, sign_pos)
            #
            # overlay the result on the image
            if confi > 994: # over 99.4% confidence
                #print ("found sign {} {:s} fps {}".format (confi, class_desc, net.GetNetworkFPS ()))
                # update the indicator
                global cs_spd
                if class_idx == 0:#kph20
                    cs_spd = 20
                if class_idx == 1:#kph30
                    cs_spd = 30
                if class_idx == 2:#kph50
                    cs_spd = 50
                if class_idx == 3:#kph60
                    cs_spd = 60
                if class_idx == 4:#kph70
                    cs_spd = 70
                if class_idx == 5:#kph80
                    cs_spd = 80
                if class_idx == 6:#kph100
                    cs_spd = 100
                if class_idx == 7:#kph120
                    cs_spd = 120
                #
                global cs_sec, lFps_rS
                cs_sec = lFps_rS
                print ("%dkph sign %d %s on %d" % (cs_spd, confi, class_desc, kFot))
                # store captured sign for display
                global ui_find, ui_sign
                ui_find = tsr_img
                # is this a new speed limit?
                if int(do_ai._lspd) != int(cs_spd):
                    do_ai._lspd = int(cs_spd)
                    print ("NEW %dkph sign %d %s on %d" % (cs_spd, confi, class_desc, kFot))
            #
    return confi
#
do_ai._lspd = -1
##
ref_frames = 90
def check_red_circles (image, kTS):
    sub_img = img_subrange (image)
    """
    #print('subimg shape {}'.format(sub_img))
    sub_img = exposure.equalize_adapthist (sub_img, clip_limit=0.1)
    #print('subimg equ shape {}'.format(sub_img))
    sub_img *= 255
    sub_img = sub_img.astype(np.uint8)
    #sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
    #print('subimg cnv shape {}'.format(sub_img))
    """
    # reset frames count on 0
    if check_red_circles._lcf == 0:
        check_red_circles._lcf = ref_frames
    # if we have identified a sign, skip the next ref_frames frames
    if nolearn == True and check_red_circles._lcf < ref_frames:
        check_red_circles._lcf = check_red_circles._lcf - 1
        return sub_img
    # process subimage
    #blurred = cv2.blur (sub_img, (5, 5))
    blurred = cv2.GaussianBlur (sub_img, (5, 5), 0)
    #canny = cv2.Canny (blurred, 50, 150)
    hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor (blurred, cv2.COLOR_RGB2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    # lower mask (0-10)
    #mask0 = cv2.inRange (hsv, lower_white, upper_white)
    mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
    # upper mask (170-180)
    mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
    # join my masks
    cmask = mask0 + mask1
    #cmask = mask1
    #
    result = sub_img
    #result = cmask
    #cmask = cv2.erode (cmask, None, iterations=2)
    #cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 
                200, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #            60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      #kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #iname = "/mnt/raw/img-{}-frame.png".format (kTS)
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2]) #autocrop the 'red' circle
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        if c_x > c_r and c_y > c_r: #and c_r > 6
            tsr_img = sub_img.copy()
            c_r = c_r + 10 # store a bigger sign frame for better identification
            sx = c_x - c_r
            sy = c_y - c_r
            sw = c_r * 2
            sh = c_r * 2
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            tsr_img = tsr_img[sy:sy + sh, sx:sx + sw]
            #
            global kFot
            kFot = kFot + 1
            # skip frames if we found good sign, over 994
            if do_ai (tsr_img, kTS, kFot, image, False, True, i) > 994:
                check_red_circles._lcf = ref_frames - 1
            #
        if show_display == True:
            cv2.circle (result, (c_x, c_y), c_r, (0,0,255), 2)
        #.for
    #.if circles is not None:
    #
    #return tsr_img
    #return canny
    return result
#
check_red_circles._lcf = ref_frames
##
def update_pov (key):
    #
    global ctrx, ctry, sizx, sizy, c_xx, c_yy, c_rx, c_ry
    #
    m_h = 0.0
    m_v = 0.0
    s_h = 0.0
    s_v = 0.0
    new_pov = False
    # move POV
    povg = 0.01
    if key == ord('a'):
        #left
        if (ctrx - sizx / 2 - povg) > 0.0:
            m_h = -povg
            new_pov = True
    if key == ord('d'):
        #right
        if (ctrx + sizx / 2 + povg) < 1.0:
            m_h = povg
            new_pov = True
    if key == ord('w'):
        #up
        if (ctry - sizy / 2 - povg) > 0.0:
            m_v = -povg
            new_pov = True
    if key == ord('s'):
        #down
        if (ctry + sizy / 2 + povg) < 1.0:
            m_v = povg
            new_pov = True
    # size POV
    if key == ord('A'):
        #horiz reduce
        if (ctrx - sizx / 2 - povg) > 0.0:
            s_h = -povg
            new_pov = True
    if key == ord('D'):
        #horiz enlarge
        if (ctrx + sizx / 2 + povg) < 1.0:
            s_h = povg
            new_pov = True
    if key == ord('W'):
        #vert reduce
        if (ctry - sizy / 2 - povg) > 0.0:
            s_v = -povg
            new_pov = True
    if key == ord('S'):
        #vert enlarge
        if (ctry + sizy / 2 + povg) < 1.0:
            s_v = povg
            new_pov = True
    #compute new POV
    if new_pov == True:
        ctrx = ctrx + m_h
        ctry = ctry + m_v
        sizx = sizx + s_h
        sizy = sizy + s_v
        c_xx = int(opt.width * ctrx) # horizontal mid-point
        c_yy = int(opt.height * ctry) #c_ry # vertical mid-point
        c_rx = int(opt.width * sizx / 2) # horizontal width (half): x-rx, x+rx
        c_ry = int(opt.height * sizy / 2) # vertical width (half):   y-ry, y+ry
        print("POV {:.2f}x{:.2f}-{:.2f}x{:.2f}/{:d}x{:d}-{:d}x{:d}".format(ctrx, ctry, sizx, sizy, c_xx, c_yy, c_rx * 2, c_ry * 2))
    #
##
#
# get screen size on Linux
# import struct, os, sys, fcntl
# fbfd = os.open('/dev/fb0',os.O_RDWR)
# struct.unpack("8I12I16I4I", fcntl.ioctl(fbfd, 0x4600, " "*160))
# >>> struct.unpack("8I12I16I4I", fcntl.ioctl(fbfd, 0x4600, " "*160))
# (720, 480, 720, 480, 0, 0, 32, 4294967168, 16, 8, 4189770400, 8, 8, 4294967232, 0, 8, 140676952, 24, 8, 4294967232, 1, 0, 4278129984, 4294967232, 4157961728, 37037, 60, 16, 30, 9, 62, 6, 0, 6291456, 4189770832, 0, 0, 0, 0, 0)
# >>> struct.unpack("8I12I16I4I", fcntl.ioctl(fbfd, 0x4600, " "*160))[0]
# 720
# >>> struct.unpack("8I12I16I4I", fcntl.ioctl(fbfd, 0x4600, " "*160))[1]
# 480
try:
    import struct, os, fcntl
    fbfd = os.open('/dev/fb0', os.O_RDWR)
    fbinfo = fcntl.ioctl(fbfd, 0x4600, " "*160)
    _SW = int(struct.unpack("8I12I16I4I", fbinfo)[0])
    # 720
    _SH = int(struct.unpack("8I12I16I4I", fbinfo)[1])
    # 480
    os.close(fbfd)
except:
    _SW = 720
    _SH = 480
#
_SW = 480
_SH = 320
print ('display size: %dx%d' % (_SW, _SH))
_hSH = _SH//2
_hSW = _SW//2
##
ui_sign = None
ui_find = None
ui_shot = None
#fb = np.memmap('/dev/fb2', dtype='uint8',mode='r+', shape=(320, 480, 4))
fb = np.memmap('/dev/fb2', dtype='uint8',mode='r+', shape=(320, 480, 2))
#
### - from https://github.com/adafruit/Adafruit_CircuitPython_RGB_Display
def color565(r, g=0, b=0):
    """Convert red, green and blue values (0-255) into a 16-bit 565 encoding.  As
    a convenience this is also available in the parent adafruit_rgb_display
    package namespace."""
    try:
        r, g, b = r  # see if the first var is a tuple/list
    except TypeError:
        pass
    return (r & 0xf8) << 8 | (g & 0xfc) << 3 | b >> 3

def image_to_data(data):
    """Generator function to convert a PIL image to 16-bit 565 RGB bytes."""
    #NumPy is much faster at doing this. NumPy code provided by:
    #Keith (https://www.blogger.com/profile/02555547344016007163)
    #data = np.array(data.convert('RGB')).astype('uint16')
    #npframe = np.zeros((320, 480, 3), dtype=np.uint16) #+ (255, 255, 255) # background color (0, 0, 0)
    #npframe[:, :] = data
    #data = npframe
    #data = cv2.cvtColor (data, cv2.COLOR_BGR2RGB)
    #print('data b4 %x %x %x' % (data[-1:,-1:,0][0][0], data[-1:,-1:,1][0][0], data[-1:,-1:,2][0][0]))
    #color = ((npframe[:, :, 2] & 0xF8) << 8) | ((npframe[:, :, 1] & 0xFC) << 3) | (npframe[:, :, 0] >> 3)
    color = (((data[:, :, 0] >> 3) & 0x1F) << 11) | (((data[:, :, 2] >> 2) & 0x3F) << 5) | ((data[:, :, 1] >> 3) & 0x1F)
    ## data b4 [[152]] [[178]] [[155]]
    ## 152 0x13 0x9800 | 0x2c 0x580 | 0x13 = 0x9D93
    ## data after [134] [102]
    ## data b4: 9c b2 9a = 0x51200 | 0x128 | 0x19 = 0x51339
    ## 
    ## data after 26 46 = 0x13 0x39
    #print('data after %x %x' % (color[-1:,0][0], color[-1:,0][0]))
    #print('color {}'.format(color.shape))
    #arru16 = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.uint16)
    arru16 = np.dstack(((color >> 8) & 0xFF, color & 0xFF)) #(color >> 8) & 0xFF, color & 0xFF
    return arru16
## 
def show_ui():
    if show_ui == False:
        return
    # draw black bars
    wframe = np.zeros((_SH, _SW, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
    ##wframe = np.zeros((480, 640, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
    global ui_sign, ui_find, ui_shot
    if ui_sign is not None:
        if ui_sign.shape[0] != _hSH or ui_sign.shape[1] != _hSW:
            ui_sign = cv2.resize(ui_sign, (_hSW, _hSH), interpolation = cv2.INTER_AREA)
            #print ("resize sign")
        wframe[0:_hSH, 0:_hSW] = ui_sign
        #wframe[0:0 + result.shape[0], 0:0 + result.shape[1]] = result
    if ui_find is not None:
        if ui_find.shape[0] != _hSH or ui_find.shape[1] != _hSW:
            ui_find = cv2.resize(ui_find, (_hSW, _hSH), interpolation = cv2.INTER_AREA)
            #print ("resize find")
        wframe[0:_hSH, _hSW:_SW] = ui_find
        #wframe[0:0 + result.shape[0], 0:0 + result.shape[1]] = result
    if ui_shot is not None:
        if ui_shot.shape[0] != _hSH or ui_shot.shape[1] != _SW:
            ui_shot = cv2.resize(ui_shot, (_SW, _hSH), interpolation = cv2.INTER_AREA)
            #print ("resize shot")
        #ui_shot = cv2.cvtColor (ui_shot, cv2.COLOR_BGR2RGB)
        wframe[_hSH:_SH, 0:_SW] = ui_shot
        #wframe[0:0 + result.shape[0], 0:0 + result.shape[1]] = result
    cv2.imshow ('result', wframe)
    #cv2.imwrite ('/dev/fb2', wframe)
    # copy to framebuffer
    global fb
    #fb[0:320, 0:480] = np.copy(wframe)
    wf565 = image_to_data(wframe)
    #print('wf565 {}'.format(wf565.shape))
    fb[0:320, 0:480] = wf565 #np.copy (wf565)
    #
##
def show_sign(sign):
    # draw black bars
    wframe = np.zeros((_SH, _SW, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
    ##wframe = np.zeros((480, 640, 3), dtype=np.uint8) #+ (255, 255, 255) # background color (0, 0, 0)
    if sign is not None:
        sign = cv2.resize(sign, (_SH, _SH), interpolation = cv2.INTER_AREA)
        sy, sx = (wframe.shape[0] - sign.shape[0])//2, (wframe.shape[1] - sign.shape[1])//2
        wframe[sy:sy + sign.shape[0], sx:sx + sign.shape[1]] = sign
        #wframe[0:0 + result.shape[0], 0:0 + result.shape[1]] = result
    cv2.imshow ('result', wframe)
##
def prep_camera (video_file, csi_camera, opt_camera, wbmode):
    if video_file is not None:
        camera = cv2.VideoCapture (video_file)
    else:
        # camera setup
        #display = jetson.utils.glDisplay ()
        if csi_camera == False:
            camera = jetson.utils.gstCamera (opt.width, opt.height, opt.camera)
            img, width, height = camera.CaptureRGBA (zeroCopy = True)
            jetson.utils.cudaDeviceSynchronize ()
            jetson.utils.saveImageRGBA ("camera.jpg", img, width, height)
            # create a numpy ndarray that references the CUDA memory
            # it won't be copied, but uses the same memory underneath
            aimg = jetson.utils.cudaToNumpy (img, width, height, 4)
            #print (aimg)
            #aimg1 = aimg.astype (numpy.uint8)
            #print ("img shape {}".format (aimg1.shape))
            aimg1 = cv2.cvtColor (aimg, cv2.COLOR_RGBA2BGR)
            #print (aimg1)
            cv2.imwrite ("array.jpg", aimg1)
            # save as image
            #exit()
        else:
            #camera = CSICamera (width=opt.width, height=opt.height)
            # CSI
            # or
            #camera = USBCamera (width=opt.width, height=opt.height, capture_device=3)
            # camstr = 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, , format=(string)BGR ! appsink wait-on-eos=false drop=true'.format(
            #        1, opt.width, opt.height, 30)
            # image_resized = cv2.resize(image,(int(self.width),int(self.height)))
            # > gst-inspect-1.0 nvarguscamerasrc
            #   wbmode              : White balance affects the color temperature of the photo
            #                flags: readable, writable
            #                Enum "GstNvArgusCamWBMode" Default: 1, "auto"
            #                   (0): off              - GST_NVCAM_WB_MODE_OFF
            #                   (1): auto             - GST_NVCAM_WB_MODE_AUTO
            #                   (2): incandescent     - GST_NVCAM_WB_MODE_INCANDESCENT
            #                   (3): fluorescent      - GST_NVCAM_WB_MODE_FLUORESCENT
            #                   (4): warm-fluorescent - GST_NVCAM_WB_MODE_WARM_FLUORESCENT
            #                   (5): daylight         - GST_NVCAM_WB_MODE_DAYLIGHT
            #                   (6): cloudy-daylight  - GST_NVCAM_WB_MODE_CLOUDY_DAYLIGHT
            #                   (7): twilight         - GST_NVCAM_WB_MODE_TWILIGHT
            #                   (8): shade            - GST_NVCAM_WB_MODE_SHADE
            #                   (9): manual           - GST_NVCAM_WB_MODE_MANUAL
            # Please copy camera_overrides.isp to /var/nvidia/nvcam/settings and do below commands to install the ISP file.
            # sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
            # sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
            # --
            """
            GST_ARGUS: Available Sensor modes :
            GST_ARGUS: 3264 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
            GST_ARGUS: 3264 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
            GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
            GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
            GST_ARGUS: 1280 x 720 FR = 120.000005 fps Duration = 8333333 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
            GST_ARGUS: Running with following settings:
            Camera index = 0 
            Camera mode  = 4 
            Output Stream W = 1280 H = 720 
            seconds to Run    = 0 
            Frame Rate = 120.000005 
            GST_ARGUS: Setup Complete, Starting captures for 0 seconds
            --
            for 30fps we can use max: 1000/30 = 33ms exposure or 33000000ns: use with exposuretimerange="33000000 33000000"
            """
            #
            if int(opt_camera) > 0:
                camstr = 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, , format=(string)BGR ! appsink wait-on-eos=false drop=true'.format(
                    usb_camera, opt.width, opt.height, 30)
            else: #wbmode=0 awblock=true gainrange="1 1" ispdigitalgainrange="1 1" exposuretimerange="5000000 5000000" aelock=true
                #camstr = 'nvarguscamerasrc wbmode=0 awblock=true gainrange="1 1" ispdigitalgainrange="1 1" exposuretimerange="20000000 20000000" aelock=true ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink wait-on-eos=false drop=true  max-buffers=1' % (
                #camstr = 'nvarguscamerasrc sensor-id=0 wbmode=0 awblock=true gainrange="1 1" ispdigitalgainrange="1 1" exposuretimerange="80000000 80000000" aelock=true ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink wait-on-eos=false drop=true max-buffers=1' % (
                #camstr = 'nvarguscamerasrc sensor-id=0 wbmode=0 awblock=true gainrange="%d %d" ispdigitalgainrange="1 1" exposuretimerange="20000000 20000000" aelock=true ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink wait-on-eos=false drop=true max-buffers=1' % (
                camstr = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink wait-on-eos=false drop=true max-buffers=1' % (
                    #wbmode, wbmode, opt.width, opt.height, 30, opt.width, opt.height)
                    opt.width, opt.height, 30, opt.width, opt.height)
                print ('wbmode %d: %s' % (wbmode, camstr))
            #
            camera = cv2.VideoCapture (camstr, cv2.CAP_GSTREAMER)
        #
    return camera
##
##
import RPi.GPIO as GPIO
#
def gpio_buttons ():
    global pin18btn
    cpin_val = GPIO.input(pin18btn)
    if cpin_val != gpio_buttons.pin18val:
        gpio_buttons.pin18val = cpin_val
        print('btn18 flip')
        if gpio_buttons.pin18flp == 0 and cpin_val == 1:
            gpio_buttons.pin18flp = 1
    return
gpio_buttons.pin18val = 1
gpio_buttons.pin18flp = 0
##
# camera analog gain
camgain_val = 0
##
# -- main block
#
cv2.namedWindow('result', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#show_sign(sign010)
ui_sign = extract_sign(30, False)
ui_find = extract_sign(30, False)
ui_shot = extract_sign(30, False)
show_ui()
#show_sign(None)
# wait for the UI to get rendered
cv2.waitKey(100)
## -
## setup GPIO buttons
# check pins at: /proc/device-tree/pinmux@700008d4/common/<pin-name>/nvidia,function
#
pin18btn = 32
GPIO.setmode (GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup (pin18btn, GPIO.IN)  # button pin set as input
# use a 4.7k pull-up to 3v3 and button to GND
#GPIO.add_event_detect (but_pin, GPIO.FALLING, callback=wbmode_switch, bouncetime=10)
## -
camera = prep_camera(opt.file, csi_camera, opt.camera, 1)
#
# prep video storing
if video_file == False and save_video == True:
    vname = "./raw/video-{}p-{}.avi".format (opt.height, datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
    #fourcc = cv2.VideoWriter_fourcc(*'X264')  # cv2.VideoWriter_fourcc() does not exist
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc() does not exist
    video_writer = cv2.VideoWriter (vname, fourcc, 30, (opt.width, opt.height))
    tsr_vs = TSRvideoSave ()
    tsr_vs.start (video_writer)
#
# load the recognition network
imgnet = jetson.inference.imageNet (opt.network, sys.argv)
# load the object detection network
#detnet = jetson.inference.detectNet ("ssd-mobilenet-v2", threshold=0.5)
# create the camera and display
#font = jetson.utils.cudaFont ()
# process frames until user exits
tsr_fs = TSRframeSave ()
tsr_fs.start ()
#windowWidth = cv2.getWindowImageRect("result")[2]
#windowHeight = cv2.getWindowImageRect("result")[3]
#print("UI:screen size {}x{}".format(windowWidth, windowHeight))
#
#
aFps = 0
#while display.IsOpen():
while True:
    try:
        ## test gpio buttons
        gpio_buttons()
        #
        if gpio_buttons.pin18flp == 1:
            gpio_buttons.pin18flp = 0
            camgain_val = camgain_val + 1
            if camgain_val > 10:
                camgain_val = 0
            #
            camera.release ()
            camera = prep_camera (opt.file, csi_camera, opt.camera, camgain_val)
        #   #
        # capture the image
        if video_file == True:
            ret, aimg1 = camera.read()
            if ret == False:
                break
        else:
            if csi_camera == False:
                img, width, height = camera.CaptureRGBA (zeroCopy = True)
                jetson.utils.cudaDeviceSynchronize ()
                # create a numpy ndarray that references the CUDA memory
                # it won't be copied, but uses the same memory underneath
                aimg = jetson.utils.cudaToNumpy (img, width, height, 4)
                #print ("img shape {}".format (aimg1.shape))
                aimg1 = cv2.cvtColor (aimg.astype (np.uint8), cv2.COLOR_RGBA2BGR)
            else:
                ret, aimg = camera.read()
                if ret == False:
                    break
                #aimg1 = cv2.flip (aimg, -1)
                aimg1 = aimg
        #
        cFk = cFk + 1
        #
        if save_video == True:
            # add frame to video
            #video_writer.write (aimg1)
            tsr_vs.save (aimg1)
        # do filter and classification
        kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        # on 10watt nvpmodel -m0 && jetson_clocks:
        # img_subrange 28fps
        # check_red_circles 28fps
        # subrange + classify 38fps
        # red detect + classify: approx.30fps-60fps
        ###
        # subrange + classify 1 or 91 17fps
        #
        # on 5watt nvpmodel -m1 && jetson_clocks
        # img_subrange 28fps
        # check_red_circles NO AI 22fps
        # check_red_circles + classify +/- frame save 14fps 
        #
        result = check_red_circles (aimg1, kTS) #img_subrange (aimg1) #check_red_circles (aimg1, kTS) #img_subrange (aimg1) #check_red_circles (aimg1, kTS)
        #result = aimg1
        #
        #fps computation
        cFps_sec = datetime.now().second
        lFps_k = lFps_k + 1
        if lFps_sec != cFps_sec:
            # we advance 1 sec
            lFps_c = lFps_k - 1
            lFps_k = 0
            lFps_rS = lFps_rS + 1 #increment seconds - we assume we get here every second
            # update ui once every second
            #turn off sign display
            if cs_sec > 0 and cs_spd > 0:
                # flash the sign indicator 
                if lFps_rS % 2 == 0:
                    write_to_7seg (-1)
                    ui_sign = extract_sign(cs_spd, False)
                else:
                    write_to_7seg (cs_spd)
                    ui_sign = extract_sign(cs_spd, True)
                # stop flashing and show the last speed
                if cs_sec + 5 < lFps_rS:
                    cs_sec = 0
                    write_to_7seg (cs_spd)
                    do_ai._lspd = -1
                    ui_sign = extract_sign(cs_spd, True)
            #   #
            if show_display == True:
                cv2.imshow ('result', result)
            else:
                cfpst = "A{}/C{}".format (aFps, lFps_c)
                cv2.putText (result, cfpst, (result.shape[1]-280, result.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, 0)
                ui_shot = result
                show_ui()
        # max fps?
        if lFps_M < lFps_k:
            lFps_M = lFps_k
        lFps_sec = cFps_sec
        #
        if show_fps == True:
            if False and show_display == True:
                cv2.putText (result, cfpst, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, 0)
            else:
                if (cFk % 50) == 0:
                    #print ("#i:max fps {}".format (lFps_M))
                    if lFps_rS > 0:
                        aFps = int (cFk / lFps_rS)
                    else:
                        aFps = 1
                    cfpst = "FPS M{} / A{} / C{} : T{}f P{}f {}s".format (lFps_M, aFps, lFps_c, cFk, kFot, lFps_rS)
                    print (cfpst)
        #   
        #if show_display == True:
        #    # only display every 5 frames
        #    if cFk % 5 == 0:
        #        cv2.imshow ('result', result)
        # process input
        key = cv2.waitKey (1)
        #quit
        if key == ord('Q'):
            break
        if key == ord('q'):
            break
        if key == ESC:
            break
        # update POV
        update_pov(key)
        #
    #
    except KeyboardInterrupt:
        break
#
#if video_file == True:
camera.release()
#
GPIO.cleanup()  # cleanup all GPIOs
#
if ser is not None:
    st = 'oooo'.format ()
    ser.write (st.encode ())
#
tsr_fs.stop()
print ("#w:dropping {} frames".format (tsr_fs.count()))
#
if save_video == True:
    tsr_vs.stop()
    print ("#w:dropping {} video frames".format (tsr_vs.count()))
    video_writer.release()
#
if show_display == True:
    cv2.destroyAllWindows()
#
