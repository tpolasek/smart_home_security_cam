#!/usr/bin/python3
import cv2
import time
import argparse
import base64
import smtplib
import sys
from datetime import datetime
from collections import deque

def draw_max_counter_point(frame, contours):
    ca = {}
    for c in contours:
        ca[cv2.contourArea(c)] = c

    if len(ca.keys()) > 0:
        c = ca[max(ca.keys())]
        moment = cv2.moments(c)
        moment_x = int(moment["m10"] / (moment["m00"] + 0.001))
        moment_y = int(moment["m01"] / (moment["m00"] + 0.001))
        cv2.circle(frame, (moment_x, moment_y), 3, (0, 0, 255), -1)


def contour_trigger(contours):
    ca = {}
    for c in contours:
        ca[cv2.contourArea(c)] = c

    sorted_contour_area_desc = sorted(list(ca.keys()), key=float, reverse=True)
    contour_top_area_sum = sum(sorted_contour_area_desc[0:max(2,len(sorted_contour_area_desc))])

    if 2000 < contour_top_area_sum < 100000:
        return True

    return False


def send_sms(file_path):
    data = """MIME-Version: 1.0
    Date: %s
    Subject: 
    From: myemail@myemail.ca
    To: %s
    Content-Type: multipart/mixed; boundary="0000000000004b42cd057da9db17"

    --0000000000004b42cd057da9db17
    Content-Type: image/jpeg; name="test.jpg"
    Content-Disposition: attachment; filename="test.jpg"
    Content-Transfer-Encoding: base64
    X-Attachment-Id: 167d98fe8b5bb9430851
    Content-ID: <167d98fe8b5bb9430851>

    %s
    --0000000000004b42cd057da9db17--"""

    fromaddr = "thomas@polasek.ca"
    toaddrs = ["######@mms.telusmobility.com"]

    img = base64.b64encode((open(file_path, "rb").read()))

    server = smtplib.SMTP('smtp.telus.net')
    #server.sendmail(fromaddr, toaddrs, data % (time.strftime("%a, %d %b %Y %H:%M:%S %z"), ','.join(toaddrs), img))
    message = """\
    Subject: Hi there

    This message is sent from Python."""
    server.sendmail(fromaddr,toaddrs[0],message)
    server.quit()


class Recorder:
    def __init__(self, max_frames=13*20):
        self.frames = deque()
        self.limit = True
        self.max_frames = max_frames

    def update(self, frame):
        self.frames.appendleft(frame)
        if len(self.frames) > self.max_frames and self.limit:
            self.frames.pop()

    def flush(self):
        print("RECORD DUMP")
        size =self.frames[0].shape[1],self.frames[0].shape[0]
        output = cv2.VideoWriter("%s.mp4" % datetime.now(), cv2.VideoWriter_fourcc(*'mp4v'), 13, size )
        while True:
            if len(self.frames) == 0:
                break
            output.write(self.frames.pop())
        output.release()

class TriggerWatcher:
    def __init__(self, trigger_threshold, frame_count_max):
        self.frame_count_max = frame_count_max
        self.frame_count = 0
        self.trigger_threshold = trigger_threshold
        self.trigger_count = 0
        self.last_trigger_count = 0
        self.frame = None
        self.trigger_frame = None
        self.alert_time = time.time() + 1000000
        self.alert_delay_time = 5
        self.cooldown_time = 0
        self.cooldown_time_delay_sec = 10
        self.recorder = Recorder()

    def update(self, frame, trigger):
        self.frame = frame
        self.frame_count += 1

        self.recorder.update(frame)

        if trigger:
            self.trigger_count += 1

        if self.frame_count >= self.frame_count_max:
            self.frame_count = 0
            self.last_trigger_count = self.trigger_count
            self.trigger_count = 0

            if self.last_trigger_count > self.trigger_threshold:
                if time.time() < self.cooldown_time:
                    print("ALERT with cooldown at @ %s" % datetime.now())
                else:
                    print("ALERT @ %s with trigger_count %d" % (datetime.now(), self.last_trigger_count))
                    self.alert_time = time.time() + self.alert_delay_time
                    if self.trigger_frame is None:
                        self.trigger_frame = self.frame

        if time.time() > self.alert_time:
            self.alert_time = time.time() + 100000
            self.cooldown_time = time.time() + self.cooldown_time_delay_sec
            self.alert()

    def alert(self):
        image_file_path = "%s.png" % datetime.now()
        cv2.imwrite(image_file_path, self.trigger_frame)
        #send_sms(image_file_path)
        self.trigger_frame = None
        self.recorder.flush()

    def get_last_frame_set_count(self):
        return self.last_trigger_count

def main(args):
    tw = TriggerWatcher(35, 40)

    log_file = open('badboi.log', 'a')

    bitmask_pic = cv2.imread('bitmask.png')

    loop_count = 0
    last_time = 0
    vcap = cv2.VideoCapture(args.input)
    last_frame = None
    last_frame_d = None

    while True:
        ret, frame = vcap.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

        # initial frame should be black for testing triggers
        if loop_count == 0:
            frame = cv2.imread('black.png')

        if args.bitmask:
            gray = cv2.bitwise_and(frame, bitmask_pic)
        else:
            gray = frame
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)

        if last_frame is None:
            last_frame = gray

        # compute delta frame
        frame_d = cv2.absdiff(last_frame, gray)
        if last_frame_d is None:
            last_frame_d = frame_d
        frame_d = cv2.addWeighted(frame_d, args.motion_decay, last_frame_d, 1 - args.motion_decay, 0)
        last_frame_d = frame_d

        # dilate and apply contour detection
        thresh = cv2.threshold(frame_d, args.binary_thresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=10)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        tw.update(frame, contour_trigger(contours))

        # cv2.imshow('VIDEO', thresh)
        #cv2.imshow('gray',gray)
        cv2.imshow('ori', frame)
        cv2.waitKey(1)

        #pix_delta = cv2.countNonZero(thresh)
        last_frame = cv2.addWeighted(last_frame, args.bg_decay, gray, 1 - args.bg_decay, 0)

        now_time = time.time()
        #print(1.0/(now_time - last_time))
        last_time = now_time
        loop_count += 1
        time.sleep(0.05)

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input video file or stream', default="rtsp://admin:<standard_k22_pass>@192.168.0.151:554/h264/ch01/main/av_stream")
    parser.add_argument('--bitmask', dest='bitmask', action='store_true')
    parser.add_argument('--no-bitmask', dest='bitmask', action='store_false')
    parser.set_defaults(bitmask=False)
    parser.add_argument('--motion_decay', help='motion decay rate ', type=float, default=0.6)
    parser.add_argument('--bg_decay', help='bg decay rate ', type =float, default=0.95)
    parser.add_argument('--binary_thresh', help='binary thresholding min bound 0 to 255', type=int, default=40)

    args = parser.parse_args()

    print(args)

    #send_sms("test.jpg")
    while True:
        main(args)