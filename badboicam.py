from collections import deque
import numpy as np
import argparse
import cv2
import time
import sys
from datetime import datetime
from sklearn.linear_model import LinearRegression
import logging


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
        dump_file = "%s.mp4" % datetime.now()
        logging.info('Dumping video record: %s', dump_file)
        size =self.frames[0].shape[1],self.frames[0].shape[0]
        output = cv2.VideoWriter(dump_file, cv2.VideoWriter_fourcc(*'mp4v'), 13, size )
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
        self.recorder = Recorder(max_frames=frame_count_max*2)

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

            if self.last_trigger_count >= self.trigger_threshold:
                if time.time() < self.cooldown_time:
                    logging.info('Alert cooldown at %s', datetime.now())
                else:
                    logging.info('Alert trigger with count: %d', datetime.now(), self.trigger_count)
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
        self.trigger_frame = None
        self.recorder.flush()

    def get_last_frame_set_count(self):
        return self.last_trigger_count


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def main(logging, args):
    pts = deque(maxlen=args.buffer)
    counter = 0

    loop_count = 0
    last_time = 0
    vcap = cv2.VideoCapture(args.input)
    fps = 11

    tw = TriggerWatcher(3, 15*fps)
    last_frame = None
    while(1):
        ret, frame_ori = vcap.read()
        if frame_ori is None:
            break

        if frame_ori.shape[1] > 1000:
            frame = cv2.resize(frame_ori,(int(frame_ori.shape[1]/2),int(frame_ori.shape[0]/2)))
        else:
            frame = frame_ori

        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        if last_frame is None:
            last_frame = mask
            continue

        frameDelta = cv2.absdiff(mask, last_frame)
        last_frame = cv2.addWeighted(last_frame, 0.95, mask, 0.05, 0)

        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)

        cnts,hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            sum_x = 0
            sum_y = 0
            total_count = 0
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius < 20:
                    continue

                M = cv2.moments(c)
                area = cv2.contourArea(c)
                sum_x += (M["m10"] / M["m00"])*area
                sum_y += (M["m01"] / M["m00"])*area
                total_count += area

            if total_count != 0:
                center = (int(sum_x/total_count), int(sum_y/total_count))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
        else:
            if len(pts) > 0:
                pts.pop()

        trigger = False
        if len(pts) > args.buffer*args.buffer_trigger_thresh:
            X = np.array([x[0] for x in pts])
            Y = np.array([x[1] for x in pts])

            X = X.reshape(-1, 1)
            Y = Y.reshape(-1, 1)
            reg = LinearRegression().fit(X, Y)
            score = reg.score(X,Y)
            X = np.array([0, frame.shape[1]]).reshape(-1, 1)
            predict_y = reg.predict(X)

            for i in range(0, len(X)-1):
                if score < args.regression_score_min:
                    break
                logging.debug("Trigger score ACCEPTED: %2.2f", score)
                point_a = (int(X[i][0]), int(predict_y[i][0]))
                point_b = (int(X[i+1][0]), int(predict_y[i+1][0]))

                point_c = (0, frame.shape[0]-100)
                point_d = (frame.shape[1]-400, frame.shape[0])
                line1 = [point_a, point_b]
                line2 = [point_c, point_d]
                intersect = line_intersection(line1, line2)
                cv2.line(frame, point_a, point_b, (255, 0, 255), 3)
                if intersect and \
                        intersect[0] < frame.shape[1] and \
                        intersect[1] < frame.shape[0] and \
                        intersect[0] >= 0 and \
                        intersect[1] >= 0:
                    # trigger
                    trigger = True
                    cv2.line(frame, point_c, point_d, (255, 0, 0), 3)

            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 3)

        tw.update(frame, trigger)

        for i in np.arange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 3)

        if trigger:
            pts.clear()

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        loop_count += 1

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input video file or stream', default="rtsp://admin:<standard_k22_pass>@192.168.0.151:554/h264/ch01/main/av_stream")

    parser.add_argument('--buffer', help='point buffer size ', type =int, default=20)
    parser.add_argument('--buffer_trigger_thresh', help='trigger thresh', type =float, default=0.8)
    parser.add_argument('--regression_score_min', help='linear regression min score', type =float, default=0.4)

    args = parser.parse_args()

    logging.basicConfig(filename="badboicam.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.debug('Startup')
    while True:
        main(logging, args)
