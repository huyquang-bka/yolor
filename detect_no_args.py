import cv2

from utils.torch_utils import select_device

from models.models import *
from utils.datasets import *
from utils.general import *
import time
from wang_pakage.process_map import is_in_parking_line

import opt


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


out, source, weights, view_img, save_txt, imgsz, cfg, names = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names

# Initialize
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = Darknet(cfg, imgsz).cuda()
model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
# model = attempt_load(weights, map_location=device)  # load FP32 model
# imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
model.to(device).eval()
if half:
    model.half()  # to FP16

# Get names and colors
names = load_classes(names)
colors = (0, 0, 255)


# Run inference
# img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def detect(img0):
    H, W, _ = img0.shape
    img = letterbox(img0, new_shape=imgsz, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                # if not is_in_parking_line(x_center, y_center):
                #     continue
                cv2.rectangle(img0, (x1, y1), (x2, y2), colors, 2)
                cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    return img0


if __name__ == '__main__':
    path = r"C:\Users\Admin\Downloads\ch16_C5 chụp gần ra chiền 10012022 3.mp4"
    cap = cv2.VideoCapture(path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    # fourcc = 'mp4v'  # output video codec
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # vid_writer = cv2.VideoWriter(r"D:\Lab IC\demo\ch16_C3 vào_sau 17h25 06012022.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    check = -1
    with torch.no_grad():
        while True:
            t = time.time()
            ret, img0 = cap.read()
            if check == 1:
                # img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # img0 = cv2.resize(img0, dsize=None, fx=0.8, fy=0.8)
                img0 = detect(img0)
                # img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Image", img0)
            # vid_writer.write(img0)
            key = cv2.waitKey(1)
            print("FPS: ", 1 // (time.time() - t))
            if key == ord("q"):
                break
            if key == ord("c"):
                check = -check
