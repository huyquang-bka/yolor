from utils.torch_utils import select_device

from models.models import *
from utils.datasets import *
from utils.general import *
import time

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


def detect(path):
    cap = cv2.VideoCapture(path)
    while True:
        s = time.time()
        ret, img0 = cap.read()
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
            im0 = img0.copy()

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cv2.rectangle(im0, (x1, y1), (x2, y2), colors, 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        fps = int(1 / (time.time() - s))
        print(fps)
        try:
            cv2.putText(im0, f"FPS: {fps}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imshow("Video", im0)
        except:
            cv2.putText(img0, f"FPS: {fps}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imshow("Video", img0)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == '__main__':
    path = r"D:\30_31-1-21-c9\1.mp4"
    with torch.no_grad():
        detect(path)
