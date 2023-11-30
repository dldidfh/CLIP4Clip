import os , cv2 
import numpy as np 
import torch as th 
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class RamRawVideoExtractor:
    def __init__(self, feature_path, img_size=224, sample_fq:int=0, max_frame:int=12) -> None:
        self.max_frame=max_frame
        self.sample_fq = sample_fq
        self.img_size=img_size
        self.feature_path = feature_path

    def prepare_video_datas_with_ram(self, video_id):
        video_path = os.path.join(self.feature_path, "{}.mp4".format(video_id))
        if os.path.exists(video_path) is False:
            video_path = video_path.replace(".mp4", ".webm")
        raw_video_data = self.video_to_tensor(video_path)
        return (video_id,raw_video_data)

    def transform(self):
        return Compose([
                Resize(self.img_size, interpolation=Image.BICUBIC),
                CenterCrop(self.img_size),
                # lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    def video_to_tensor(self, video_file, start_time=None, end_time=None):
        sample_fp = self.sample_fq
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images = []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for i, ind in enumerate(inds):
                if i < self.max_frame : 
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(frame_rgb))
                # images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        # if len(images) > 0:
        #     video_data = th.tensor(np.stack(images))
        # else:
        #     video_data = th.zeros(1)
        return images
ram_raw_video_extractor = RamRawVideoExtractor

# from glob import glob 
# file_list = os.listdir(root_path)
# video_dict = {}
# for i in range(len(file_list)):
#     file_name = file_list[i].split('.')[0]
#     ret = prepare_video_datas_with_ram(file_name,root_path)
#     video_dict[ret[0]] = ret[1]
