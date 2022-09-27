import os
import os.path as osp
from tqdm import tqdm
from glob import glob


video_list = sorted(glob(f"../data/fullFrame-256x256px/*/*/*/"))
print(len(video_list))

for image_dir in tqdm(video_list):
    write_json = image_dir.replace(
        "fullFrame-256x256px", "fullFrame-256x256px-openpose"
    )
    write_images = image_dir.replace(
        "fullFrame-256x256px", "fullFrame-256x256px-openpose-vis"
    )
    os.makedirs(write_json, exist_ok=True)
    os.makedirs(write_images, exist_ok=True)

    cmd = f"CUDA_VISIBLE_DEVICES=0 /openpose/signLanguage/openpose/build/build_windows/examples/openpose/openpose.bin  \
        --image_dir {image_dir} --display 0  \
            --number_people_max 1 --face --hand \
                --write_json {write_json}/ \
                    --num_gpu 1  \
                        --render_pose 1 --write_images {write_images}"
    print(cmd)
    os.system(cmd)