from functools import cached_property
import os
import shutil
import imageio
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPVisionModelWithProjection

from models.guider import Guider
from models.referencenet import ReferenceNet2DConditionModel
from models.unet import UNet3DConditionModel
from models.video_pipeline import VideoPipeline

from dataset.val_dataset import ValDataset, val_collate_fn
from track import Tracker


def load_model_state_dict(model, model_ckpt_path, name):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()
    model_new_sd = {}
    count = 0
    for k, v in ckpt.items():
        if k in model_state_dict:
            count += 1
            model_new_sd[k] = v
    miss, _ = model.load_state_dict(model_new_sd, strict=False)
    print(f"load {name} from {model_ckpt_path}\n - load params: {count}\n - miss params: {miss}")


class VideoInference:
    def __init__(self, config):
        """初始化模型，只需执行一次"""
        # dist.init_process_group(backend="nccl")
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(self.local_rank)

        # 设置权重类型
        if config.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        elif config.weight_dtype == "fp32":
            self.weight_dtype = torch.float32
        else:
            raise ValueError(f"Do not support weight dtype: {config.weight_dtype}")

        # 初始化模型
        print("init model")
        self.vae = AutoencoderKL.from_pretrained(config.vae_model_path).to(dtype=self.weight_dtype, device="cuda")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(dtype=self.weight_dtype, device="cuda")
        self.referencenet = ReferenceNet2DConditionModel.from_pretrained_2d(
            config.base_model_path, subfolder="unet", referencenet_additional_kwargs=config.model.referencenet_additional_kwargs
        ).to(device="cuda")
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            config.base_model_path,
            motion_module_path=config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=config.model.unet_additional_kwargs,
        ).to(device="cuda")
        self.lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda")

        # 加载模型权重
        print("load model")
        load_model_state_dict(self.referencenet, f"{config.init_checkpoint}/referencenet.pth", "referencenet")
        load_model_state_dict(self.unet, f"{config.init_checkpoint}/unet.pth", "unet")
        load_model_state_dict(self.lmk_guider, f"{config.init_checkpoint}/lmk_guider.pth", "lmk_guider")

        # 启用 xformers（如果可用）
        if config.enable_xformers_memory_efficient_attention and is_xformers_available():
            self.referencenet.enable_xformers_memory_efficient_attention()
            self.unet.enable_xformers_memory_efficient_attention()
        elif config.enable_xformers_memory_efficient_attention:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.unet.set_reentrant(use_reentrant=False)
        self.referencenet.set_reentrant(use_reentrant=False)

        # 设置为评估模式
        self.vae.eval()
        self.image_encoder.eval()
        self.unet.eval()
        self.referencenet.eval()
        self.lmk_guider.eval()

        # 初始化噪声调度器
        print("init noise scheduler")
        sched_kwargs = OmegaConf.to_container(config.scheduler)
        if config.enable_zero_snr:
            sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction")
        self.noise_scheduler = DDIMScheduler(**sched_kwargs)

        # 初始化 pipeline
        self.pipeline = VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            referencenet=self.referencenet,
            unet=self.unet,
            lmk_guider=self.lmk_guider,
            scheduler=self.noise_scheduler,
        ).to(self.vae.device, dtype=self.weight_dtype)

        # 初始化随机数生成器
        self.generator = torch.Generator(device=self.vae.device)
        self.generator.manual_seed(config.seed)

        # 保存配置参数
        self.config = config

    @torch.no_grad()
    def infer(self, input_path, lmk_path, output_path, limit=1, num_inference_steps=30, guidance_scale=3.5):
        """执行推理过程，可多次调用"""
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 初始化数据集和数据加载器
        print("init dataset")
        val_dataset = ValDataset(
            input_path=input_path, lmk_path=lmk_path, resolution_h=self.config.resolution_h, resolution_w=self.config.resolution_w
        )
        print(f"Dataset size: {len(val_dataset)}")
        # sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, collate_fn=val_collate_fn)

        for i, batch in enumerate(val_dataloader):
            ref_frame = batch["ref_frame"][0]
            clip_image = batch["clip_image"][0]
            motions = batch["motions"][0]
            file_name = batch["file_name"][0]
            if motions is None:
                continue
            lmk_name = batch.get("lmk_name", ["lmk"])[0].split(".")[0]
            print(file_name, lmk_name)

            # 处理参考帧
            ref_frame = torch.clamp((ref_frame + 1.0) / 2.0, min=0, max=1)
            ref_frame = ref_frame.permute((1, 2, 3, 0)).squeeze()
            ref_frame = (ref_frame * 255).cpu().numpy().astype(np.uint8)
            ref_image = Image.fromarray(ref_frame)

            # 处理动作帧
            motions = motions.permute((1, 2, 3, 0))
            motions = (motions * 255).cpu().numpy().astype(np.uint8)
            lmk_images = [Image.fromarray(motion) for motion in motions]
            print("Len target images", len(lmk_images))

            # 运行 pipeline
            preds = self.pipeline(
                ref_image=ref_image,
                lmk_images=lmk_images,
                width=self.config.resolution_w,
                height=self.config.resolution_h,
                video_length=len(lmk_images),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=self.generator,
                clip_image=clip_image,
            ).videos

            preds = preds.permute((0, 2, 3, 4, 1)).squeeze(0)
            preds = (preds * 255).cpu().numpy().astype(np.uint8)

            # 保存视频
            mp4_path = os.path.join(output_path, f"{file_name.split('.')[0]}_oo.mp4")
            mp4_writer = imageio.get_writer(mp4_path, fps=25)
            print(preds.shape)
            for pred in preds:
                mp4_writer.append_data(pred)
            mp4_writer.close()

            mp4_path = os.path.join(output_path, f"{file_name.split('.')[0]}_all.mp4")
            mp4_writer = imageio.get_writer(mp4_path, fps=25)
            if "frames" in batch:
                frames = batch["frames"][0]
                frames = torch.clamp((frames + 1.0) / 2.0, min=0, max=1)
                frames = frames.permute((1, 2, 3, 0))
                frames = (frames * 255).cpu().numpy().astype(np.uint8)
                for frame, motion, pred in zip(frames, motions, preds):
                    out = np.concatenate((frame, motion, ref_frame, pred), axis=1)
                    mp4_writer.append_data(out)
            else:
                for motion, pred in zip(motions, preds):
                    out = np.concatenate((motion, ref_frame, pred), axis=1)
                    mp4_writer.append_data(out)
            mp4_writer.close()

            if i >= limit - 1:
                break


from expdataloader import *


class FollowYourEmojiLoader(RowDataLoader):
    def __init__(self, name="FollowYourEmoji"):
        super().__init__(name)

    @cached_property
    def tracker(self):
        return Tracker()

    def get_lmk_path(self, row: RowData):
        return os.path.join(row.output_dir, "mp_ldmk.npy")

    def run_video(self, row):
        lmk_path = self.get_lmk_path(row)
        if not os.path.exists(lmk_path):
            self.tracker.track(row.target.video_path, lmk_path)
        config = OmegaConf.load("./configs/infer.yaml")
        infer = VideoInference(config)
        input_path = row.source_img_path
        output_video_path = os.path.join(row.output_dir, f"{row.source_name}_oo.mp4")
        if not os.path.exists(output_video_path):
            infer.infer(input_path, lmk_path, row.output_dir)
        shutil.copyfile(output_video_path, row.output_video_path)
        row.output.human()


def main():
    loader = FollowYourEmojiLoader()
    # loader.run_video(loader.all_data_rows[0])
    loader.run_all()


def test():
    config = OmegaConf.load("./configs/infer.yaml")
    inference = VideoInference(config)
    input_path = "./data/images"
    lmk_path = "./inference_temple/test_temple.npy"
    output_path = "./data/out"
    inference.infer(input_path, lmk_path, output_path, limit=100000000, num_inference_steps=30, guidance_scale=3.5)


if __name__ == "__main__":
    main()
