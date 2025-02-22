ml proxy

conda create -n fye python=3.10 -y 
conda activate fye

pip install -r requirements.txt
pip install -e ../expdata
conda install ffmpeg -c conda-forge -y
# 安装 huggingface-cli（如果还未安装）
pip install huggingface_hub==0.25.0

# 下载 follow-your-emoji 主模型文件到 pretrained_models/follow-your-emoji
huggingface-cli download YueMafighting/FollowYourEmoji --include "*.pth" --local-dir pretrained_models/follow-your-emoji
mv pretrained_models/follow-your-emoji/ckpts/* pretrained_models/follow-your-emoji
rm -r pretrained_models/follow-your-emoji/ckpts

# 下载 sd-image-variations-diffusers 模型
huggingface-cli download lambdalabs/sd-image-variations-diffusers --local-dir pretrained_models/sd-image-variations-diffusers

# 下载 sd-vae-ft-mse 模型
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir pretrained_models/sd-vae-ft-mse

# 下载 AnimateDiff 的 mm_sd_v15_v2.ckpt
huggingface-cli download guoyww/animatediff mm_sd_v15_v2.ckpt --local-dir pretrained_models/AnimateDiff