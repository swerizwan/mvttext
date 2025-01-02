mkdir -p checkpoints/
cd checkpoints/
echo -e "The pretrained models will stored in the 'checkpoints' folder\n"
mkdir -p motiontext_humanml3d_checkpoint/
cd motiontext_humanml3d_checkpoint/
# mld_humanml3d.ckpt
gdown "https://drive.google.com/uc?id=1hplrnQwUK_cZFHirZIOuVP0RSyZEC1YM"

echo -e "Downloading done!"
