pip install gdown

# check if DeAOT model exists
if [ -f "./Segment-and-Track-Anything/ckpt/DeAOTT_PRE_YTB_DAV.pth" ]; 
then
    echo "DeAOT model exists"
else
    # download aot-ckpt 
    gdown --id '1ThWIZQS03cYWx1EKNN8MIMnJS5eRowzr' --output ./Segment-and-Track-Anything/ckpt/DeAOTT_PRE_YTB_DAV.pth
fi

# check if SAM model exists
if [ -f "./Segment-and-Track-Anything/ckpt/sam_vit_b_01ec64.pth" ]; 
then
    echo "SAM model exists"
else
    # download sam-ckpt
    wget -P ./Segment-and-Track-Anything/ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# check if DINOv2 model exists
if [ -f "./Segment-and-Track-Anything/dinov2_vits14_pretrain.pth" ]; 
then
    echo "DINOv2 model exists"
else
    # download dinov2 model
    wget -P ./Segment-and-Track-Anything/ckpt https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
fi


