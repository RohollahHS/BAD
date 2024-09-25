echo -e "Downloading pretrained models"

mkdir -p ./output/vq/
mkdir -p ./output/t2m/
mkdir -p ./checkpoints/t2m/length_estimator/model/

cd ./output

gdown --fuzzy https://drive.google.com/file/d/1fchcM7vWJpMKbDP7wTVrufIgfyYaXK49/view?usp=sharing
mv vq_last.pth vq/


gdown --fuzzy https://drive.google.com/file/d/1ZldeaE9mYOAsG9B2UM-Oc_BpQw75xW4l/view?usp=sharing
mv trans_best_fid.pth t2m/

cd ..

cd ./checkpoints/t2m/length_estimator/model/
gdown --fuzzy https://drive.google.com/file/d/1eFphHaWX669pXVgXJTRvCOdKonQ0Pns_/view?usp=sharing


echo -e "Finished"