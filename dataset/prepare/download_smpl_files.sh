mkdir -p checkpoints/smpl_models
cd checkpoints/smpl_models/

echo -e "The smpl files will be stored in the 'checkpoints/smpl_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"
cd ../..