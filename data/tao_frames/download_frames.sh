# unzip all files into the same TAO dataset
TAO_PATH="/path/to/TAO-Amodal"

mkdir -p $TAO_PATH
cd $TAO_PATH
# Download dataset

wget --progress=bar:force --output-file=Download_log -O TAO_TRAIN.zip "https://motchallenge.net/data/1-TAO_TRAIN.zip"
wget --progress=bar:force --output-file=Download_log -O TAO_VAL.zip "https://motchallenge.net/data/2-TAO_VAL.zip" 
wget --progress=bar:force --output-file=Download_log -O TAO_TEST.zip "https://motchallenge.net/data/3-TAO_TEST.zip" 



unzip TAO_TRAIN.zip -d $TAO_PATH
unzip TAO_VAL.zip -d $TAO_PATH
unzip TAO_TEST.zip -d $TAO_PATH

# remove all .zip
rm TAO_TRAIN.zip
rm TAO_VAL.zip
rm TAO_TEST.zip

# download the AVA HACS (need to request first)
wget "https://motchallenge.net/data/1_AVA_HACS_TRAIN_c6f3b138ebdf6a348b7c8f932f5a403f.zip" --progress=bar:force --output-file=Download_log -O $TAO_PATH/TAO_AVA_HACS_TRAIN.zip
wget "https://motchallenge.net/data/2_AVA_HACS_VAL_c6f3b138ebdf6a348b7c8f932f5a403f.zip" --progress=bar:force --output-file=Download_log -O $TAO_PATH/TAO_AVA_HACS_VAL.zip
wget "https://motchallenge.net/data/3_AVA_HACS_TEST_c6f3b138ebdf6a348b7c8f932f5a403f.zip" --progress=bar:force --output-file=Download_log -O $TAO_PATH/TAO_AVA_HACS_TEST.zip

unzip TAO_AVA_HACS_TRAIN.zip -d $TAO_PATH
unzip TAO_AVA_HACS_VAL.zip -d $TAO_PATH
unzip TAO_AVA_HACS_TEST.zip -d $TAO_PATH

# remove all .zip
rm TAO_AVA_HACS_TRAIN.zip
rm TAO_AVA_HACS_VAL.zip
rm TAO_AVA_HACS_TEST.zip
