# ### cubestat monitoring
echo "Setting up cubestat"
# cubestat itself
git clone https://github.com/okuvshynov/cubestat.git
# dependency to monitor nvidia cards
pip install pynvml

# ### basic vim config
echo "Setting up vim"
# vundle
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
# vimrc
wget -O ~/.vimrc https://raw.githubusercontent.com/okuvshynov/vimrc/master/.vimrc
# install plugins
vim -c ":PluginInstall" -c ":qa"

# ### torch to TensorRT
echo "Setting up torch2trt"
pip install tensorrt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt/
sudo chown ubuntu /usr/local/lib/python3.8/dist-packages/
python setup.py install
cd ..

# ### rlscout itself
echo "Setting up rlscout"
git clone https://github.com/okuvshynov/rlscout.git
cd rlscout/mnklib && make all && cd ../..

echo "Done."