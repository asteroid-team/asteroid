#!/bin/bash

# Make lfs firectory
mkdir -p lfs_install
cd lfs_install

# Download .tar file and extract
wget https://github.com/git-lfs/git-lfs/releases/download/v2.10.0/git-lfs-linux-amd64-v2.10.0.tar.gz
tar -xzvf git-lfs-linux-amd64-v2.10.0.tar.gz

# To install without sudo? git-lfs will be installed in ~/.local/bin instead of /usr/local/bin
sed -i 's+/usr/local+$HOME/.local+g' install.sh

# Run the install script
. ./install.sh

# Export path
export PATH=$PATH:$HOME/.local/bin

echo -e "Installed git-lfs and temporarly added it to your path. To make it permanent, run the following command \n"
echo "echo 'export PATH=\$PATH:\$HOME/.local/bin' >> ~/.bashrc"
cd ../

