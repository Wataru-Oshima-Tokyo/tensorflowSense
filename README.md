# tensorflowSense

# For launching detectObject

    roslaunch tensorflowSense detectObject.launch
        //rostopic is below
            /camera/tensorflow/image_raw


# For launching detectPerson (not yet completed)

     roslaunch tensorflowSense detectPerson.launch

      //rostopic is below
        /camera/tensorflow/image_raw
        /camera/tensorflow/object
        /camera/tensorflow/distance

make sure that you have not launched the realsense node

## install tensorflow for Linux x86
    sudo apt install npm curl -y
    npm install -g @bazel/bazelisk
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ./configure
## install tensorflow for Raspberry Pi
    sudo apt-get install gfortran
    sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
    sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
    sudo apt-get install openmpi-bin libopenmpi-dev
    sudo apt-get install liblapack-dev cython
    sudo pip3 install keras_applications==1.0.8 --no-deps
    sudo pip3 install keras_preprocessing==1.1.0 --no-deps
    sudo pip3 install -U --user six wheel mock
    sudo -H pip3 install pybind11
    sudo -H pip3 install h5py==2.10.0
    # upgrade setuptools 40.8.0 -> 52.0.0
    sudo -H pip3 install --upgrade setuptools
    # download the wheel
    wget https://github.com/Qengineering/Tensorflow-Raspberry-Pi/raw/master/tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
    # install TensorFlow
    sudo -H pip3 install tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
    # and complete the installation by rebooting
    sudo reboot
