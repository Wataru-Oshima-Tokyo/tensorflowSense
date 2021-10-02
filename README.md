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

## install tensorflow
    sudo apt install npm curl -y
    npm install -g @bazel/bazelisk
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ./configure
