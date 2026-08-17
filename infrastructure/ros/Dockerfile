FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/root/ros_ws

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-rosbash \
    ros-noetic-std-msgs \
    ros-noetic-rospy \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install kafka-python pandas numpy

RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

CMD ["/bin/bash"]