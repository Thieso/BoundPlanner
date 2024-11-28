ARG from=ubuntu:jammy
FROM ${from}

ARG CMAKE_INSTALL_PREFIX=/usr/local
ARG NUM_THREADS=8

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update all packages
RUN apt-get update && apt-get -y dist-upgrade

# ros-core include begin
# https://github.com/osrf/docker_images/blob/master/ros/humble/ubuntu/jammy/ros-core/Dockerfile
# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  dirmngr \
  gnupg2 \
  curl \
  && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO humble

# ros-base include begin
# https://github.com/osrf/docker_images/blob/master/ros/galactic/ubuntu/focal/ros-core/Dockerfile
# https://github.com/osrf/docker_images/blob/master/ros/humble/ubuntu/jammy/ros-base/Dockerfile
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  gfortran \
  liblapack-dev \
  git \
  python3-colcon-common-extensions \
  python3-colcon-mixin \
  python3-rosdep \
  python3-vcstool \
  && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init

# install dependencies via apt
ENV DEBCONF_NOWARNINGS yes


RUN apt-get update -y -qq && \
  : "system dependencies" && \
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get install -y -qq \
  vim \
  aptitude \
  autoconf \
  automake \
  build-essential \
  cmake \
  curl \
  dnsutils \
  git \
  git-lfs \
  gitk \
  gtk-doc-tools \
  intltool \
  iputils-ping \
  libblosc1 \
  libbondcpp-dev \
  libcgal-dev \
  mlocate \
  npm \
  pkg-config \
  python-odf-doc \
  python-odf-tools \
  python-tables-data \
  python3-bottleneck \
  python3-bs4 \
  python3-defusedxml \
  python3-et-xmlfile \
  python3-html5lib \
  python3-ipython \
  python3-jdcal \
  python3-numexpr \
  python3-odf \
  python3-opencv \
  python3-openpyxl \
  python3-pandas \
  python3-pandas-lib \
  python3-pip \
  python3-soupsieve \
  python3-tables \
  python3-webencodings \
  python3-xlrd \
  python3-xlwt \
  tar \
  unzip \
  wget \
  xterm \
  && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/* && \
  : "use pip for packages not in ubuntu repo"  && \
  # python code format
  /bin/python3 -m pip install -U autopep8

RUN apt-get update -y -qq && \
  : "install ROS basics" && \
  apt-get -y install --no-install-recommends \
  ros-humble-desktop \
  && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

RUN \
  # set -x for debug output
  set -x && \
  apt-get update -y -qq && \
  : "install ROS dependencies" && \
  apt-get -y install --no-install-recommends \
  ros-humble-ament-cmake-google-benchmark \
  ros-humble-ament-cmake-clang-format \
  ros-humble-backward-ros \
  ros-humble-bond \
  ros-humble-bondcpp \
  ros-humble-compressed-depth-image-transport \
  ros-humble-compressed-image-transport \
  ros-humble-control-msgs \
  ros-humble-control-toolbox \
  ros-humble-controller-interface \
  ros-humble-controller-manager \
  ros-humble-controller-manager-msgs \
  ros-humble-diagnostic-aggregator \
  ros-humble-diagnostic-updater \
  ros-humble-effort-controllers  \
  ros-humble-filters \
  ros-humble-force-torque-sensor-broadcaster \
  ros-humble-forward-command-controller \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-generate-parameter-library \
  ros-humble-hardware-interface \
  ros-humble-image-transport-plugins \
  ros-humble-imu-sensor-broadcaster \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui\
  # ros-humble-joint-trajectory-controller \
  ros-humble-nav2-bringup \
  ros-humble-navigation2 \
  ros-humble-octomap \
  ros-humble-octomap-msgs \
  ros-humble-perception-pcl \
  ros-humble-position-controllers \
  ros-humble-realtime-tools \
  ros-humble-robot-localization \
  ros-humble-ros2-control-test-assets \
  ros-humble-ros2-control\
  # ros-humble-ros2-controllers \
  ros-humble-ros2controlcli \
  ros-humble-rsl \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-rqt-robot-monitor \
  ros-humble-rqt-robot-steering \
  ros-humble-smclib \
  ros-humble-tcb-span \
  ros-humble-theora-image-transport \
  ros-humble-transmission-interface \
  ros-humble-ublox \
  ros-humble-velocity-controllers \
  ros-humble-xacro \
  && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

# install eigen
RUN \
  cd /opt/ && \
  git clone https://gitlab.com/libeigen/eigen.git && cd eigen && git checkout tags/3.4.0 &&\
  cd /opt/eigen && \
  mkdir build && cd build && \
  cmake .. && \
  make && \
  make install 

ARG USERNAME=ubuntu
ARG USER_UID=1005
ARG USER_GID=$USER_UID
ARG HOME=/home/${USERNAME}

RUN groupadd ubuntu -g${USER_GID}
RUN useradd -ms /bin/bash ${USERNAME} -u${USER_UID} -g${USER_GID} \
  && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers && \
  chmod 0440 /etc/sudoers && \
  chmod g+w /etc/passwd 

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

RUN echo "source /opt/ros/humble/setup.bash" >> "/home/${USERNAME}/.bashrc" \
  && usermod -aG 104 ${USERNAME} \
  && usermod -aG video ${USERNAME} \
  && usermod -aG dialout ${USERNAME}

# install python packages

# install coinhsl solver
# assumes coinhsl directory exists next to dockerfile
#COPY coinhsl /tmp/coinhsl
#RUN cd /tmp/coinhsl \
#  && autoreconf \
#  && ./configure --prefix=/opt/coinhsl/ LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O2 -fopenmp" FCFLAGS="-g -O2 -fopenmp" CFLAGS="-g -O2 -fopenmp" \
#  && make install \
#  && cd /opt/coinhsl/lib \
#  && ln -s libcoinhsl.so libhsl.so

# switch to regular user
USER ${USERNAME}

# add LD paths (coinhsl / local lib for arc)
#RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> "/home/${USERNAME}/.bashrc"
#RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/coinhsl/lib" >> "/home/${USERNAME}/.bashrc"

# create default ROS workspace so it can be sourced and added to bashrc
RUN \
  : "create default ROS2 workspace" && \
  mkdir -p /home/${USERNAME}/ros2_ws/src \
  && cd /home/${USERNAME}/ros2_ws/ \
  && colcon build

RUN echo "source /home/${USERNAME}/ros2_ws/install/setup.bash" >> "/home/${USERNAME}/.bashrc"

# copy requirements
COPY --chown=${USERNAME}:${USERNAME} bound_planner/requirements.txt /home/${USERNAME}/requirements.txt
RUN \
  : "install additional pip requirements" && \
  cd ${HOME} \
  && pip install -r requirements.txt

WORKDIR ${HOME}/ros2_ws

# create entrypoint to keep running
ENTRYPOINT ["tail", "-f", "/dev/null"]

