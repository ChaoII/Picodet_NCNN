# Picodet_NCNN

# 1.编译opencv

```bash
# 如果下载不下来，请使用训练拉取release中的源码
# 或者直接下载二进制文件
git clone https://github.com/opencv/opencv.git
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=install
make -j8
make install
```

具体opencv的编译选项见文件:[build_opencv_options.sh](./build_opencv_options.sh)，如果有权限问题请在命令前添加sudo

# 2.编译ncnn

```shell
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
cmake .. -DCMAKE_INSTALL_PREFIX=install
make -j8
make install
```

ncnn开启GPU（vulkan），其它编译选项请看[how to build](https://github.com/Tencent/ncnn/wiki/how-to-build)
,开发板等其它嵌入式，资源少的设备请查看[build minimal library](https://github.com/Tencent/ncnn/wiki/build-minimal-library)

# 3. 执行编译

```bash
mkdir build && cd build
cmake .. -DOpenCV_DIR="/opt/opencv-4.7.0/build/install/lib/static/cmake/opencv4" \
         -Dncnn_DIR = "/opt/ncnn/build/install/lib/cmake/ncnn"
make -j8
```

# 4. 运行测试

```
./picodet_demo
```

# 5. x86主机下arm交叉编译环境搭建（基于docker）
```shell
# 1.安装docker
# 2.拉取arm镜像
docker pull --platform=arm64 ubuntu:18.04
# 进入镜像
docker run -it --name iot ubuntu:18.04
# 安装编译环境
apt-get install build-essential git cmake vim
# 编译安装依赖
# 拷贝源码
docker cp C:\xx containerid:/opt
# 执行编译
# 导出镜像
docker export containerid > iot.tar
# 导入镜像
docker import iot.tar iot_build_env:v1.0
```

