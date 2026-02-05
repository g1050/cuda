if [ "$ARCH" = "arm" ]; then \
        sed -i "s@http://.*ports.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list; \
else \
    sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list; \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list; \
fi; \


apt install -y libarmadillo-dev libopenblas-dev liblapack-dev
apt install -y libgoogle-glog-dev
apt install -y libgtest-dev
apt install doxygen graphviz

docker pull swr.cn-north-4.myhuaweicloud.com/xkgao/infer_env:v1.0