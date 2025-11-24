# server/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates curl unzip \
    python3 python3-pip libssl-dev pkg-config

# --- Install nlohmann json (dev package) for convenience (optional) ---
RUN apt-get install -y nlohmann-json3-dev
RUN apt-get install -y zsh

# --- Fetch single-header cpp-httplib and nlohmann json (single-header fallback) ---
RUN mkdir -p /opt/third_party
RUN curl -fsSL https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -o /opt/third_party/httplib.h
RUN curl -fsSL https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o /opt/third_party/json.hpp

# Copy third party into build context
RUN mkdir -p /app/third_party
RUN cp /opt/third_party/httplib.h /app/third_party/
RUN cp /opt/third_party/json.hpp /app/third_party/

# --- Download ONNX Runtime prebuilt package ---
# Set an ONNXRUNTIME_VERSION build-arg when building if you want a specific version.
ARG ONNXRUNTIME_VERSION=1.20.0
WORKDIR /opt
RUN echo "Downloading onnxruntime version ${ONNXRUNTIME_VERSION}" && \
    curl -fsSL -o onnxruntime.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime.tgz && \
    rm onnxruntime.tgz && \
    mv onnxruntime-linux-x64-${ONNXRUNTIME_VERSION} onnxruntime

# oh-my-zsh 설치
RUN sh -c "$(curl -fsSL https://gitee.com/mcornella/ohmyzsh/raw/master/tools/install.sh)" "" --unattended

# zsh 플러그인 설치
RUN git clone https://github.com/zsh-users/zsh-autosuggestions.git ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# .zshrc 수정
RUN sed -i 's/ZSH_THEME=".*"/ZSH_THEME="robbyrussell"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# 기본 셸을 zsh로 설정
RUN chsh -s $(which zsh)

EXPOSE 8000
# 시작 시 zsh 실행
CMD ["zsh"]