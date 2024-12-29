# cuda,cudnn環境構築
FROM nvidia/cuda:11.1.1-cudnn8-devel
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# 必要なパッケージのインストール。Advanceにて書き換える箇所。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    vim \
    sudo \
    curl \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    x11-apps \
    xauth && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y mesa-utils

# user設定
ARG user=${user:-user}
ARG uid=${uid:-uid}
ARG gid=${gid:-gid}

# Anaconda環境構築
ENV ANACONDA_ROOT=/opt/conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${ANACONDA_ROOT} && \
    rm ~/miniconda.sh && \
    ${ANACONDA_ROOT}/bin/conda clean -ya

# condaのpathを通す
ENV PATH=${ANACONDA_ROOT}/bin:$PATH

# condaのpathが通っているか確認
ARG PYTHON_VERSION=3.9
RUN conda install -y python=$PYTHON_VERSION && conda clean -ya

# 必要な追加モジュールのインストール
# pipのバージョンをダウングレード
RUN python -m pip install --no-cache-dir pip==23.2.1
RUN python3 -m pip install --no-cache-dir wheel==0.38.4 setuptools==66

# root権限で作成したディレクトリなどがsudo権限がないと操作できない問題対策
# dockerという仮のグループを作成し、ユーザーを追加することで権限を下げる。
RUN groupadd -g ${gid} docker && \
    useradd -g docker -u ${uid} -s /bin/bash ${user}

# sudoerに自分を追加、パスワードなしでsudoコマンドを使えるようにする
RUN echo "${user} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ホームディレクトリを最初に表示する
WORKDIR /home/${user}

# ホームディレクトリの権限をユーザに下げる
RUN chown -R ${user} /home/${user}

# 作成したユーザーに切り替える
USER ${uid}

# pipする際に警告を出さないためにpathを通す
ENV PATH=/home/${user}/.local/bin:$PATH

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN git config --global --add safe.directory /workspace

# キャッシュ無効化のための一時的なファイル削除
RUN rm -rf ~/.cache/pip

# 新しい環境変数の追加
ENV PYTHONPATH=/home/${user}/workspace/VimaBench:$PYTHONPATH
