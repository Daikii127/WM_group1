#!/bin/bash
#.origファイルのコマンドの出力を補完して.ymlを作成する
printf  "cat <<EOF\n`cat docker-compose.yml.orig`\nEOF\n" | sh > docker-compose.yml
#workspaseの作成
mkdir -p ~/workspace
#docker-composeの起動。-d:バックグラウンド処理 --build:Imageファイルの有無にかかわらずDockerfileからビルドを行う
docker-compose up -d --build
