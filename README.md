# VIMA: General Robot Manipulation with Multimodal Prompts
## ICML 2023
<div align="center">

[[Website]](https://vimalabs.github.io/)
[[arXiv]](https://arxiv.org/abs/2210.03094)
[[PDF]](https://vimalabs.github.io/assets/vima_paper.pdf)
[[Pretrained Models]](#Pretrained-Models)
[[Baselines Implementation]](#Baselines-Implementation)
[[VIMA-Bench]](https://github.com/vimalabs/VimaBench)
[[Training Data]](https://huggingface.co/datasets/VIMA/VIMA-Data)
[[Model Card]](model-card.md)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/vimalabs/VIMA)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/vimalabs/VIMA)](https://github.com/vimalabs/VIMA/blob/main/LICENSE)
______________________________________________________________________
![](images/pull.png)
</div>

プロンプトベースの学習は自然言語処理において成功したパラダイムとして登場しており、1つの汎用的な言語モデルが入力プロンプトによって指定された任意のタスクを実行できるように指示されることが可能です。しかし、ロボティクスにおけるさまざまなタスクは依然として専門的なモデルによって解決されています。本研究では、マルチモーダルプロンプトを用いることで、テキストと視覚的なトークンを組み合わせることで、幅広いロボット操作タスクを表現できることを示します。

我々はVIMA (VisuoMotor Attention agent) を紹介します。これは、マルチモーダルプロンプトを通じて一貫したシーケンス入出力インターフェースを実現し、スケーラブルなマルチタスクロボット学習を可能にする新しい手法です。このアーキテクチャは、自然言語処理において効果的かつスケーラブルであることが証明されたエンコーダ-デコーダ型のトランスフォーマーデザインに従っています。VIMAは、[事前学習]((https://www.deepmind.com/publications/multimodal-few-shot-learning-with-frozen-language-models))済みの[言語モデル](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)やマルチモーダル少数ショット学習を使用して、テキストと視覚プロンプトトークンが交互に並ぶ入力シーケンスをエンコードし、各環境でのインタラクションステップごとにロボット制御アクションを自己回帰的にデコードします。

トランスフォーマーデコーダは、プロンプトを条件付けとしてクロスアテンション層を介して動作し、この層は通常の因果的自己アテンション層と交互に配置されています。VIMAは生のピクセルで動作する代わりに、オブジェクト中心のアプローチを採用しています。プロンプトまたは観察内のすべての画像を[既存の検出器](https://arxiv.org/abs/1703.06870)によってオブジェクトとして解析し、それらをオブジェクトトークンのシーケンスとしてフラット化します。

これらすべての設計選択を組み合わせることで、概念的にシンプルでありながら、モデルおよびデータのスケーリング特性を備えたアーキテクチャが実現されています。

# Pretrained Models
VIMAは、幅広いモデル容量をカバーする事前学習済みモデルを[Hugging Face](https://huggingface.co/VIMA/VIMA)でホストしています。ダウンロードリンクは以下に記載されています。Mask R-CNNモデルは[こちら](https://huggingface.co/VIMA/VIMA/resolve/main/mask_rcnn.pth)から入手できます。

| Model Size | Download Link                                                                 |
|------------|------------------------------------------------------------------------------|
| 200M       | [200M](https://huggingface.co/VIMA/VIMA/resolve/main/200M.ckpt)             |
| 92M        | [92M](https://huggingface.co/VIMA/VIMA/resolve/main/92M.ckpt)               |
| 43M        | [43M](https://huggingface.co/VIMA/VIMA/resolve/main/43M.ckpt)               |
| 20M        | [20M](https://huggingface.co/VIMA/VIMA/resolve/main/20M.ckpt)               |
| 9M         | [9M](https://huggingface.co/VIMA/VIMA/resolve/main/9M.ckpt)                 |
| 4M         | [4M](https://huggingface.co/VIMA/VIMA/resolve/main/4M.ckpt)                 |
| 2M         | [2M](https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt)                 |

# Baselines Implementation
我々のマルチモーダルプロンプト設定にそのまま適用可能な既存の方法が存在しないため、代表的なトランスフォーマーベースのエージェントアーキテクチャをいくつか選定し、これらをVIMA-Benchに対応できるよう再解釈する最善の努力を行いました。それらには、`VIMA-Gato`、`VIMA-Flamingo`、および`VIMA-GPT`が含まれます。これらの実装は`policy`フォルダ内にあります。

# Demo
ライブデモを実行するには、まずVIMA-Benchをインストールするための[インストール手順](https://github.com/vimalabs/VimaBench/tree/main#installation)に従ってください。その後、以下のコマンドでライブデモを実行できます。

```
python3 scripts/example.py --ckpt={ckpt_path} --device={device} --partition={eval_level} --task={task}
```
ここで、eval_levelは4つの評価レベルのいずれかを意味し、以下から選択できます：

	•	placement_generalization
	•	combinatorial_generalization
	•	novel_object_generalization
	•	novel_task_generalization

taskは特定のタスクテンプレートを意味します。詳細については、[タスクスイート](https://github.com/vimalabs/VimaBench/tree/main#task-suite)および[ベンチマーク](https://github.com/vimalabs/VimaBench/tree/main#evaluation-benchmark)を参照してください。例えば、以下のコマンドを実行します：
```
python3 scripts/example.py --ckpt=200M.ckpt --partition=placement_generalization --task=follow_order
```

上記のコマンドを実行すると、PyBulletのGUIがポップアップし、マルチモーダルプロンプトを表示する小さなウィンドウが現れます。その後、ロボットアームが対応するタスクを完了する動作を行うはずです。なお、このデモはPyBullet GUIがディスプレイを必要とするため、ヘッドレスマシンでは動作しない場合があります。

# Setup時のメモ
- python環境
`3.10.4`
- transformersのバージョン
`4.34.1`

## CPU環境での実行
```
python scripts/example.py --ckpt=pretrained/2M.ckpt --device='cpu' --partition=placement_generalization --task=follow_order
```

## ToDo
- [x] CPUでの実行環境
- [x] オブジェクトエンコーダーの残差接続
- [x] コード理解
- [ ] 逆動力学に基づく事前学習の導入
- [ ] Transformerをエンコーダデコーダモデルからデコーダモデルへ変換
- [ ] プロンプトエンコーディングと動作次元間の依存関係をモデル内で統一
- [ ] 各動作次元を個別のトークンとして表現
- [ ] 各トークンをオートリグレッシブにデコードする仕組みを導入
- [ ] 逆動力学を用いた事前学習
	- [ ] 任意のロボット軌道データを「動作追従タスク」として再構成し，これを学習データに活用
- [ ] マルチタスクデータセットを使ってモデルを微調整
- [ ] オートリグレッシブデコードのロス関数変更

## 用語説明
### 動作次元
動作次元とは、ロボットが実行する動作を構成する個々の要素（パラメータ）のこと。これらの要素は、ロボットが物体を操作する際に必要な情報を具体化したもので、動作を適切に制御するために必要。

例）

例えば、ロボットアームが物体を「ピックアンドプレース（Pick and Place）」する場合。この動作には以下のような動作次元がある：

1. 初期位置 (Tinitial)
ロボットが物体を掴む（ピックアップする）場所の座標。

具体的には、物体の位置（X, Y, Z座標）や回転角度（オリエンテーション）など。
2. 目標位置 (Ttarget)
ロボットが物体を配置する（プレースする）場所の座標。

同様に、配置する位置のX, Y, Z座標や回転角度が含まれます。

**動作次元の特徴**
- 独立した情報: 各次元は個別に定義される情報だが，最終的な動作を完成させるためには，互いに関連している場合が多い．
- ロボットの種類やタスクによって異なる: 動作次元はロボットの設計やタスク内容によって異なる

### オートリグレッシブとは
シーケンスの要素を前の要素に依存しながら一つずつ予測していく手法．