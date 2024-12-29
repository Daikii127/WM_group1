import os
import torch

from .policy import *


def create_policy_from_ckpt(ckpt_path, device):
    # チェックポイントのパスが存在するか確認
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"

    # 重みファイルの読み込み
    # ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # チェックポイントに保存された設定を用いてポリシーインスタンスを生成
    # ckpt["cfg"]にはポリシーの設定情報が格納されていると想定されている
    policy_instance = VIMAPolicy(**ckpt["cfg"])

    # チェックポイントに保存された状態をポリシーインスタンスにロード
    # ステートディクショナリ内のキーのプレフィックス"policy."を削除して適用
    # strict=Trueにより，全てのパラメータが一致しない場合にエラーを発生させる
    policy_instance.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )

    # ポリシーを評価モードに設定
    # 評価モードでは，ドロップアアウトやバッチ正規化が固定される
    policy_instance.eval()

    # ポリシーインスタンスを返す
    return policy_instance
