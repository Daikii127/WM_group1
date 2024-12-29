from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse
import copy

# 並列トークナイザーの設定を有効化
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# AddTokenの初期設定
_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

# プレースホルダーとなるトークンを定義
PLACEHOLDER_TOKENS = [
    AddedToken("{base_obj}", **_kwargs),
    AddedToken("{base_obj_1}", **_kwargs),
    AddedToken("{base_obj_2}", **_kwargs),
    AddedToken("{dragged_obj}", **_kwargs),
    AddedToken("{dragged_obj_1}", **_kwargs),
    AddedToken("{dragged_obj_2}", **_kwargs),
    AddedToken("{dragged_obj_3}", **_kwargs),
    AddedToken("{dragged_obj_4}", **_kwargs),
    AddedToken("{dragged_obj_5}", **_kwargs),
    AddedToken("{swept_obj}", **_kwargs),
    AddedToken("{bounds}", **_kwargs),
    AddedToken("{constraint}", **_kwargs),
    AddedToken("{scene}", **_kwargs),
    AddedToken("{demo_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_blicker_obj_3}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
    AddedToken("{start_scene}", **_kwargs),
    AddedToken("{end_scene}", **_kwargs),
    AddedToken("{before_twist_1}", **_kwargs),
    AddedToken("{after_twist_1}", **_kwargs),
    AddedToken("{before_twist_2}", **_kwargs),
    AddedToken("{after_twist_2}", **_kwargs),
    AddedToken("{before_twist_3}", **_kwargs),
    AddedToken("{after_twist_3}", **_kwargs),
    AddedToken("{frame_0}", **_kwargs),
    AddedToken("{frame_1}", **_kwargs),
    AddedToken("{frame_2}", **_kwargs),
    AddedToken("{frame_3}", **_kwargs),
    AddedToken("{frame_4}", **_kwargs),
    AddedToken("{frame_5}", **_kwargs),
    AddedToken("{frame_6}", **_kwargs),
    AddedToken("{ring}", **_kwargs),
    AddedToken("{hanoi_stand}", **_kwargs),
    AddedToken("{start_scene_1}", **_kwargs),
    AddedToken("{end_scene_1}", **_kwargs),
    AddedToken("{start_scene_2}", **_kwargs),
    AddedToken("{end_scene_2}", **_kwargs),
    AddedToken("{start_scene_3}", **_kwargs),
    AddedToken("{end_scene_3}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]

# T5ベースのトークナイザーを使用してプレースホルダーを追加
# トークナイザーの初期化
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)


@torch.no_grad()
def main(cfg):
    # 引数の確認
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]

    # シード値の固定
    seed = 42

    # 学習済みポリシーをチェックポイントから生成
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)

    # 環境を初期化(タイムリミットラッパーで包む)
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
                seed=seed,
                render_prompt=True,
                display_debug_window=True,
                hide_arm_rgb=False,
            )
        ),
        bonus_steps=2,
    )

    # 無限ループで環境ステップを実行
    while True:
        env.global_seed = seed

        # 環境をリセット
        obs = env.reset()
        env.render()

        # 環境から得られるメタ情報やプロンプトを取得
        meta_info = env.meta_info
        prompt = env.prompt
        prompt_assets = env.prompt_assets
        elapsed_steps = 0
        inference_cache = {}

        while True:
            # デバッグ出力用
            # print("Before prepare_obs call, obs:", obs)
            if elapsed_steps == 0:
                # 最初のステップでプロンプトの処理
                prompt_token_type, word_batch, image_batch = prepare_prompt(
                    prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
                )
                word_batch = word_batch.to(cfg.device)
                image_batch = image_batch.to_torch_tensor(device=cfg.device)
                prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                    (prompt_token_type, word_batch, image_batch)
                )

                # 推論キャッシュの初期化
                inference_cache["obs_tokens"] = []
                inference_cache["obs_masks"] = []
                inference_cache["action_tokens"] = []

            # 観測データを準備
            obs["ee"] = np.asarray(obs["ee"])
            obs = add_batch_dim(obs)
            # print("Before prepare_obs, obs keys:", obs.keys())
            obs_copy = copy.deepcopy(obs)
            # obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
            #     device=cfg.device
            # )
            obs = prepare_obs(obs=obs_copy, rgb_dict=None, meta=meta_info).to_torch_tensor(
                device=cfg.device
            )
            obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
            obs_token_this_step = obs_token_this_step.squeeze(0)
            obs_mask_this_step = obs_mask_this_step.squeeze(0)
            inference_cache["obs_tokens"].append(obs_token_this_step[0])
            inference_cache["obs_masks"].append(obs_mask_this_step[0])

            # 観測トークンを整列して準備
            max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
            obs_tokens_to_forward, obs_masks_to_forward = [], []
            obs_tokens_this_env, obs_masks_this_env = [], []
            for idx in range(len(inference_cache["obs_tokens"])):
                obs_this_env_this_step = inference_cache["obs_tokens"][idx]
                obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
                required_pad = max_objs - obs_this_env_this_step.shape[0]
                obs_tokens_this_env.append(
                    any_concat(
                        [
                            obs_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                obs_this_env_this_step.shape[1],
                                device=cfg.device,
                                dtype=obs_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
                obs_masks_this_env.append(
                    any_concat(
                        [
                            obs_mask_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                device=cfg.device,
                                dtype=obs_mask_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
            obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
            obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
            obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
            obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
            obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
            obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

            if elapsed_steps == 0:
                action_tokens_to_forward = None
            else:
                action_tokens_to_forward = any_stack(
                    [any_stack(inference_cache["action_tokens"], dim=0)],
                    dim=0,
                )
                action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)

            # ポリシーを使用して次のアクションを推論
            predicted_action_tokens = policy.forward(
                obs_token=obs_tokens_to_forward,
                action_token=action_tokens_to_forward,
                prompt_token=prompt_tokens,
                prompt_token_mask=prompt_masks,
                obs_mask=obs_masks_to_forward,
            )  # (L, B, E)
            predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(
                0
            )  # (1, B, E)
            dist_dict = policy.forward_action_decoder(predicted_action_tokens)

            # アクションの選択と調整
            actions = {k: v.mode() for k, v in dist_dict.items()}
            action_tokens = policy.forward_action_token(actions)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            inference_cache["action_tokens"].append(action_tokens[0])
            actions = policy._de_discretize_actions(actions)
            action_bounds = [meta_info["action_bounds"]]
            action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
            action_bounds_high = [
                action_bound["high"] for action_bound in action_bounds
            ]
            action_bounds_low = np.asarray(action_bounds_low)
            action_bounds_high = np.asarray(action_bounds_high)
            action_bounds_low = torch.tensor(
                action_bounds_low, dtype=torch.float32, device=cfg.device
            )
            action_bounds_high = torch.tensor(
                action_bounds_high, dtype=torch.float32, device=cfg.device
            )
            actions["pose0_position"] = (
                actions["pose0_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose1_position"] = (
                actions["pose1_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose0_position"] = torch.clamp(
                actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose1_position"] = torch.clamp(
                actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
            actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
            actions["pose0_rotation"] = torch.clamp(
                actions["pose0_rotation"], min=-1, max=1
            )
            actions["pose1_rotation"] = torch.clamp(
                actions["pose1_rotation"], min=-1, max=1
            )
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            actions = any_slice(actions, np.s_[0, 0])
            # obs, _, done, info = env.step(actions)

            # 環境にアクションを送信して次の状態を取得
            step_result = env.step(actions)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                observation, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected number of values returned from step: {len(step_result)}")
            print(done)
            elapsed_steps += 1
            if done:
                break


def prepare_prompt(*, prompt: str, prompt_assets: dict, views: list[str]):
    # 視点をソートして固定
    views = sorted(views)

    # プロンプトをトークナイズ
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens

    # プロンプト内のプレースホルダーがプロンプトアセットと一致していることを確認
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            # プレースホルダーでない場合はそのまま追加
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            # プレースホルダーの場合は対応するアセットを取得して処理
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]

            # オブジェクトの表現を準備
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                # 各視点のRGB画像とセグメンテーションマスクを処理
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    # オブジェクトのバウンディングボックスを計算
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])

                    # オブジェクトの画像を切り抜き
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        # 正方形でない場合はパディングを追加
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"

                    # リサイズして保存
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)

                # バウンディングボックスと切り抜き画像を保存
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]

    # 各視点で最大オブジェクト数を計算
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                # トークンが整数の場合（通常の単語トークン）
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                # トークンが辞書の場合（オブジェクトトークン）
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # マスクを追加
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=np.bool)
                    for view in views
                }
                # パディングを計算
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=np.bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)

    # 検証：トークン数の整合制
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)

    # トークンをスタック
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    # トーチテンソルに変換
    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()

    return raw_prompt_token_type, word_batch, image_batch


def prepare_obs(
    *,
    obs: dict, # 入力観測データ
    rgb_dict: dict | None = None, # RGBデータの辞書
    meta: dict, # メタ情報
):
    # print("Initial rgb_dict:", rgb_dict)  # デバッグ出力
    # print("Initial obs keys:", obs.keys())  # デバッグ出力
    assert not (rgb_dict is not None and "rgb" in obs)

    # rgb_dictがNoneの場合、obsから"rgb"キーを取り出して代入
    rgb_dict = rgb_dict or obs.pop("rgb")

    # rgb_dictの状態をデバッグ用に出力
    # print("After assignment rgb_dict:", rgb_dict)

    # obsから"segm"キーを取り出してセグメンテーションデータとして保持
    segm_dict = obs.pop("segm")

    # RGBビューのキーをソートしてリスト化
    views = sorted(rgb_dict.keys())

    # メタ情報の整合制をチェック
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys()) # オブジェクトIDのリストを取得

    # バッチサーズを取得
    L_obs = get_batch_size(obs)

    # 初期化された観測データリストを作成
    obs_list = {
        "ee": obs["ee"], # エフェクター情報
        "objects": {
            "cropped_img": {view: [] for view in views}, # 各ビューの切り抜き画像
            "bbox": {view: [] for view in views}, # 各ビューのバウンディングボックス
            "mask": {view: [] for view in views}, # 各ビューのマスク情報
        },
    }

    # 各観測ステップに対する処理
    for l in range(L_obs):
        # 現在のステップとRGBセグメンテーションデータを取得
        rgb_dict_this_step = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = any_slice(segm_dict, np.s_[l])

        # ビューごとの処理
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]

            # バウンディングボックスと切り抜き画像の初期化
            bboxes = []
            cropped_imgs = []
            n_pad = 0 # パディング数のカウント

            # 各オブジェクトIDに対する処理
            for obj_id in objects:
                # セグメンテーションマスクから現在のオブジェクトの座標を取得
                ys, xs = np.nonzero(segm_this_view == obj_id)

                # 有効な領域がない場合はパディング数を増加
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue

                # バウンディングボックスの計算
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])

                # 切り抜き画像を取得
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]

                # 正方形でない場合はパディングを適用
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    # パディング後に正方形になっていることを確認
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"

                # 画像の形式を変換し，リサイズ
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                # リサイズ後の画像をリストに追加
                cropped_imgs.append(cropped_img)

            # バウンディングボックスと切り抜き画像をNumpy配列に変換
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)

            # マスクを初期化
            mask = np.ones(len(bboxes), dtype=bool)

            # パディングが必要な場合，ゼロパディングを適用
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, 32, 32),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)

            # 各ビューごとに観測リストを更新
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)

    # バッチ次元を追加
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    # 最終的な観測データを変換し，PyTorchテンソル形式に変更
    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs) # 軸の順序を調整
    return obs


class ResetFaultToleranceWrapper(Wrapper):
    # 最大リトライ回数を設定
    max_retries = 10

    def __init__(self, env):
        super().__init__(env) # 親クラスの初期化を呼び出す

    def reset(self):
        # 環境のリセットを最大max_retries回試行
        for _ in range(self.max_retries):
            try:
                # 環境のリセットを試行
                return self.env.reset()
            except:
                # リセットに失敗した場合，シード値を変更
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        # リトライがずべて失敗した場合はエラーをスロー
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        # 環境にタイムリミットを設定する
        # デフォルトに最大ステップ数に加え，ボーナスステップを追加
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cpu")
    arg = arg.parse_args()
    main(arg)
