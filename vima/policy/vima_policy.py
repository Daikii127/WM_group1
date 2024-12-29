from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import vima.nn as vnn
from ..utils import *

# VIMAPolicyクラス: 強化学習やロボティクスシステムで使用するためのポリシーネットワークを定義
class VIMAPolicy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        xf_n_layers: int,
        sattn_n_heads: int,
        xattn_n_heads: int,
    ):
        super().__init__()

        # 埋め込み次元数を設定
        self.embed_dim = embed_dim

        # Cross Attention Transformerモデルの定義
        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=xf_n_layers,
            n_head=sattn_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )

        # オブジェクトエンコーダ: 観測されたオブジェクトの情報を抽出
        self.obj_encoder = vnn.ObjEncoder(
            transformer_emb_dim=embed_dim,
            views=["front", "top"],
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
        )

        # エンドエフェクター（ロボットの手先）のエンコーダ
        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, embedding_dim=2)

        # 観測データを統合するための層
        self.obs_fusion_layer = nn.Linear(self.obj_encoder.output_dim + 2, embed_dim)

        # アクションエンコーダ: アクションデータを埋め込みベクトルに変換
        self.action_encoder = vnn.ActionEmbedding(
            output_dim=embed_dim,
            embed_dict={
                "pose0_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            },
        )

        # アクションデコーダ: 埋め込みベクトルから具体的なアクションを生成
        self.action_decoder = vnn.ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        # テキストプロンプト用の埋め込み層
        self.prompt_embedding = vnn.WordEmbedding()

        # T5モデルを使用したプロンプトエンコーダ
        self.t5_prompt_encoder = vnn.T5PromptEncoder()

        # T5出力を埋め込み次元に変換するための層
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )

        # オブジェクトプロンプトの埋め込み後処理用層
        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )

        # ビューや離散化に使用するパラメータを定義
        self._views = ["front", "top"]
        self._n_discrete_x_bins = 50
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 50
        self._n_discrete_rot_bins = 50

    # ポリシーネットワークの順伝播処理
    def forward(
        self,
        obs_token: torch.Tensor,
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ):

        # オブジェクトトークンを保存
        saved_obj_token = obs_token.clone()
        flattened_saved_obj_token = saved_obj_token.reshape(saved_obj_token.shape[0], saved_obj_token.shape[1], -1)

        # 観測トークンとアクショントークンを統合して順伝播を実行
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        n_max_objs = obs_token.shape[-2]
        L = L_obs * n_max_objs + L_action

        # トークンおよびマスク用のテンソルを初期化
        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(L, B, dtype=torch.bool, device=obs_token.device)

        # 観測トークンとマスクを整形
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")

        # 観測トークンとマスクを統合
        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs :: n_max_objs + 1] = action_token

        # トークンの位置IDを計算
        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1

        # Cross Attention Transformerにトークンを入力
        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        # 予測されたアクショントークンを取得
        predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]
        residual_output = predicted_action_tokens
        return predicted_action_tokens

    # プロンプトの組み立てを行う順伝播処理
    def forward_prompt_assembly(self, prompts):
        # 入力プロンプト（トークンタイプ，単語パッチ，画像パッチを展開）
        raw_prompts_token_type, word_batch, image_batch = prompts

        # 単語パッチを埋め込みベクトルに変換
        batch_word_emb = self.prompt_embedding(word_batch)

        # 画像パッチをオブジェクトエンコーダに通し，埋め込みベクトルに変換
        batch_image_emb = self.obj_encoder(**image_batch)

        # オブジェクトエンコーダの出力をさらにMLPで処理
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)

        # 各画像が持つ最大オブジェクト数を取得
        n_max_objs = batch_image_emb.shape[-2]

        # プロンプト中で必要な最大トークン数を計算
        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0: # 単語トークン
                    L_this += 1
                elif item == 1: # オブジェクトトークン
                    L_this += n_max_objs
                else: # 無効なトークンタイプの場合はエラー
                    raise ValueError(f"Invalid prompt token type {item}")

            # 最大トークン数を更新
            L_max = max(L_max, L_this)

        # プロンプトトークンとそのマスクを格納するリストを初期化
        prompt_tokens, prompt_masks = [], []

        # 単語と画像パッチのポインタを初期化
        word_ptr, img_ptr = 0, 0

        # 各プロンプトを順に処理してトークンを組み立て
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = [] # トークンを格納
            assembled_mask = [] # マスクを格納
            for item in raw_prompt: 
                if item == 0: # 単語トークンの場合
                    assembled_prompt.append(batch_word_emb[word_ptr]) # 単語埋め込みを追加
                    word_ptr += 1 # 単語ポインタを勧める
                    assembled_mask.append(True) # マスクを有効に設定
                elif item == 1: # オブジェクトトークンの場合
                    # 各ビューのマスクを連結
                    obj_mask = any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    # オブジェクト埋め込みとマスクを追加
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1 # 画像ポインタを進める
                else: # 無効なトークンタイプの場合はエラー
                    raise ValueError(f"Invalid type: {type(item)}")

            # プロンプトを最大長に合わせてパディング
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=assembled_prompt.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        any_to_torch_tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                        torch.zeros(
                            num_padding,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        # プロンプトトークンとマスクをテンソル化してスタック
        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)

        # 時間次元とパッチ次元を入れ替え
        prompt_tokens = prompt_tokens.transpose(0, 1)

        saved_obj_tokens = prompt_tokens.clone()  # 全体からコピー
        obj_mask_indices = (torch.tensor(raw_prompts_token_type) == 1).nonzero()

        # T5プロンプトエンコーダを使用する場合
        if self.t5_prompt_encoder is not None:
            # プロンプトトークンをエンコーダに入力
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )

            # オブジェクト部分だけ残差接続を適用
            for idx in obj_mask_indices:
                prompt_tokens[idx] += saved_obj_tokens[idx]

            # 出力を埋め込み次元に変換
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)

        # プロンプトトークンとそのマスクを返す
        return prompt_tokens, prompt_masks

    # 観測データからトークンを生成
    def forward_obs_token(self, obs):
        # 観測データからオブジェクトデータとエンドエフェクターデータを取得
        objects, ee = obs["objects"], obs["ee"]

        # エンドエフェクターの先頭次元を取得（バッチサイズと時系列長）
        leading_dims = ee.shape[:2]

        # オブジェクトデータの構造を変更して2次元目以降を平坦化
        objects = objects.map_structure(func=lambda x: x.reshape(-1, *x.shape[2:]))

        # オブジェクトデータをオブジェクトエンコーダに通して埋め込み特徴を取得
        img_feats = self.obj_encoder(**objects)

        # 埋め込み特徴を元の形に再構成
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])

        # オブジェクトごとのマスクを取得し，再構成
        obj_mask = {
            k: objects["mask"][k].reshape(*leading_dims, -1) for k in objects["mask"]
        }

        # エンドエフェクターのデータを埋め込みベクトルに変換
        ee_feats = self.end_effector_encoder(ee)

        # 埋め込みベクトルをオブジェクト数に合わせて繰り返し
        ee_feats = ee_feats.unsqueeze(2).repeat(1, 1, img_feats.shape[-2], 1)

        # オブジェクトの埋め込み特徴とエンドエフェクターの特徴を結合して統合的な観測特徴を生成
        obs_feats = self.obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))

        # 複数のビューにわたるオブジェクトマスクを結合
        obj_mask = any_concat([obj_mask[view] for view in sorted(self._views)], dim=-1)

        # 統合された観測特徴とマスクを返す
        return obs_feats, obj_mask

    # アクションデータからトークンを生成
    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    # アクショントークンをデコードしてアクションに変換
    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    # アクションを離散化する処理
    def discretize_action(self, action):
        device = action["pose0_position"].device
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    # 離散化されたアクションをもとに戻す処理
    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions
