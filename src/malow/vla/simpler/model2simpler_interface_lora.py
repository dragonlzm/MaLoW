"""
model2simpler_interface_lora.py

Interface for MaLoW VLA with LoRA adapters for SimplerEnv evaluation.
Supports loading normalization stats and config from LoRA checkpoint structure.

This is a modified version of model2simpler_interface.py that uses the LoRA
checkpoint loading pattern (config.yaml/json + norm_stats.json) instead of
read_mode_config.
"""

from collections import deque
from typing import Optional, Sequence, Dict
import os
import json
import numpy as np

from malow.vla.simpler.tools.websocket_policy_client import WebsocketClientPolicy

from malow.vla.simpler.adaptive_ensemble import AdaptiveEnsembler
from pathlib import Path


class M1InferenceLora:
    """
    Inference interface for MaLoW VLA with LoRA checkpoints for SimplerEnv.

    This class handles the LoRA checkpoint structure where:
    - config.yaml/config.json is in the checkpoint directory
    - norm_stats.json/dataset_statistics.json may be in:
      - The checkpoint directory itself
      - The parent run directory (for older checkpoints)
    """

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble=True,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10093,
        output_prior_action: bool = False,
    ) -> None:
        # Build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)

        # Store output_prior_action flag
        self.output_prior_action = output_prior_action
        if output_prior_action:
            print("*** Using prior action from VLM instead of flow model ***")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = unnorm_key  # resolved later after loading norm_stats
            self._default_unnorm_keys = ["bridge_dataset", "oxe_bridge"]
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = unnorm_key  # resolved later after loading norm_stats
            self._default_unnorm_keys = ["fractal20220817_data", "oxe_rt1"]
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. "
                f"The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        self.cfg_scale = cfg_scale
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Load config and stats from LoRA checkpoint
        self.model_config, self.norm_stats = self._load_lora_checkpoint_config(policy_ckpt_path)
        self.action_norm_stats = self._get_action_stats(self.unnorm_key)

    def _load_lora_checkpoint_config(self, checkpoint_path: str) -> tuple:
        """
        Load config and normalization stats from a LoRA checkpoint.

        LoRA checkpoint structure:
            checkpoint_path/
            ├── config.yaml or config.json
            ├── lora_config.json
            ├── norm_stats.json or dataset_statistics.json (may be in parent dir)
            └── ...

        Args:
            checkpoint_path: Path to the LoRA checkpoint directory

        Returns:
            tuple: (model_config dict, norm_stats dict)
        """
        checkpoint_path = Path(checkpoint_path)

        # 1. Load model config
        config_path = checkpoint_path / "vla_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            config_path = checkpoint_path / "config.yaml"
            model_config = None
        if config_path.exists():
            if model_config is None:
                from omegaconf import OmegaConf

                cfg = OmegaConf.load(config_path)
                model_config = OmegaConf.to_container(cfg, resolve=True)
        else:
            config_path = checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    model_config = json.load(f)
            else:
                raise FileNotFoundError(f"No config.yaml or config.json found in {checkpoint_path}")

        print(f"✅ Loaded config from {config_path}")

        # 2. Load normalization stats from multiple possible locations
        norm_stats = None
        norm_stats_locations = [
            checkpoint_path / "norm_stats.json",
            checkpoint_path / "dataset_statistics.json",
            checkpoint_path.parent.parent / "norm_stats.json",
            checkpoint_path.parent.parent / "dataset_statistics.json",
            checkpoint_path.parent / "dataset_statistics.json",
            checkpoint_path.parent / "norm_stats.json",
        ]

        for stats_path in norm_stats_locations:
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    norm_stats = json.load(f)
                # Convert lists back to numpy arrays
                for key, value in norm_stats.items():
                    if isinstance(value, dict):
                        norm_stats[key] = {k: np.array(v) if isinstance(v, list) else v for k, v in value.items()}
                    elif isinstance(value, list):
                        norm_stats[key] = np.array(value)
                print(f"✅ Loaded normalization stats from {stats_path}")
                break

        if norm_stats is None:
            raise FileNotFoundError(
                f"Normalization stats not found in any expected location. "
                f"Checked: {[str(p) for p in norm_stats_locations]}"
            )

        return model_config, norm_stats

    def _get_action_stats(self, unnorm_key: Optional[str] = None) -> dict:
        """Get action normalization statistics."""
        unnorm_key = self._resolve_unnorm_key(self.norm_stats, unnorm_key)
        self.unnorm_key = unnorm_key  # store the resolved key for later use
        print(f"✅ Resolved unnorm_key: {unnorm_key}")
        return self.norm_stats[unnorm_key]["action"]

    def _resolve_unnorm_key(self, norm_stats: dict, unnorm_key: Optional[str]) -> str:
        """
        Resolve the unnormalization key against the loaded norm_stats.

        If unnorm_key is explicitly provided and exists, use it directly.
        Otherwise, try each candidate in _default_unnorm_keys and return the
        first match found in norm_stats.
        """
        # If explicitly provided and it exists, use it
        if unnorm_key is not None and unnorm_key in norm_stats:
            return unnorm_key

        # Try each default alias for this policy_setup
        for candidate in self._default_unnorm_keys:
            if candidate in norm_stats:
                return candidate

        # If only one key exists, use it
        if len(norm_stats) == 1:
            return next(iter(norm_stats.keys()))

        raise KeyError(
            f"Could not resolve unnorm_key. "
            f"Tried explicit key={unnorm_key!r} and defaults={self._default_unnorm_keys}, "
            f"but norm_stats only has: {list(norm_stats.keys())}"
        )

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))

        image = self._resize_image(image)
        vla_input = {
            "batch_images": [[image]],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "cfg_scale": self.cfg_scale,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        # Subclass extension point; the base returns {} so the wire payload is unchanged.
        vla_input.update(self._extra_infer_payload())

        response = self.client.infer(vla_input)

        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
        normalized_actions = normalized_actions[0]

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
        )

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        from transforms3d.euler import euler2axangle

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        # Subclass extension point, called once the executed action is final; the base
        # does nothing. `raw_actions[0]` is what the environment acts on this step: the
        # ensembler's blend when ensembling is on, chunk step 0 when it is off.
        self._on_action_executed(raw_actions[0], raw_action, action)
        return raw_action, action

    # --- subclass hooks -------------------------------------------------------------

    def _extra_infer_payload(self) -> Dict:
        """Extra keys to merge into the `infer` payload. Empty here: no protocol change."""
        return {}

    def _on_action_executed(self, executed_raw_action: np.ndarray, raw_action: Dict, action: Dict) -> None:
        """Called with the action the environment is about to receive. No-op here."""
        return None

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def normalize_actions(actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Inference compatibility implementation; see SOURCE_NOTICE.md."""
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        denom = np.where(action_high != action_low, action_high - action_low, 1.0)
        normalized = np.where(mask, 2.0 * (actions - action_low) / denom - 1.0, actions)
        return np.clip(normalized, -1.0, 1.0)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        import cv2 as cv

        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        import matplotlib.pyplot as plt

        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
        """Check and return the unnormalization key."""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
