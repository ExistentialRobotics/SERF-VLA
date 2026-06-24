import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import random
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.learning.serf_b1k_utils import (
    collect_initial_predicate_states,
    compute_step_q_score,
    jsonable_step_q_score,
    should_record_step_q_score,
    task_name_to_task_id,
)
from omnigibson.macros import gm, create_module_macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Dict, Optional, Tuple, List

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info

DEFAULT_THIRD_PERSON_VIEW_CONFIG = {
    "name": "third_person_cam",
    "relative_prim_path": "/third_person_cam",
    "resolution": (1800, 3200),
    "horizontal_aperture": 40,
    "position": (22.3, 6.8, 2.6),
    "orientation": (0.0, 0.0, 0.0, 0.1),
}

_COMMON_THIRD_PERSON_VIEW_FIELDS = {
    "name": "third_person_cam",
    "relative_prim_path": "/third_person_cam",
    "resolution": (1800, 3200),
    "horizontal_aperture": 40,
}

# Task-specific third-person camera overrides.
THIRD_PERSON_VIEW_CONFIGS: dict[str, dict[str, Any]] = {
    "task-0021": {
        **_COMMON_THIRD_PERSON_VIEW_FIELDS,
        "position": (22.3, 7.0500, 2.7),
        "orientation": (0.0, 0.0, 0.0, 1.0),
    },
    "task-0022": {
        **_COMMON_THIRD_PERSON_VIEW_FIELDS,
        "position": (0.2344, 7.6898, 1.3903),
        "orientation": (-0.0040, 0.5023, 0.8646, -0.0068),
    },
    "task-0026": {
        **_COMMON_THIRD_PERSON_VIEW_FIELDS,
        "position": (1.6142, 5.2129, 2.2912,),
        "orientation": (-0.2879, 0.2954, 0.6524, -0.6358),
    },
}

def get_third_person_view_config(task_name: str) -> Dict[str, Any]:
    task_id = task_name_to_task_id(task_name)
    return dict(THIRD_PERSON_VIEW_CONFIGS.get(task_id, DEFAULT_THIRD_PERSON_VIEW_CONFIG))


def get_third_person_sensor_kwargs(view_config: Dict[str, Any]) -> Dict[str, Any]:
    sensor_kwargs = dict(
        image_height=view_config["resolution"][0],
        image_width=view_config["resolution"][1],
        horizontal_aperture=view_config["horizontal_aperture"],
    )
    if "focal_length" in view_config:
        sensor_kwargs["focal_length"] = view_config["focal_length"]
    return sensor_kwargs


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    return value


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_to_jsonable(payload), f, cls=TorchEncoder)
    os.replace(tmp_path, path)


def resolve_eval_output_paths(
    config: DictConfig,
    *,
    include_tracking_video: bool = False,
) -> Tuple[Path, Optional[Path], bool, bool]:
    write_video = bool(getattr(config, "write_video", False))
    write_third_person_video = bool(getattr(config, "write_third_person_video", False))
    should_make_video_dir = write_video or write_third_person_video or include_tracking_video

    video_path = None
    if should_make_video_dir:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    return metrics_path, video_path, write_video, write_third_person_video


def resolve_instances_to_run(
    config: DictConfig,
    *,
    num_train_instances: int,
    num_eval_instances: int,
) -> List[int]:
    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."

    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run: List[int] = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(int((episode["episode_index"] // 10) % 1e3))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(num_train_instances))
            ), f"eval instance ids must be in range({num_train_instances})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
        return instances_to_run

    instances_to_run = (
        config.eval_instance_ids if config.eval_instance_ids is not None else set(range(num_eval_instances))
    )
    assert set(instances_to_run).issubset(
        set(range(num_eval_instances))
    ), f"eval instance ids must be in range({num_eval_instances})"

    if config.test_hidden:
        return [int(instance_id) for instance_id in instances_to_run]

    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]
    assert (
        lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
    ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
    test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
    return [int(test_instances[i]) for i in instances_to_run]


def log_episode_summary(
    logger_: logging.Logger,
    evaluator: "Evaluator",
    episode_outputs: Dict[str, Any],
    *,
    write_video: bool,
    write_third_person_video: bool,
) -> None:
    logger_.info(f"Evaluation finished at step {evaluator.env._current_step}.")
    logger_.info(
        f"Evaluation exit state: {episode_outputs['terminated']}, {episode_outputs['truncated']}"
    )
    logger_.info(f"Total trials: {evaluator.n_trials}")
    logger_.info(f"Total success trials: {evaluator.n_success_trials}")
    if episode_outputs.get("video") is not None:
        logger_.info(f"Saved video to {episode_outputs['video']}")
    if episode_outputs.get("third_person_video") is not None:
        logger_.info(f"Saved video to {episode_outputs['third_person_video']}")
    if not write_video and not write_third_person_video:
        logger_.warning("No observations were recorded.")

class Evaluator:
    """
    Evaluator class for running and evaluating policies for behavior task.
    This class manages the setup, execution, and evaluation of policy rollouts in OmniGibson environment,
    tracking metrics such as the number of trials, successes, and total time. It supports loading environments,
    robots, policies, and metrics, and provides methods for stepping through the environment, resetting state,
    and handling video outputs and loggings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.third_person_view_config = get_third_person_view_config(self.cfg.task.name)

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None
        self._third_person_video_writer = None
        self._step_stats_handle = None
        self._step_stats_path = None
        self._episode_initial_predicate_states = None
        self._last_base_position = None
        self._cumulative_base_distance = 0.0
        self._repro_root = Path(self.cfg.log_path).expanduser() / "repro"
        self._episode_initial_state_path = None
        self._episode_initial_state_manifest_path = None
        self._episode_action_dir = None
        self._episode_action_manifest_path = None
        self._episode_action_manifest_meta = None
        self._action_chunk_buffer = None
        self._action_chunk_buffer_count = 0
        self._action_shape = None
        self._action_chunk_index = 0
        self._recorded_action_count = 0
        self._replay_action_enabled = False
        self._replay_action_dir = None
        self._replay_manifest = None
        self._replay_until_step = None
        self._replay_chunk_index = 0
        self._replay_current_chunk = None
        self._replay_current_chunk_offset = 0
        self._replay_loaded_action_count = 0

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        third_person_view_config = self.third_person_view_config

        cfg["env"]["external_sensors"] = [
            dict(
                sensor_type="VisionSensor",
                relative_prim_path=third_person_view_config["relative_prim_path"],
                name=third_person_view_config["name"],
                modalities=["rgb"],
                sensor_kwargs=get_third_person_sensor_kwargs(third_person_view_config),
                include_in_obs=False,
                position=list(third_person_view_config["position"]),
                orientation=list(third_person_view_config["orientation"])
            )
        ]

        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """
        Loads and returns the policy instance.
        """
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.

        Returns:
            Tuple[bool, bool]:
                - terminated (bool): Whether the episode has terminated (i.e., reached a terminal state).
                - truncated (bool): Whether the episode was truncated (i.e., stopped due to a time limit or other constraint).

        Workflow:
            1. Computes the next action using the policy based on the current observation.
            2. Steps the environment with the computed action and retrieves the next observation,
               termination and truncation flags, and additional info.
            3. If the episode has ended (terminated or truncated), increments the trial counter and
               updates the count of successful trials if the task was completed successfully.
            4. Preprocesses the new observation.
            5. Invokes step callbacks for all registered metrics to update their state.
            6. Returns the termination and truncation status.
        """
        self.robot_action = self._select_action_for_step()

        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        return self._finalize_step(obs, terminated, truncated, info)

    def _finalize_step(self, obs: dict, terminated: bool, truncated: bool, info: dict) -> Tuple[bool, bool]:
        self._record_step_stats(terminated=terminated, truncated=truncated, info=info)
        self._after_env_step(info)
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated

    def _select_action_for_step(self) -> Any:
        step_idx = int(self.env._current_step)
        if self._replay_action_enabled:
            replay_until_step = self._replay_until_step
            if replay_until_step is not None and step_idx >= replay_until_step:
                logger.info(
                    f"Stopping replay at configured replay_until_step={replay_until_step}; switching to policy inference."
                )
                self._disable_replay_mode()
            else:
                try:
                    self.robot_action = self._load_replay_action(step_idx)
                    return self.robot_action
                except RuntimeError as exc:
                    if not self._is_replay_fallback_error(exc):
                        raise
                    logger.warning(
                        f"Replay stopped at env step {step_idx}: {exc}. Falling back to policy inference."
                    )
                    self._disable_replay_mode()

        action = self.policy.forward(obs=self.obs)
        self._record_action_step(step_idx=step_idx, action=action)
        self.robot_action = action
        return self.robot_action

    def _after_env_step(self, info: dict) -> None:
        """Subclass hook for consuming env-step info without overriding step()."""

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        self._close_video_writer(self._video_writer)
        self._video_writer = video_writer

    @property
    def third_person_video_writer(self) -> Tuple[Container, Stream]:
        return self._third_person_video_writer

    @third_person_video_writer.setter
    def third_person_video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        self._close_video_writer(self._third_person_video_writer)
        self._third_person_video_writer = video_writer

    @staticmethod
    def _close_video_writer(video_writer: Optional[Tuple[Container, Stream]]) -> None:
        if video_writer is None:
            return
        container, stream = video_writer
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # append task id to obs
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _write_third_person_video(self) -> None:
        third_person_view_config = self.third_person_view_config
        frame = None
        external_sensors = getattr(self.env, "external_sensors", None)
        if external_sensors:
            third_person_sensor = external_sensors.get(third_person_view_config["name"])
            if third_person_sensor is not None:
                og.sim.render()
                frame = third_person_sensor.get_obs()[0].get("rgb")
        if frame is None:
            frame = self.env.render()
        if frame is None:
            return
        if isinstance(frame, th.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = cv2.resize(frame, third_person_view_config["resolution"])
        write_video(
            np.expand_dims(frame, 0),
            video_writer=self.third_person_video_writer,
            batch_size=1,
            mode="rgb",
        )

    def start_episode_logging(self, metrics_path: Path, instance_id: int, episode_id: int) -> None:
        self.close_episode_logging()
        self._step_stats_path = metrics_path / f"{self.cfg.task.name}_{instance_id}_{episode_id}_step_stats.jsonl"
        self._step_stats_handle = open(self._step_stats_path, "w")
        self._episode_initial_predicate_states = collect_initial_predicate_states(
            self.cfg,
            self.env.task.ground_goal_state_options,
        )
        self._last_base_position = self.robot.get_position_orientation()[0].detach().cpu().clone()
        self._cumulative_base_distance = 0.0

    def close_episode_logging(self) -> None:
        if self._step_stats_handle is not None:
            self._step_stats_handle.close()
        self._step_stats_handle = None
        self._step_stats_path = None
        self._episode_initial_predicate_states = None
        self._last_base_position = None
        self._cumulative_base_distance = 0.0

    def _compute_current_q_score(self) -> Optional[float]:
        if not should_record_step_q_score(self.cfg):
            return None
        return compute_step_q_score(
            cfg=self.cfg,
            task_success=bool(self.env.task.success),
            ground_goal_state_options=self.env.task.ground_goal_state_options,
            initial_predicate_states=self._episode_initial_predicate_states,
        )

    def _record_step_stats(self, terminated: bool, truncated: bool, info: dict) -> None:
        if self._step_stats_handle is None:
            return

        current_base_position = self.robot.get_position_orientation()[0].detach().cpu()
        if self._last_base_position is not None:
            base_delta = th.linalg.norm(current_base_position - self._last_base_position).item()
            self._cumulative_base_distance += float(base_delta)
        self._last_base_position = current_base_position.clone()

        q_score = self._compute_current_q_score()
        step_stats = {
            "step": int(self.env._current_step),
            "q_score": jsonable_step_q_score(q_score),
            "base_distance": {"cumulative": float(self._cumulative_base_distance)},
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("done", {}).get("success", False)),
        }
        self._step_stats_handle.write(json.dumps(step_stats) + "\n")
        self._step_stats_handle.flush()

    def should_write_video(self) -> bool:
        return bool(getattr(self.cfg, "write_video", False))

    def should_write_third_person_video(self) -> bool:
        return bool(getattr(self.cfg, "write_third_person_video", False))

    def should_use_repro_initial_state(self) -> bool:
        return bool(getattr(self.cfg, "repro_initial_state", True))

    def should_use_repro_action_log(self) -> bool:
        return bool(getattr(self.cfg, "repro_action_log", True))

    def repro_action_chunk_size(self) -> int:
        return max(1, int(getattr(self.cfg, "repro_action_chunk_size", 100)))

    def replay_mode(self) -> str:
        replay_mode = str(getattr(self.cfg, "replay_mode", "auto")).strip().lower()
        if replay_mode not in {"none", "auto", "force"}:
            raise ValueError(f"Invalid replay_mode={replay_mode!r}. Expected one of: none, auto, force.")
        return replay_mode

    def replay_action_log_path(self) -> Optional[Path]:
        replay_action_log_path = getattr(self.cfg, "replay_action_log_path", None)
        if replay_action_log_path in (None, ""):
            return None
        return Path(replay_action_log_path).expanduser()

    def replay_until_step(self) -> Optional[int]:
        replay_until_step = getattr(self.cfg, "replay_until_step", None)
        if replay_until_step is None:
            return None
        replay_until_step = int(replay_until_step)
        if replay_until_step < 0:
            raise ValueError(f"replay_until_step must be >= 0, got {replay_until_step}")
        return replay_until_step

    def resume_record_suffix(self) -> str:
        resume_record_suffix = str(getattr(self.cfg, "resume_record_suffix", "resumed")).strip()
        if not resume_record_suffix:
            raise ValueError("resume_record_suffix must be a non-empty string.")
        return resume_record_suffix

    def prepare_episode(self, instance_id: int, episode_id: int) -> None:
        """Hook for subclasses that need per-episode setup before reset()."""

    def _initial_state_path(self, instance_id: int, episode_id: int) -> Path:
        return self._default_action_artifact_dir(instance_id, episode_id) / "initial_state.npz"

    def _initial_state_manifest_path(self, instance_id: int, episode_id: int) -> Path:
        return self._default_action_artifact_dir(instance_id, episode_id) / "initial_state_manifest.json"

    def _default_action_artifact_dir(self, instance_id: int, episode_id: int) -> Path:
        return self._repro_root / "actions" / self.cfg.task.name / str(instance_id) / str(episode_id)

    def _manifest_path_from_dir(self, action_dir: Path) -> Path:
        return action_dir / "manifest.json"

    def _chunk_path_from_dir(self, action_dir: Path, chunk_idx: int) -> Path:
        return action_dir / f"chunk_{chunk_idx:06d}.npy"

    def _action_chunk_path(self, chunk_idx: int) -> Path:
        assert self._episode_action_dir is not None
        return self._chunk_path_from_dir(self._episode_action_dir, chunk_idx)

    def _reset_episode_repro_context(self) -> None:
        self._episode_initial_state_path = None
        self._episode_initial_state_manifest_path = None
        self._episode_action_dir = None
        self._episode_action_manifest_path = None
        self._episode_action_manifest_meta = None
        self._action_chunk_buffer = None
        self._action_chunk_buffer_count = 0
        self._action_shape = None
        self._action_chunk_index = 0
        self._recorded_action_count = 0
        self._replay_action_enabled = False
        self._replay_action_dir = None
        self._replay_manifest = None
        self._replay_until_step = None
        self._replay_chunk_index = 0
        self._replay_current_chunk = None
        self._replay_current_chunk_offset = 0
        self._replay_loaded_action_count = 0

    def _remove_path_tree(self, path: Path) -> None:
        if not path.exists():
            return
        if path.is_file() or path.is_symlink():
            path.unlink()
            return
        for child in path.iterdir():
            if child.is_dir():
                self._remove_path_tree(child)
            else:
                child.unlink()
        path.rmdir()

    def _clear_action_artifacts(self, action_dir: Path) -> None:
        """Remove action stream files while preserving colocated initial-state artifacts."""
        manifest_path = self._manifest_path_from_dir(action_dir)
        if manifest_path.exists():
            manifest_path.unlink()
        for chunk_path in action_dir.glob("chunk_*.npy"):
            if chunk_path.is_file() or chunk_path.is_symlink():
                chunk_path.unlink()

    def _configure_repro_paths(self, instance_id: int, episode_id: int) -> None:
        self._reset_episode_repro_context()
        self._episode_action_dir = self._default_action_artifact_dir(instance_id, episode_id)
        self._episode_action_manifest_path = self._manifest_path_from_dir(self._episode_action_dir)
        self._set_episode_initial_state_paths(self._episode_action_dir)

    def _set_episode_initial_state_paths(self, state_dir: Path) -> None:
        self._episode_initial_state_path = state_dir / "initial_state.npz"
        self._episode_initial_state_manifest_path = state_dir / "initial_state_manifest.json"

    def _resolve_episode_initial_state_dir(self) -> Optional[Path]:
        if self._replay_action_enabled and self._replay_action_dir is not None:
            return self._replay_action_dir
        return self._episode_action_dir

    def prepare_episode_artifacts(self, instance_id: int, episode_id: int) -> None:
        self._configure_repro_paths(instance_id, episode_id)
        self._prepare_episode_action_artifacts(instance_id, episode_id)
        initial_state_dir = self._resolve_episode_initial_state_dir()
        if initial_state_dir is not None:
            self._set_episode_initial_state_paths(initial_state_dir)
        self._prepare_robot_initial_state(instance_id=instance_id, episode_id=episode_id)

    def _refresh_obs_from_env(self) -> None:
        self.obs = self._preprocess_obs(self.env.get_obs()[0])

    def _capture_rng_state(self) -> Dict[str, Any]:
        np_state = np.random.get_state()
        rng_state: Dict[str, Any] = {
            "python_random": list(random.getstate()),
            "numpy_random": {
                "bit_generator": np_state[0],
                "state": np_state[1].tolist(),
                "pos": int(np_state[2]),
                "has_gauss": int(np_state[3]),
                "cached_gaussian": float(np_state[4]),
            },
            "torch_cpu": th.random.get_rng_state().detach().cpu().tolist(),
        }
        if th.cuda.is_available():
            rng_state["torch_cuda"] = [
                state.detach().cpu().tolist() for state in th.cuda.get_rng_state_all()
            ]
        return rng_state

    def _restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        python_state = rng_state.get("python_random")
        if python_state is not None:
            version, internal_state, gaussian = python_state
            random.setstate((version, tuple(internal_state), gaussian))

        np_state = rng_state.get("numpy_random")
        if np_state is not None:
            np.random.set_state(
                (
                    np_state["bit_generator"],
                    np.asarray(np_state["state"], dtype=np.uint32),
                    int(np_state["pos"]),
                    int(np_state["has_gauss"]),
                    float(np_state["cached_gaussian"]),
                )
            )

        torch_cpu_state = rng_state.get("torch_cpu")
        if torch_cpu_state is not None:
            th.random.set_rng_state(th.as_tensor(torch_cpu_state, dtype=th.uint8))

        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and th.cuda.is_available():
            th.cuda.set_rng_state_all(
                [th.as_tensor(state, dtype=th.uint8) for state in torch_cuda_state]
            )

    def _write_initial_sim_state(self, path: Path, state: th.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
        state_np = np.asarray(state.detach().cpu().numpy())
        np.savez_compressed(tmp_path, state=state_np)
        os.replace(tmp_path, path)

    def _capture_initial_state_manifest(
        self,
        *,
        instance_id: int,
        episode_id: int,
        state_size: int,
    ) -> Dict[str, Any]:
        return {
            "task": self.cfg.task.name,
            "instance_id": int(instance_id),
            "episode_id": int(episode_id),
            "state_format": "og.sim.dump_state(serialized=True)",
            "state_size": int(state_size),
            "rng_state": self._capture_rng_state(),
        }

    def _capture_full_initial_state(self, *, instance_id: int, episode_id: int) -> None:
        assert self._episode_initial_state_path is not None
        assert self._episode_initial_state_manifest_path is not None
        state = og.sim.dump_state(serialized=True)
        self._write_initial_sim_state(self._episode_initial_state_path, state)
        _atomic_write_json(
            self._episode_initial_state_manifest_path,
            self._capture_initial_state_manifest(
                instance_id=instance_id,
                episode_id=episode_id,
                state_size=len(state),
            ),
        )

    def _restore_full_initial_state(self) -> None:
        assert self._episode_initial_state_path is not None
        payload = np.load(self._episode_initial_state_path, allow_pickle=False)
        state = th.as_tensor(payload["state"])
        og.sim.load_state(state, serialized=True)

        if (
            self._episode_initial_state_manifest_path is not None
            and self._episode_initial_state_manifest_path.exists()
        ):
            with open(self._episode_initial_state_manifest_path, "r") as f:
                manifest = json.load(f)
            rng_state = manifest.get("rng_state")
            if rng_state is not None:
                self._restore_rng_state(rng_state)
        self._refresh_obs_from_env()

    def _prepare_robot_initial_state(self, *, instance_id: int, episode_id: int) -> None:
        if not self.should_use_repro_initial_state() or self._episode_initial_state_path is None:
            return

        if self._episode_initial_state_path.exists():
            self._restore_full_initial_state()
            return

        self._capture_full_initial_state(
            instance_id=instance_id,
            episode_id=episode_id,
        )

    @staticmethod
    def _action_to_numpy(action: Any) -> np.ndarray:
        action_np = np.asarray(
            th.as_tensor(action, dtype=th.float32).detach().cpu().numpy(),
            dtype=np.float32,
        )
        if action_np.dtype != np.float32:
            action_np = action_np.astype(np.float32, copy=False)
        return action_np

    def _ensure_action_chunk_buffer(self, action_np: np.ndarray) -> None:
        action_shape = tuple(action_np.shape)
        if self._action_shape is None:
            self._action_shape = action_shape
            self._action_chunk_buffer = np.empty(
                (self.repro_action_chunk_size(),) + action_shape,
                dtype=np.float32,
            )
            self._action_chunk_buffer_count = 0
            return
        if action_shape != self._action_shape:
            raise RuntimeError(
                f"Action shape changed within episode: expected {self._action_shape}, got {action_shape}"
            )

    def _resolve_replay_action_dir(self, instance_id: int, episode_id: int) -> Optional[Path]:
        explicit_replay_action_log_path = self.replay_action_log_path()
        if explicit_replay_action_log_path is not None:
            if self._manifest_path_from_dir(explicit_replay_action_log_path).exists():
                return explicit_replay_action_log_path
            return explicit_replay_action_log_path / str(instance_id) / str(episode_id)
        return self._default_action_artifact_dir(instance_id, episode_id)

    def _build_action_manifest(
        self,
        *,
        instance_id: int,
        episode_id: int,
        complete: bool,
        total_chunks: int,
        final_step_count: int,
        action_shape: Optional[Tuple[int, ...]],
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manifest = {
            "task": self.cfg.task.name,
            "instance_id": int(instance_id),
            "episode_id": int(episode_id),
            "chunk_size": self.repro_action_chunk_size(),
            "total_chunks": int(total_chunks),
            "final_step_count": int(final_step_count),
            "action_shape": list(action_shape) if action_shape is not None else None,
            "action_dtype": "float32",
            "complete": bool(complete),
        }
        if extra_fields is not None:
            manifest.update(extra_fields)
        return manifest

    def _load_action_manifest_if_complete(self, action_dir: Path) -> Optional[Dict[str, Any]]:
        if not self.should_use_repro_action_log():
            return None
        manifest_path = self._manifest_path_from_dir(action_dir)
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if not manifest.get("complete", False):
            return None

        initial_state_path = action_dir / "initial_state.npz"
        initial_state_manifest_path = action_dir / "initial_state_manifest.json"
        if not initial_state_path.exists() or not initial_state_manifest_path.exists():
            logger.warning(
                f"Ignoring replay log at {action_dir}: missing colocated initial simulator state."
            )
            return None

        try:
            total_chunks = int(manifest.get("total_chunks", 0))
            final_step_count = int(manifest.get("final_step_count", 0))
        except (TypeError, ValueError):
            return None
        if total_chunks <= 0:
            return None
        if final_step_count <= 0:
            logger.warning(
                f"Ignoring replay log at {action_dir}: manifest.final_step_count must be > 0, "
                f"got {manifest.get('final_step_count')!r}."
            )
            return None

        manifest_action_shape = manifest.get("action_shape")
        if manifest_action_shape is not None:
            try:
                manifest_action_shape = tuple(int(dim) for dim in manifest_action_shape)
            except (TypeError, ValueError):
                logger.warning(
                    f"Ignoring replay log at {action_dir}: invalid manifest.action_shape={manifest.get('action_shape')!r}."
                )
                return None

        counted_steps = 0
        observed_action_shape: Optional[Tuple[int, ...]] = None
        for chunk_idx in range(total_chunks):
            chunk_path = self._chunk_path_from_dir(action_dir, chunk_idx)
            if not chunk_path.exists():
                logger.warning(
                    f"Ignoring replay log at {action_dir}: missing replay chunk {chunk_idx} "
                    f"({chunk_path.name})."
                )
                return None
            chunk = np.load(chunk_path, allow_pickle=False)
            if chunk.ndim == 0:
                logger.warning(
                    f"Ignoring replay log at {action_dir}: replay chunk {chunk_path.name} is scalar, "
                    "expected a batched action array."
                )
                return None
            if len(chunk) == 0:
                logger.warning(
                    f"Ignoring replay log at {action_dir}: replay chunk {chunk_path.name} is empty."
                )
                return None

            chunk_action_shape = tuple(chunk.shape[1:])
            if observed_action_shape is None:
                observed_action_shape = chunk_action_shape
            elif chunk_action_shape != observed_action_shape:
                logger.warning(
                    f"Ignoring replay log at {action_dir}: replay chunk {chunk_path.name} has action shape "
                    f"{chunk_action_shape}, expected {observed_action_shape}."
                )
                return None
            counted_steps += int(len(chunk))

        if manifest_action_shape is not None and observed_action_shape != manifest_action_shape:
            logger.warning(
                f"Ignoring replay log at {action_dir}: manifest action_shape={manifest_action_shape} "
                f"does not match replay chunks {observed_action_shape}."
            )
            return None
        if counted_steps != final_step_count:
            logger.warning(
                f"Ignoring replay log at {action_dir}: manifest.final_step_count={final_step_count}, "
                f"but replay chunks contain {counted_steps} actions."
            )
            return None
        return manifest

    def _prepare_episode_action_artifacts(self, instance_id: int, episode_id: int) -> None:
        if not self.should_use_repro_action_log():
            return

        replay_mode = self.replay_mode()
        replay_source_dir = self._resolve_replay_action_dir(instance_id, episode_id)
        replay_manifest = None
        if replay_mode != "none" and replay_source_dir is not None:
            replay_manifest = self._load_action_manifest_if_complete(replay_source_dir)

        logger.info(f"Replay mode for this episode: {replay_mode}")
        if replay_source_dir is not None:
            logger.info(f"Replay source directory: {replay_source_dir}")

        record_action_dir = self._default_action_artifact_dir(instance_id, episode_id)
        if replay_manifest is not None:
            self._replay_action_enabled = True
            self._replay_action_dir = replay_source_dir
            self._replay_manifest = replay_manifest
            self._replay_until_step = self.replay_until_step()
            logger.info(
                f"Replay enabled for episode from {self._replay_action_dir} with replay_until_step={self._replay_until_step}."
            )
            default_record_dir = record_action_dir
            record_action_dir = default_record_dir.with_name(
                f"{default_record_dir.name}__{self.resume_record_suffix()}"
            )
            if replay_source_dir is not None and record_action_dir == replay_source_dir:
                record_action_dir = record_action_dir.with_name(f"{record_action_dir.name}__record")
            logger.info(f"Recording resumed or fallback actions to {record_action_dir}")
        elif replay_mode == "force":
            replay_source_label = str(replay_source_dir) if replay_source_dir is not None else "<unset>"
            raise RuntimeError(
                f"replay_mode='force' requires a valid complete replay log, but none was found at {replay_source_label}"
            )
        else:
            logger.info("Replay not enabled for this episode; using policy inference.")

        self._episode_action_dir = record_action_dir
        self._episode_action_manifest_path = self._manifest_path_from_dir(record_action_dir)
        if self._episode_action_dir.exists():
            self._clear_action_artifacts(self._episode_action_dir)
        self._episode_action_dir.mkdir(parents=True, exist_ok=True)

        self._episode_action_manifest_meta = {
            "task": self.cfg.task.name,
            "instance_id": int(instance_id),
            "episode_id": int(episode_id),
            "resumed_from_replay": bool(self._replay_action_enabled),
            "replay_source_dir": str(self._replay_action_dir) if self._replay_action_dir is not None else None,
            "replay_consumed_steps": 0,
        }
        _atomic_write_json(
            self._episode_action_manifest_path,
            self._build_action_manifest(
                instance_id=instance_id,
                episode_id=episode_id,
                complete=False,
                total_chunks=0,
                final_step_count=0,
                action_shape=None,
                extra_fields=self._episode_action_manifest_meta,
            ),
        )

    def _record_action_step(self, step_idx: int, action: Any) -> None:
        if not self.should_use_repro_action_log() or self._replay_action_enabled:
            return
        action_np = self._action_to_numpy(action)
        self._ensure_action_chunk_buffer(action_np)
        assert self._action_chunk_buffer is not None
        self._action_chunk_buffer[self._action_chunk_buffer_count] = action_np
        self._action_chunk_buffer_count += 1
        self._recorded_action_count += 1
        if self._action_chunk_buffer_count >= self.repro_action_chunk_size():
            self._flush_action_chunk_buffer()

    def _flush_action_chunk_buffer(self) -> None:
        if self._action_chunk_buffer is None or self._action_chunk_buffer_count == 0:
            return
        chunk_path = self._action_chunk_path(self._action_chunk_index)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            chunk_path,
            self._action_chunk_buffer[: self._action_chunk_buffer_count],
            allow_pickle=False,
        )
        self._action_chunk_buffer_count = 0
        self._action_chunk_index += 1

    def _finalize_action_artifacts(self, *, episode_completed_normally: bool) -> None:
        if (
            not self.should_use_repro_action_log()
            or self._episode_action_manifest_path is None
            or self._episode_action_dir is None
            or self._episode_action_manifest_meta is None
        ):
            return
        self._flush_action_chunk_buffer()
        _atomic_write_json(
            self._episode_action_manifest_path,
            self._build_action_manifest(
                instance_id=int(self._episode_action_manifest_meta["instance_id"]),
                episode_id=int(self._episode_action_manifest_meta["episode_id"]),
                complete=episode_completed_normally,
                total_chunks=int(self._action_chunk_index),
                final_step_count=int(self._recorded_action_count),
                action_shape=self._action_shape,
                extra_fields={
                    **self._episode_action_manifest_meta,
                    "replay_consumed_steps": int(self._replay_loaded_action_count),
                },
            ),
        )

    def finalize_episode_artifacts(self, *, episode_completed_normally: bool) -> None:
        self._finalize_action_artifacts(episode_completed_normally=episode_completed_normally)

    def _disable_replay_mode(self) -> None:
        self._replay_action_enabled = False
        self._replay_manifest = None
        self._replay_current_chunk = None
        self._replay_current_chunk_offset = 0
        self._replay_chunk_index = 0
        self._replay_until_step = None

    @staticmethod
    def _is_replay_fallback_error(exc: RuntimeError) -> bool:
        error_message = str(exc)
        return (
            "Replay action stream exhausted before episode termination." in error_message
            or "Replay action step mismatch:" in error_message
        )

    def _load_replay_chunk(self) -> None:
        assert self._replay_manifest is not None
        assert self._replay_action_dir is not None
        total_chunks = int(self._replay_manifest["total_chunks"])
        if self._replay_chunk_index >= total_chunks:
            raise RuntimeError(
                "Replay action stream exhausted before episode termination. "
                f"Consumed {self._replay_loaded_action_count} actions from "
                f"{self._replay_action_dir} (manifest.final_step_count="
                f"{self._replay_manifest.get('final_step_count')})."
            )
        chunk_path = self._chunk_path_from_dir(self._replay_action_dir, self._replay_chunk_index)
        self._replay_current_chunk = np.load(chunk_path, allow_pickle=False)
        self._replay_current_chunk_offset = 0
        self._replay_chunk_index += 1

    def _load_replay_action(self, step_idx: int) -> th.Tensor:
        if self._replay_current_chunk is None or self._replay_current_chunk_offset >= len(self._replay_current_chunk):
            self._load_replay_chunk()

        assert self._replay_current_chunk is not None
        expected_step = int(self._replay_loaded_action_count)
        if expected_step != int(step_idx):
            raise RuntimeError(
                f"Replay action step mismatch: expected recorded step {expected_step}, got env step {step_idx}"
            )
        action = self._replay_current_chunk[self._replay_current_chunk_offset]
        self._replay_current_chunk_offset += 1
        self._replay_loaded_action_count += 1
        return th.from_numpy(np.asarray(action, dtype=np.float32))

    def open_episode_outputs(
        self,
        *,
        instance_id: int,
        episode_id: int,
        video_path: Optional[Path],
    ) -> Dict[str, Optional[str]]:
        output_paths: Dict[str, Optional[str]] = {
            "video": None,
            "third_person_video": None,
        }
        if video_path is None:
            return output_paths

        if self.should_write_video():
            video_name = str(video_path / f"{self.cfg.task.name}_{instance_id}_{episode_id}.mp4")
            self.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(448, 672),
            )
            output_paths["video"] = video_name

        if self.should_write_third_person_video():
            third_person_video_name = str(
                video_path / f"{self.cfg.task.name}_{instance_id}_{episode_id}_3rd_person_view.mp4"
            )
            self.third_person_video_writer = create_video_writer(
                fpath=third_person_video_name,
                resolution=self.third_person_view_config["resolution"],
            )
            output_paths["third_person_video"] = third_person_video_name

        return output_paths

    def close_episode_outputs(self, *, episode_completed_normally: bool = False) -> None:
        self.finalize_episode_artifacts(episode_completed_normally=episode_completed_normally)
        self.video_writer = None
        self.third_person_video_writer = None
        self.close_episode_logging()
        self._close_additional_episode_outputs()
        self._reset_episode_repro_context()

    def _close_additional_episode_outputs(self) -> None:
        """Subclass hook for extra episode-local cleanup."""

    def write_episode_videos(self) -> None:
        if self.should_write_video():
            self._write_video()
        if self.should_write_third_person_video():
            self._write_third_person_video()

    def gather_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.gather_results())
        return metrics

    def run_episode(
        self,
        *,
        instance_id: int,
        episode_id: int,
        metrics_path: Path,
        video_path: Optional[Path],
    ) -> Dict[str, Any]:
        self.prepare_episode(instance_id=instance_id, episode_id=episode_id)
        self.reset()
        self.prepare_episode_artifacts(instance_id=instance_id, episode_id=episode_id)
        self.start_episode_logging(metrics_path=metrics_path, instance_id=instance_id, episode_id=episode_id)

        terminated = False
        truncated = False
        episode_completed_normally = False
        output_paths = self.open_episode_outputs(
            instance_id=instance_id,
            episode_id=episode_id,
            video_path=video_path,
        )

        try:
            for metric in self.metrics:
                metric.start_callback(self.env)

            done = False
            while not done:
                terminated, truncated = self.step()
                if terminated or truncated:
                    done = True
                    episode_completed_normally = True
                self.write_episode_videos()
                if self.env._current_step % 1000 == 0:
                    logger.info(f"Current step: {self.env._current_step}")

            for metric in self.metrics:
                metric.end_callback(self.env)

            metrics = self.gather_metrics()
            metrics_file = metrics_path / f"{self.cfg.task.name}_{instance_id}_{episode_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f)
        finally:
            self.close_episode_outputs(episode_completed_normally=episode_completed_normally)

        return {
            "terminated": terminated,
            "truncated": truncated,
            "metrics_file": str(metrics_file),
            **output_paths,
        }

    def reset(self) -> None:
        """
        Reset the environment state and policy for the next episode.
        """
        self.obs = self._preprocess_obs(self.env.reset()[0])
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print stats
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.close_episode_outputs()
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda:0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    gm.HEADLESS = config.headless
    metrics_path, video_path, should_write_video, should_write_third_person_video = resolve_eval_output_paths(config)
    instances_to_run = resolve_instances_to_run(
        config,
        num_train_instances=m.NUM_TRAIN_INSTANCES,
        num_eval_instances=m.NUM_EVAL_INSTANCES,
    )

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                episode_outputs = evaluator.run_episode(
                    instance_id=idx,
                    episode_id=epi,
                    metrics_path=metrics_path,
                    video_path=video_path if (should_write_video or should_write_third_person_video) else None,
                )
                log_episode_summary(
                    logger,
                    evaluator,
                    episode_outputs,
                    write_video=should_write_video,
                    write_third_person_video=should_write_third_person_video,
                )
