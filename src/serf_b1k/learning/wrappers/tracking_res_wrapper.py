from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES

logger = create_module_logger("TrackingResWrapper")

TRACKING_RESOLUTION = 480


class TrackingResWrapper(EnvironmentWrapper):
    """Render at 480x480 for tracking; model input is resized to 224x224 later."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
            robot.sensors[sensor_name].image_height = TRACKING_RESOLUTION
            robot.sensors[sensor_name].image_width = TRACKING_RESOLUTION
        env.load_observation_space()
        logger.info(
            "TrackingResWrapper: sensors set to %dx%d",
            TRACKING_RESOLUTION,
            TRACKING_RESOLUTION,
        )
