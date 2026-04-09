from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    venv_python_default = os.path.expanduser("~/venvs/multicam_det_cpu/bin/python")

    venv_python = LaunchConfiguration("venv_python")
    params_file = LaunchConfiguration("params_file")

    default_params = os.path.join(os.path.dirname(__file__), "..", "config", "params.yaml")
    default_params = os.path.normpath(default_params)

    return LaunchDescription([
        
        SetEnvironmentVariable('PYTHONPATH', os.environ.get('PYTHONPATH','')),
        SetEnvironmentVariable('AMENT_PREFIX_PATH', os.environ.get('AMENT_PREFIX_PATH','')),
        SetEnvironmentVariable('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH','')),
DeclareLaunchArgument(
            "venv_python",
            default_value=venv_python_default,
            description="Path to venv python that has torch/ultralytics installed"
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to params.yaml"
        ),
        ExecuteProcess(
            cmd=[
                venv_python,
                "-u",
                "-m", "multicam_fusion_detector.node",
                "--ros-args",
                "--params-file", params_file,
                "--log-level", "info",
            ],
            output="screen",
            emulate_tty=True,
        )
    ])
