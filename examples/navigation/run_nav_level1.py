from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # for tensorboard
    run_params = {
        "level0": (False, "configs/box_2", 500),
        #不加载 geodesics（更简单）使用简单地图 box_2 训练 500 次
        "level1": (True, "configs/level_1", 20000),
    }
    #加载默认配置模板 环境参数（sensor、地图）模型结构reward设计 optimizer
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty.yaml",
        "examples/navigation/train_cfg/nav_levelX.yaml",
    ]
    #实际执行的训练程序
    #训练脚本
    #日志目录
    #动态配置
    #policy配置
    #评估配置
    #评估结果保存
    #curriculum学习
    #自动重试
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,

        policy_config_file="examples/navigation/policy_cfg/small_yaw.yaml",
        #评估配置，加载 geodesics（更复杂）使用 level_1 地图评估 20000 次
        eval_configs=["examples/navigation/eval_cfg/nav_level1.yaml",],
        eval_csvs=["examples/navigation/logs/level1/nav_level_1.csv",],
        curriculum=True,
        max_retries=5,
    )
