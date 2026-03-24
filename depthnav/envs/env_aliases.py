from typing import Dict
from .base_env import BaseEnv
from .navigation_env import NavigationEnv
#字符串 → 环境类的映射字典（供 YAML 配置按名实例化）
env_aliases: Dict[str, BaseEnv] = {
    "navigation_env": NavigationEnv,
}
