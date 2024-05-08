import hydra
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

class HydraConfigsSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="test", path="pkg://hydra-config/test"
        )

def init_hydra():
    OmegaConf.register_new_resolver(
        "eval", lambda x: eval(x)
    )
    Plugins.instance().register(HydraConfigsSearchPathPlugin)
