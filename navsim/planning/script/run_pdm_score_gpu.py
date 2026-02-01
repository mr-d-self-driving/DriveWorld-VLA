import pandas as pd
from tqdm import tqdm
import traceback
import glob
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import logging
import lzma
import pickle
import os
import uuid

import torch  # Import PyTorch for GPU handling

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

token_list = ['418f3b3a155f5655', 'a0be9e4c6cc15ec7', 'dd25fc02c23c537b', '75ffe77189e95d43', 
'6891e8163e4d5e58', '303d452ddd2d58d1', '92427e4a55475ba6', '9fc5098e21355c6e', 
'37a2f90109f85bff', '36f96e28725c5f5e', '7021139b47cf5370', 'fa9a6007ed205fe7', 
'6247bc8d9df7529d', 
'919ee41f5fa65358', '8354786ba35c5440', '9147938e42675685', '0503cc398e2656be', 
'498bf3fdec5e5506', 'b2f320b5d94753f9', '878f3ddae1b7550f', '95effa16c4bb5c12', 
'2322850f71fa5c4d', '47e3bbbd82b2583b', 'ef41e9c7f99d5d85', '2d2b781181f156dc', 
'213870b088245664', '66a0abcb3eac57df', 'be0ea3126c955eae', 'a4ab5ff4ba7259f9', 
'313756e5fc655c0e', '06392ff957445576', '69ce68dff3ff566d', '6f2f0885518356ef', 
'6fee6df4d64b5d9d', '1309e3cfb6f25c6e', 'f35c6a6f6a1157b3', 'b08965aff8165bec', 
'ceedeccf36c35c11', '67d74f48ba2a548e', '18a85f2812b45525', '84136b2623ef5618', 
'77398398b6c550c2', '2360cd7b0fd85480', 'f0d1419c24b85651', '76df9fb807c9580a', 
'b4eb56fcd01857bb', '9bb8633f7c0157fb', '32c0c70837d05c47', '119bd713e2db5e25', 
'f6082f18c392582f', '85164858b3e55841', '9e10e31698a650a1', 'f80a6f7f4de5564d', 
'239cb4818dc458fd', '3c303aa231e65a53', '963d98eaf75259f6', '1a37d69ab5805f15', 
'4c53b20bf0925c0c', '926d8c9ced715a42', '76b1b05efba353fe', '25e2cda283355b76', 
'14b25ce243865457', '59c00098b95d53b8', '36ea923bc32b5181', 'cbadd750cbd6581b', 
'1d7ff3e8eed15ced', '975e6554112f5a46', 'b1b2eb47b045566c', 'cf533c1e7f6852e5', 
'5005f44bd2135f3b', '175f0101f12750b1', 'c227b95266d75371', 'f8b8b8ee49205def', 
'dd6f3d80501c5026', '80bc2d8487e552f1', '80386d2e9c215d3b', 'b50cf2a689a55433', 
'1e15c5256da5549a', 'afbbcee34f1850fd', 'c65ad538a2275dd8', '682660223761501d', 
'3265e31c65705a5f', '29d3a233b1915c1a', 'd925879cb467504c', '8dc7d00ce175549d', 
'8043aa566aa45878', '449ed0c45858583f', 'ea8b66a838c75042', '9157902936a456bf', 
'385b1aa4b84c56bf', '816c515aa3dc5462', '631115d8e54d58fc', 'bef85b7ac1dc5207', 
'c0d51ca7b8af5414', 'c370f836e3275da4', '5aafc350fc705533', '084ce89976f1505e', 
'661278fc8a9c57d3', 'e024bd23594b5a13', 'e88e44d720f65e0e', '611576c7588c543b', 
'd5a0b92fbf8b51df', '05f9443de2185b91', '88126672803f56a3', 'e41740b9529753af', 
'4314c162a57f568e', 'ade979d99d51517f', '89751620b3555ef8', '051b3ee34e3b59ec', 
'c195a8cb7ccb59fd', 'deff279b0c815e5a', '20728d3e677b593d', 'f29d6171f46b551d', 
'3b53493dfe335ea6', '9fa75f1d5863570b', '1c6e6f287c2354f1', '642a177df62954ad', 
'9d7ec713a2fb5e44', 'a8ee480d197e56dd', 'b5e3abde704b54c8', '5c872773198c5689', 
'12ca80cc0d575578', '567469e556cf5e6b', '5bdbe484641a55fa', '60c9aabf696a5756', 
'5317b890d3e45958', 'c5727dc4f8665554', 'f0a338ae8f3a56bd', 'bd4adf326a205d51', 
'c772799eaf1f5ad1', 'f3aaed259ab15dcc', '63bd566666b75e4a', '25e9d76574075cee', 
'7feaf6410f1854a8', '3f82940c5aae5ab1', '3466b16cf2c95855', '0855e09733fe5445', 
'1e4fb3a9572f5e7d', 'bdb0f78978cd5307', '635495c0e2295c4d', 'fe94da01811e53fc', 
'01ce9c82de0d5539', 'ab845dd1ff3c522d', 'e12cd09a8f515c73', 'fb8f21990ffb52d5', 
'c3ef435d900256c1', '42b065f7e76d50ed', 'e7a7c61c543e5b88', 'fb4ac387cd285171', 
'0f186e17ed445c5a', '2dca181fa97153a0', '3b4031def0f45d96', '627ed4c9950753e9', 
'df240e44ad0d5c3c', '95329718628e5f7f', '64b35b9f9f84585a', 'fd99b81c7f2b5ed7', 
'cd828def214c571e', '9ecb516bac035ce4', 'd7fdaa2d555c5272', 'd97108bcdab25c24', 
'ff1044970b525386', '063f7dfe767d55aa', '7bd486db42e35b18', '932f0bf1d23b5ba7', 
'44324a7aa04f5501', '0a08850db6e35ab7', '63070010ae7159f2', '5012cae5e4fc57a1', 
'39e0bf171e6d5e88', '41890b20b92953b4', '06737d2244bd53cc', 'e9b43b140b1d54bf', 
'96164ad9ed8557a7', '13d078252f4653f3', '6e778c30490d5f50', '3f265d778b65596b', 
'49f37d1d50ef5873', 'd79874fa9ba4558e', 'd0d94d5dca655084', '239705f6ca945846', 
'e7756dc30c605482', '5d058c203f765173', '43cc094e7af0518e', '97b1ea53fad65625', 
'e10a3a8d627e5700', '4ef621a8318b5085', '2391f12d7e6a5e7f', 'dd149fc9cb395631', 
'e6c8f0c4aaea52d7', '8371ac42ba585d35', '70f718ebc8b4505e', '2428ecd565d05b50', 
'0059b8e52a3f55e8', 'be382c6340e75946', '0d49407b94b259e8', '8c527efc3b3b5fa0', 
'a3399fc1f78b55f2', '3af98cdf0bc35f44', '599cd3d1986c5509', '966e42e47cc85a5c', 
'79551644e9715069', '37d89f35328e55b6', '6a6362156db75390', 'abda798b87535e07', 
'54580876ee835959', '5f3da1e584905c8e', '4e00bf86cd1e5a14', 'ff5c0e17d20351ca', 
'cf3b00d5d2b953d3', '2a68e10c7d305af4', '5b7700fa99d95a94', 'ab807bda4fdd5274', 
'cf7d3181a516574a', '2488018e08b35c68', '0613d88c01b453c9', '1f7fe5fcd7965b1c', 
'78a98ffdcd4b558a', 'b61a73309da75ac3', '8cac1b4a21585010', '9b3a284d78f458f3', 
'c12b3554dcd655c0', '20f825e0d33a5160', 'a572d70690f75ad4', '65fc12f7836f58ed', 
'a2f75428b992536c', '0a6906b694b65108', '5a80299213875068', 'b287d1e7b2965ba1', 
'5496f69033515dd4', '70f3a3d098bf5381', '308e246a1d995edd', '34cc387cf0335f28', 
'fba8aa5a1cb8583e', '360b5fec28655626', 'f40172a8fb1a54ee', '873e80bc10d156ea', 
'ac6e1ca0ebe75e15', '378f82a326bd5875', 'aa58dfcb46765181', 'e77441c822075f1b', 
'2ad9640eb724590d', '6d088615a8c05649', '2cce988d410654e2', '83aba0a1454c59e4', 
'2575048779565f0b', 'dbb53601c8fb585f', 'a12c7f84ebd25b1e', '1ec58e13fee45a30', 
'97628f2e9ee55826', '3038555c47555d08', 'e6ff81ab83355450', '58ba738c8dd15d55', 
'0fa0e8a25eec5e1e', 'fc450a5080d458e1', '96dc22c0224255ca', '8b57a471e74e573f', 
'789eca1c50f85cb4', '96f19d920d5558e9', '03817943eb905452', '7e7acb8e97a9520d', 
'f3493d3b23cd55c2', '5252cad32bf358b2', '77919997d5c352d8', '819d34b59e6159ad', 
'41f3712a30b956e6', '9f191d1313c95362', 'ec921ad4a3d05806', '9089d6dfa77e5488', 
'10b19bf49b67574f', 'f9592039b6aa5165', '5e25570a2f725a17', '785b0e9d5d505db4', 
'87bf38cb3c39548d', 'b8d114f0304d5f87', '6c84787939055fe8', '795b989aef8a5b42', 
'4417a92b5b1956b8', '48b503ec9c5a5d7a', '0bdc177e43cf5df8', 'd903d26195085adb', 
'0166f867762a56e1', '49bd1e2dd88457e3', '1b25edea36205814', 'e45023fbb46857d6', 
'33ee6f1f594d50ca', 'ec3c0587b1775b7c', '5ab779b8a0995778', 'b1e78f926612520e', 
'76d34ae8c95156b2', 'a6be7050ff205933', 'a208267045685266', '277cdfc9479e52c4', 
'70ce3fce14785ae5', 'f18ef27ec008527a', '78c9157a55905d81', 'f94a9ce36b8e5516', 
'ebc5a6ec11205f0c', '7145c064885a53c4', 'bfcb040f50425141', '68d46e380acc5f56', 
'4cf608d9de4e5349', 'fff973197a795e6a', 'f4d95a784b725915', '63f6c401c24557e1', 
'eb6c8a396bb75a89', '90ed944de9405835', 'cdc469758f7c502b', '919aa2aa25a951f4', 
'660451aead9653b7', '9157afe90f035621', '40f09c66198258e6', '167063f69e8357d8', 
'8b9afa7045785acc', '2b7bf25209dd5705', '60c5f18db58c54da', '575453863ce05f35', 
'cf9af12edb535bc0', 'c9968e9ec0135b9d', '14a11185124a508b', '41892d06df125856', 
'786c64badbb15a9c', '8931b1302d7a5f61', 'a1848a01a20b5224', '542d8c3b05145bbe', 
'aea14d7408d255bb', 'fe22e2c812e65096', '8c12150f849b5b10', '6e983b745cb9535b', 
'6688db9c3a425bb6', '1f248c6b5f2f5234', 'eec0bc4e1b185d67', '43a759f57b1d51e4', 
'566cac9a7c0358ce', '4356c31443585d79', '580a10ad9ad55a5d', '2daa400892ce5a95', 
'0d3dd11b84c8518e', '253dba7510ca56b8', 'b204289cd95c59b5', '09034d77d3b15c5a', 
'44f6f13541dd526e', '7c8cb841bdfd5c56', '729c7f95c12f5dd4', '3c8a95ec33f45af6', 
'fdf7643412e85c31', '971d199e8b9b5e71', 'b50b8f11d75a5cb0', '2c1f43c6b93c5952', 
'8ba84ea2cc4a5ead', 'daaa1858309c5938', '79cfc07ef4645c81', '1c5aaf9e884b5ce9', 
'6ec2148215205936', 'a5ee1632a31556a1', '1f67195591a95027', 'e3e49101a8b45645', 
'3f5d3b68c1a156ed', '852255373f315fb6', 'd5039871a3fb5b04', '57e7565904e05728', 
'3b56d2b022d25026', 'f5af09063e125bb2', '7140077d03e25c2c', '6326d00e52115da4', 
'deae4b3cac52513c', '684ff3674eeb54ea', 'f0cdcc3967335c01', '7dac1567fa8d574c', 
'0c9105d8ad6a5f52', '7022b42a3743507a', 'd04049c3978354f0', 'c7cb2e43b2d053b7', 
'b393ab92c8ac5e77', '6781255a85605dd5', '569da35d0f00545a', '55d6a50a401f52f7', 
'6011fd3aa3a85fa1', 'ea1c363c888b59c7', '1bea55e75a9b593b', 'b5f57f5a6b5b5244', 
'5ea2a4ecc18752fa', '4109b79d84ea5053', '173b456bbf29598f', '61d114c5cfb25663', 
'94817f3d96fe5072', '66dd7339f01b58c2', 'e859a9e666c15716', 'aafbe79abf625492', 
'8625ec015e075c02', 'c84b4ca798ca563a', '9bfa838de21f5d25', '31602c26eedc5f7e', 
'01e4bff700f15523', 'a2e2a360029f508f', '12f96c65436e56bf', '4cca4c982f9855c5', 
'4a2f87a2fd42549e', '6321c384f59c5265', '305eb53245715f7e', '85ee18c706cf5966', 
'489facb8ac705f4a', '212c67511de75043', '700cac410af95cc4', '1b35b775c4a95647', 
'd9634d08b1a85ec4', '3d4f0c1ecad55944', 'c081afa672dc56a2', 'c133861a233a51de', 
'9963c72e34ae5101', '562f38636f975be4', 'd5a79b89fb985129', '5f2d040ee3bf5c6e', 
'4299796c2c845718', 'da1a3756e4465c03', '3fa1e0c2313358ba', '59d306aa441a5665', 
'95ba88fe9c385123', '9d1d720d0e2e511e', '1c45a4d22fcc5b2e', '7ef36a2139b45d9a', 
'f8bec974b86952e9', '2eac5afe2036526c', '6832568ebd835696', 'ed2a869a8d1c5eb7', 
'2f67d7af0e0d51a1', 'c5510354351d5d5e', '2b34624bf9455480', '1ce780e012fc5d23', 
'511ba067a1835f86', 'a89c8c3249975dc9', '9eb02188c5505fe1', '498c6f15e5c856bb', 
'7bcc91de36385afe', '1766fa78fd10576e', 'a39ec7322394565f', '4dde36d102a4526b', 
'173ccb0e08885b66', '77fa15ffcefb580e', '0994517bde3c5188', '625bda810b66583a', 
'67e6aea894f0568f', '9ddee0363c1f530a', 'eadc0f19b54352ae', 'b541b6d67bad546d', 
'90314b74c92954d6', 'c240593c969a5cfc', '83ec4461559c5388', '4952a782e55057b7', 
'be0064dac5f85957', '0c8b9f08bfa25dc2', 'aa2918eb684351bd', '47f29dad0bec52d2', 
'd2440edd19d954b5', '1cd73301d5745314', '14b6e7ce317d531f', 'df2b37f4a1b85a4e', 
'925e046dba2f525f', 'ac47687039a75848', '9f461cae559d5e15', '26ba80ad72205a03', 
'd1dcd412f339598a', '268ab283b4a95126', '77519294678a5fed', '31013d67978e5284', 
'2cc421e9b3fd5c17', 'd3e99c65bb2a5d79', '94ac0f71d5615e4c', 'ac5327106349541c', 
'7dc47eb4f41e523d', 'aee1ca352fdd55f1', '4167ff5049555a2f', 'c4e94b7583555176', 
'ba8f7c5757365a36', 'bc3d973c3543556c', 'ed2466a660ba5661', '92079b2eb4675c0d', 
'32f53a8cb63f55db', '538ed1acbf145c8b', 'ce1f8997a8ce502b', 'e8edb3108f41545c', 
'f34af4dbbbd35a23', 'eaf3255921e35495', '7defd0c32cd8546a', '8fb6a2364e0c53f9', 
'6b97412307ad5c16', 'a82f1c8ad27d53cc', 'fc9b5914f47e58fe', '4a1f69f3821e5ac5', 
'31134bc6685c57ae', '96b5b1c350745a73', 'e0cd66542e715685', '9e8403c32a50530b', 
'a1e8639c17ac5089', 'ab2b84d310be5bed', '6b52f2b62f215cb4', 'dd54db98a83b5714', 
'4606b66d9fe75354', 'd8acd89d5cc95e9a', 'a8a743157c605bfe']

def get_files_using_glob(root_path):
    """使用glob模块获取文件"""
    # 获取当前目录下所有文件（不递归）
    files = glob.glob(os.path.join(root_path, "*"))
    
    # 递归获取所有文件
    all_files = glob.glob(os.path.join(root_path, "**", "*"), recursive=True)
    
    # 过滤，只保留文件
    all_files = [f for f in all_files if os.path.isfile(f)]
    
    return all_files

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    #import pdb;pdb.set_trace()
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    # MetricCacheLoader(Path('/e2e-data/evad-tech-vla/liulin/recogdrive/exp/metric_cache_train_1')) #
    #import pdb;pdb.set_trace()
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    single_eval = getattr(cfg, 'single_eval', False)
    # single-threaded worker_map
    print("Running single-threaded worker_map")
    score_rows = run_pdm_score(data_points, device)

    pdm_score_df = pd.DataFrame(score_rows)
    #import pdb;pdb.set_trace()
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
    """)

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]], device: torch.device) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # Move simulator, scorer, and agent to the specified device
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    #import pdb;pdb.set_trace()
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent).to(device)
    agent.initialize_stage_2()
    agent.is_eval = True

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    # ---------------------------------------------------
    # root_dir = '/e2e-data/evad-tech-vla/liulin/WoTE/gt_score_dir'
    # list_paths_ll = get_files_using_glob(root_dir)
    # token_list_ll = []
    # for path in list_paths_ll:
    #     path = path.split('/')[-1].split('.')[0]
    #     token_list_ll.append(path)
    # tokens_to_evaluate = list(set(tokens_to_evaluate) ^ set(token_list_ll))
    # import pdb;pdb.set_trace()
    # tokens_to_evaluate = ['71f883f1ec8f5e37','777cf50be27a5d4b','69df428fe853580e','9a675656c3c85f4c']
    tokens_to_evaluate = token_list
    #import pdb;pdb.set_trace()
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    trajectory = agent.compute_trajectory_gpu(agent_input, scene)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    #import pdb;pdb.set_trace()
                    trajectory = agent.compute_trajectory_gpu(agent_input, token)

            # Ensure trajectory is on the correct device
            if hasattr(trajectory, 'to'):
                trajectory = trajectory.to(device)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            # Move results back to CPU if necessary
            pdm_result_dict = asdict(pdm_result)
            for key, value in pdm_result_dict.items():
                if isinstance(value, torch.Tensor):
                    pdm_result_dict[key] = value.cpu().item()
            score_row.update(pdm_result_dict)
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results

if __name__ == "__main__":
    main()
