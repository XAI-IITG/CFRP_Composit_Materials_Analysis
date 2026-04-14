"""
Knowledge Graph XAI Modules for CFRP Composite Analysis
"""

from .kgbuilder import CFRPKnowledgeGraph
from .explainer import XAIExplainer
from .model_architecture import PositionalEncoding, TransformerRULPredictor, LSTMRULPredictor
from .drl_models import RULPredictionEnvironment, RULActorCritic, PPORULAgent, OrnsteinUhlenbeckNoise,\
ReplayBuffer, DDPGActor, DDPGCritic, DDPGAgent, RULPredictionEnvironmentDDPG, RULEnvironment, DRLRULPredictor, DQNAgent
# from .shap_lime_explainer import SHAPLIMEExplainer
from .rulefit import RuleFit
from .knowledge_trepan import KnowledgeTrepan
from .alpa import ALPA
from .refne import REFNE
from .query_engine import RuleQueryEngine
from .shap_explainer import SHAPBenchmarkExplainer
from .xai_benchmark import XAIBenchmark
from .temporal_rules import TemporalFeatureExtractor, TemporalRuleFit, TemporalRuleQueryEngine
from .stl_rules import STLFeatureExtractor, STLRuleFit, STLRuleQueryEngine, STLRuleTranslator

__all__ = [
    'CFRPKnowledgeGraph', 
    'XAIExplainer',
    'PositionalEncoding',
    'TransformerRULPredictor',
    'LSTMRULPredictor',
    'DDPGActor',
    'DDPGCritic',
    'PPOActorCritic',
    'DQNAgent',
    'RULPredictionEnvironment',
    'RULPredictionEnvironmentDDPG',
    'RULActorCritic',
    'PPORULAgent',
    'DDPGAgent',
    'OrnsteinUhlenbeckNoise',
    'ReplayBuffer',
    'RULEnvironment',
    'DRLRULPredictor',
    'RuleFit',
    'KnowledgeTrepan',
    'ALPA',
    'REFNE',
    # 'SHAPLIMEExplainer'
    'RuleQueryEngine',
    'SHAPBenchmarkExplainer',
    'XAIBenchmark',
    'TemporalFeatureExtractor',
    'TemporalRuleFit',
    'TemporalRuleQueryEngine',
    'STLFeatureExtractor',
    'STLRuleFit',
    'STLRuleQueryEngine',
    'STLRuleTranslator',
]
