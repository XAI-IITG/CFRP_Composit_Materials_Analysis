import networkx as nx
from typing import Dict, List, Tuple

class CFRPKnowledgeGraph:
    """Knowledge Graph for CFRP Composite Degradation"""
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Changed from self.G to self.graph
        self._build_domain_knowledge()
    
    def _build_domain_knowledge(self):
        """Build the core domain knowledge"""
        
        # 1. Add Physical Phenomena Nodes
        phenomena = [
            ('crack_propagation', {'type': 'phenomenon', 'severity': 'high'}),
            ('delamination', {'type': 'phenomenon', 'severity': 'critical'}),
            ('matrix_cracking', {'type': 'phenomenon', 'severity': 'medium'}),
            ('fiber_breakage', {'type': 'phenomenon', 'severity': 'critical'}),
            ('stiffness_loss', {'type': 'phenomenon', 'severity': 'high'})
        ]
        self.graph.add_nodes_from(phenomena)
        
        # 2. Add Feature Nodes
        features = [
            ('delta_psd', {'type': 'feature', 'sensor': 'pzt', 'unit': 'power'}),
            ('energy', {'type': 'feature', 'sensor': 'pzt', 'unit': 'joules'}),
            ('tof', {'type': 'feature', 'sensor': 'pzt', 'unit': 'seconds'}),
            ('mean_intensity', {'type': 'feature', 'sensor': 'xray', 'unit': 'grayscale'}),
            ('crack_density', {'type': 'feature', 'sensor': 'xray', 'unit': 'count/mm²'}),
            ('mean_strain', {'type': 'feature', 'sensor': 'strain', 'unit': 'microstrain'})
        ]
        self.graph.add_nodes_from(features)
        
        # 3. Add Degradation Stage Nodes
        stages = [
            ('healthy', {'type': 'stage', 'rul_min': 180000, 'rul_max': 227000}),
            ('early_damage', {'type': 'stage', 'rul_min': 120000, 'rul_max': 180000}),
            ('progressive', {'type': 'stage', 'rul_min': 60000, 'rul_max': 120000}),
            ('critical', {'type': 'stage', 'rul_min': 0, 'rul_max': 60000})
        ]
        self.graph.add_nodes_from(stages)
        
        # 4. Add Causal Relationships (Feature → Phenomenon)
        feature_to_phenomenon = [
            ('delta_psd', 'crack_propagation', {'weight': 0.9, 'relation': 'indicates'}),
            ('energy', 'delamination', {'weight': 0.85, 'relation': 'correlates'}),
            ('tof', 'crack_propagation', {'weight': 0.7, 'relation': 'measures'}),
            ('mean_intensity', 'matrix_cracking', {'weight': 0.8, 'relation': 'visualizes'}),
            ('crack_density', 'crack_propagation', {'weight': 0.95, 'relation': 'quantifies'}),
            ('mean_strain', 'stiffness_loss', {'weight': 0.9, 'relation': 'reflects'})
        ]
        self.graph.add_edges_from(feature_to_phenomenon)
        
        # 5. Add Phenomenon → Stage relationships
        phenomenon_to_stage = [
            ('crack_propagation', 'early_damage', {'weight': 0.6}),
            ('crack_propagation', 'progressive', {'weight': 0.9}),
            ('delamination', 'progressive', {'weight': 0.8}),
            ('delamination', 'critical', {'weight': 0.95}),
            ('fiber_breakage', 'critical', {'weight': 1.0}),
            ('stiffness_loss', 'progressive', {'weight': 0.7})
        ]
        self.graph.add_edges_from(phenomenon_to_stage)
    
    def get_explanation_path(self, feature_importances: Dict[str, float], 
                            predicted_rul: float) -> List[str]:
        """Generate explanation path from features to RUL prediction"""
        
        # Determine degradation stage
        if predicted_rul > 180000:
            stage = 'healthy'
        elif predicted_rul > 120000:
            stage = 'early_damage'
        elif predicted_rul > 60000:
            stage = 'progressive'
        else:
            stage = 'critical'
        
        # Find top contributing features
        top_features = sorted(feature_importances.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:3]
        
        explanations = []
        for feature, importance in top_features:
            if feature in self.graph:
                # Find path: Feature → Phenomenon → Stage
                try:
                    paths = nx.all_simple_paths(self.graph, feature, stage, cutoff=3)
                    for path in paths:
                        explanation = self._path_to_explanation(path, importance)
                        explanations.append(explanation)
                        break  # Use first path found
                except nx.NetworkXNoPath:
                    pass
        
        return explanations
    
    def _path_to_explanation(self, path: List[str], importance: float) -> str:
        """Convert graph path to natural language explanation"""
        
        if len(path) < 3:
            return f"Feature {path[0]} contributes {importance:.2%} to prediction"
        
        feature, phenomenon, stage = path[0], path[1], path[2]
        relation = self.graph[feature][phenomenon]['relation']
        
        return (f"Feature '{feature}' (importance: {importance:.2%}) "
                f"{relation} '{phenomenon}', "
                f"indicating '{stage}' degradation stage")