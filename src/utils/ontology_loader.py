"""
Ontology Loader for CFRP Knowledge Graph

This module provides utilities to load and query the RDF/OWL ontology
defined in Turtle format for CFRP composite materials analysis.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL
    from rdflib.term import URIRef, Literal
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    logging.warning("rdflib not installed. Install with: pip install rdflib")

logger = logging.getLogger(__name__)


class CFRPOntologyLoader:
    """
    Loader for CFRP RDF/OWL ontology.
    
    This class provides methods to load, query, and extract information
    from the CFRP ontology defined in Turtle format.
    """
    
    def __init__(self, ontology_path: Optional[str] = None):
        """
        Initialize the ontology loader.
        
        Args:
            ontology_path: Path to the Turtle ontology file.
                          If None, uses default location in data/ontology/
        """
        if not RDFLIB_AVAILABLE:
            raise ImportError("rdflib is required. Install with: pip install rdflib")
        
        self.graph = Graph()
        self.ex = Namespace("http://example.org/cfrp/")
        
        # Default path
        if ontology_path is None:
            project_root = Path(__file__).parent.parent.parent
            ontology_path = project_root / "data" / "ontology" / "cfrp_ontology.ttl"
        
        self.ontology_path = Path(ontology_path)
        self._load_ontology()
    
    def _load_ontology(self):
        """Load the ontology from Turtle file."""
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")
        
        logger.info(f"Loading ontology from: {self.ontology_path}")
        self.graph.parse(self.ontology_path, format="turtle")
        logger.info(f"Loaded {len(self.graph)} triples")
    
    def get_all_classes(self) -> List[str]:
        """Get all OWL classes defined in the ontology."""
        classes = []
        for s in self.graph.subjects(RDF.type, OWL.Class):
            class_name = str(s).split('/')[-1]
            classes.append(class_name)
        return sorted(classes)
    
    def get_all_properties(self) -> Dict[str, List[str]]:
        """Get all properties (object and datatype) defined in the ontology."""
        properties = {
            'object_properties': [],
            'datatype_properties': []
        }
        
        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            prop_name = str(s).split('/')[-1]
            properties['object_properties'].append(prop_name)
        
        for s in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            prop_name = str(s).split('/')[-1]
            properties['datatype_properties'].append(prop_name)
        
        return properties
    
    def get_features(self) -> List[str]:
        """Get all feature instances."""
        features = []
        feature_class = URIRef(self.ex.Feature)
        for s in self.graph.subjects(RDF.type, feature_class):
            feature_name = str(s).split('/')[-1]
            features.append(feature_name)
        return features
    
    def get_damage_modes(self) -> List[str]:
        """Get all damage mode instances."""
        damage_modes = []
        damage_class = URIRef(self.ex.DamageMode)
        
        # Get direct instances
        for s in self.graph.subjects(RDF.type, damage_class):
            damage_name = str(s).split('/')[-1]
            damage_modes.append(damage_name)
        
        # Get instances of subclasses
        for subclass in self.graph.subjects(RDFS.subClassOf, damage_class):
            for s in self.graph.subjects(RDF.type, subclass):
                damage_name = str(s).split('/')[-1]
                if damage_name not in damage_modes:
                    damage_modes.append(damage_name)
        
        return damage_modes
    
    def get_feature_correlations(self) -> Dict[str, List[str]]:
        """
        Get feature-to-damage correlations defined in the ontology.
        
        Returns:
            Dictionary mapping features to correlated damage modes
        """
        correlates_with = URIRef(self.ex.correlatesWith)
        correlations = {}
        
        for feature, damage in self.graph.subject_objects(correlates_with):
            feature_name = str(feature).split('/')[-1]
            damage_name = str(damage).split('/')[-1]
            
            if feature_name not in correlations:
                correlations[feature_name] = []
            correlations[feature_name].append(damage_name)
        
        return correlations
    
    def get_rul_predictions(self) -> List[Dict]:
        """Get all RUL prediction instances with their properties."""
        rul_pred_class = URIRef(self.ex.RULPrediction)
        predictions = []
        
        for prediction in self.graph.subjects(RDF.type, rul_pred_class):
            pred_data = {
                'name': str(prediction).split('/')[-1],
                'rul': None,
                'coupon': None
            }
            
            # Get predictsRUL value
            for rul in self.graph.objects(prediction, URIRef(self.ex.predictsRUL)):
                pred_data['rul'] = int(rul)
            
            # Get basedOnCoupon
            for coupon in self.graph.objects(prediction, URIRef(self.ex.basedOnCoupon)):
                pred_data['coupon'] = str(coupon).split('/')[-1]
            
            predictions.append(pred_data)
        
        return predictions
    
    def get_coupon_info(self, coupon_name: str) -> Dict:
        """
        Get detailed information about a specific coupon.
        
        Args:
            coupon_name: Name of the coupon (e.g., 'L2S17')
        
        Returns:
            Dictionary with coupon information
        """
        coupon_uri = URIRef(self.ex[coupon_name])
        info = {
            'name': coupon_name,
            'layup': None,
            'features': []
        }
        
        # Get layup
        for layup in self.graph.objects(coupon_uri, URIRef(self.ex.hasLayup)):
            info['layup'] = str(layup).split('/')[-1]
        
        # Get features
        for feature in self.graph.objects(coupon_uri, URIRef(self.ex.hasFeature)):
            feature_name = str(feature).split('/')[-1]
            
            # Get feature value and cycle
            feature_data = {'name': feature_name, 'value': None, 'cycle': None}
            
            for value in self.graph.objects(feature, URIRef(self.ex.hasValue)):
                feature_data['value'] = float(value)
            
            for cycle in self.graph.objects(feature, URIRef(self.ex.measuredAtCycle)):
                feature_data['cycle'] = int(cycle)
            
            info['features'].append(feature_data)
        
        return info
    
    def query_sparql(self, query: str) -> List:
        """
        Execute a SPARQL query on the ontology.
        
        Args:
            query: SPARQL query string
        
        Returns:
            List of query results
        """
        results = self.graph.query(query)
        return list(results)
    
    def export_to_networkx(self):
        """
        Export the RDF graph to NetworkX format.
        
        Returns:
            NetworkX DiGraph representation of the ontology
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        
        G = nx.DiGraph()
        
        # Add nodes and edges from RDF triples
        for s, p, o in self.graph:
            source = str(s).split('/')[-1]
            predicate = str(p).split('/')[-1]
            target = str(o).split('/')[-1] if isinstance(o, URIRef) else str(o)
            
            G.add_node(source)
            if isinstance(o, URIRef):
                G.add_node(target)
                G.add_edge(source, target, relation=predicate)
            else:
                # Add literal as node attribute
                if source in G.nodes:
                    G.nodes[source][predicate] = target
        
        logger.info(f"Exported to NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def print_summary(self):
        """Print a summary of the ontology."""
        print("="*80)
        print("CFRP ONTOLOGY SUMMARY")
        print("="*80)
        print(f"\nTotal triples: {len(self.graph)}")
        
        print(f"\nClasses ({len(self.get_all_classes())}):")
        for cls in self.get_all_classes():
            print(f"  - {cls}")
        
        properties = self.get_all_properties()
        print(f"\nObject Properties ({len(properties['object_properties'])}):")
        for prop in properties['object_properties']:
            print(f"  - {prop}")
        
        print(f"\nDatatype Properties ({len(properties['datatype_properties'])}):")
        for prop in properties['datatype_properties']:
            print(f"  - {prop}")
        
        print(f"\nFeatures ({len(self.get_features())}):")
        for feature in self.get_features():
            print(f"  - {feature}")
        
        print(f"\nDamage Modes ({len(self.get_damage_modes())}):")
        for damage in self.get_damage_modes():
            print(f"  - {damage}")
        
        print(f"\nFeature Correlations:")
        for feature, damages in self.get_feature_correlations().items():
            print(f"  {feature} → {', '.join(damages)}")
        
        print(f"\nRUL Predictions ({len(self.get_rul_predictions())}):")
        for pred in self.get_rul_predictions():
            print(f"  {pred['name']}: RUL={pred['rul']}, Coupon={pred['coupon']}")
        
        print("="*80)


def main():
    """Example usage of the ontology loader."""
    try:
        loader = CFRPOntologyLoader()
        loader.print_summary()
        
        # Example: Get coupon information
        print("\n" + "="*80)
        print("COUPON L2S17 DETAILS")
        print("="*80)
        coupon_info = loader.get_coupon_info('L2S17')
        print(f"Coupon: {coupon_info['name']}")
        print(f"Layup: {coupon_info['layup']}")
        print(f"Features:")
        for feat in coupon_info['features']:
            print(f"  - {feat['name']}: {feat['value']} @ cycle {feat['cycle']}")
        
        # Example: Convert to NetworkX
        print("\n" + "="*80)
        print("NETWORKX EXPORT")
        print("="*80)
        nx_graph = loader.export_to_networkx()
        print(f"Nodes: {nx_graph.number_of_nodes()}")
        print(f"Edges: {nx_graph.number_of_edges()}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
