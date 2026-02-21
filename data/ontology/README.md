# CFRP Composite Materials Knowledge Graph Ontology

This directory contains the RDF/OWL ontology for CFRP (Carbon Fiber Reinforced Polymer) composite materials analysis and Remaining Useful Life (RUL) prediction.

## Overview

The ontology defines the domain knowledge for:
- **Material properties** (e.g., CFRP_T700G)
- **Sensor features** (PZT, strain, X-ray measurements)
- **Damage modes** (matrix cracks, delamination)
- **Test specimens** (coupons, layups)
- **RUL predictions** and their relationships

## File Structure

```
data/ontology/
├── cfrp_ontology.ttl      # Main ontology in Turtle format
└── README.md              # This file
```

## Ontology Structure

### Classes

- **Material**: Base material types (e.g., CFRP_T700G)
- **Feature**: Measurable sensor features
- **DamageMode**: Types of damage (abstract)
  - **MatrixCrack**: Matrix cracking damage
  - **Delamination**: Delamination damage
- **Layup**: Fiber layup configurations
- **Coupon**: Test specimen
- **RULPrediction**: Predicted remaining useful life
- **FatigueTest**: Fatigue testing information

### Object Properties

- `correlatesWith`: Links features to damage modes
- `hasFeature`: Links coupons to their measured features
- `hasLayup`: Links coupons to their layup configuration
- `influencesRUL`: Links damage modes to RUL predictions
- `basedOnCoupon`: Links RUL predictions to coupons

### Datatype Properties

- `hasValue`: Numerical value of a feature (float)
- `measuredAtCycle`: Fatigue cycle when measurement was taken (integer)
- `predictsRUL`: Predicted RUL value in cycles (integer)

## Example Instances

### Features
- `DeltaPSD` - Change in Power Spectral Density (correlates with matrix cracks)
- `DeltaToF` - Change in Time of Flight (correlates with delamination)
- `ScatterEnergy` - Scattering energy (correlates with delamination)

### Damage Modes
- `MatrixCrackDensity` - Density of matrix cracks
- `DelaminationArea` - Area of delamination

### Test Specimens
- `L2S17` - Layup 2, Specimen 17
  - Layup: `Layup2`
  - Feature: `L2S17_DeltaPSD_400k` (0.82 @ 400k cycles)

### Predictions
- `Prediction_001` - Predicts 20,000 cycles RUL for coupon L2S17

## Usage

### Python (using rdflib)

```python
from src.utils.ontology_loader import CFRPOntologyLoader

# Load ontology
loader = CFRPOntologyLoader()

# Print summary
loader.print_summary()

# Get feature correlations
correlations = loader.get_feature_correlations()
print(correlations)
# {'DeltaPSD': ['MatrixCrackDensity'], 
#  'DeltaToF': ['DelaminationArea'], ...}

# Get coupon information
coupon_info = loader.get_coupon_info('L2S17')
print(coupon_info)

# Query with SPARQL
query = """
PREFIX ex: <http://example.org/cfrp/>
SELECT ?feature ?damage
WHERE {
    ?feature ex:correlatesWith ?damage .
}
"""
results = loader.query_sparql(query)

# Export to NetworkX for integration
nx_graph = loader.export_to_networkx()
```

### Jupyter Notebook

See [kg_xai_system.ipynb](../../notebooks/kg_xai_system.ipynb) for interactive examples.

## Extending the Ontology

### Adding New Features

```turtle
ex:NewFeature a ex:Feature ;
    ex:correlatesWith ex:SomeaDamageMode .
```

### Adding New Coupons

```turtle
ex:L1S01 a ex:Coupon ;
    ex:hasLayup ex:Layup1 ;
    ex:hasFeature ex:L1S01_DeltaPSD_300k .

ex:L1S01_DeltaPSD_300k a ex:Feature ;
    ex:hasValue "0.65"^^xsd:float ;
    ex:measuredAtCycle 300000 .
```

### Adding New Predictions

```turtle
ex:Prediction_002 a ex:RULPrediction ;
    ex:basedOnCoupon ex:L1S01 ;
    ex:predictsRUL 35000 .
```

## Integration with XAI System

The ontology can be integrated with the explainable AI system to:

1. **Validate predictions** against domain knowledge
2. **Generate explanations** using semantic relationships
3. **Reason about feature importance** based on correlations
4. **Track prediction provenance** (which features influenced which predictions)

## Tools and Validation

### Validation

Use [Protégé](https://protege.stanford.edu/) to:
- Visualize the ontology
- Check consistency
- Perform reasoning
- Add annotations

### SPARQL Queries

Example queries for semantic reasoning:

```sparql
# Find all features that influence RUL through damage modes
PREFIX ex: <http://example.org/cfrp/>
SELECT ?feature ?damage ?prediction
WHERE {
    ?feature ex:correlatesWith ?damage .
    ?damage ex:influencesRUL ?prediction .
}

# Get all coupons with measurements above a threshold
PREFIX ex: <http://example.org/cfrp/>
SELECT ?coupon ?feature ?value
WHERE {
    ?coupon ex:hasFeature ?feature .
    ?feature ex:hasValue ?value .
    FILTER (?value > 0.7)
}
```

## Dependencies

- **Python**: `rdflib` for RDF/OWL processing
- **Optional**: `owlrl` for reasoning, `rdflib-jsonld` for JSON-LD support

Install with:
```bash
pip install rdflib owlrl rdflib-jsonld
```

## References

- [RDF 1.1 Turtle](https://www.w3.org/TR/turtle/)
- [OWL 2 Web Ontology Language](https://www.w3.org/TR/owl2-overview/)
- [rdflib Documentation](https://rdflib.readthedocs.io/)

## Future Enhancements

- [ ] Add temporal ontology for time-series data
- [ ] Include material properties (fiber angle, resin type, etc.)
- [ ] Add environmental conditions (temperature, humidity)
- [ ] Link to experimental protocols and standards
- [ ] Add uncertainty quantification properties
- [ ] Include maintenance and inspection ontology
