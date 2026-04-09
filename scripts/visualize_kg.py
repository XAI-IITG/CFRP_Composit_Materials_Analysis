"""
Visualize the CFRP Knowledge Graph

Reads the populated ontology and generates an interactive HTML visualization.

Usage:
    python scripts/visualize_kg.py

Output:
    outputs/kg_visualization.html  (open in browser)
"""

import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

ONTOLOGY_PATH = os.path.join(PROJECT_ROOT, 'data', 'ontology', 'cfrp_ontology_populated.ttl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
OUTPUT_HTML = os.path.join(OUTPUT_DIR, 'kg_visualization.html')

# Color scheme for node types
NODE_COLORS = {
    'Coupon': '#4FC3F7',
    'Material': '#81C784',
    'Layup': '#FFB74D',
    'FatigueCheckpoint': '#CE93D8',
    'StrainMeasurement': '#F48FB1',
    'DamageObservation': '#EF5350',
    'DamageMode': '#FF7043',
    'Feature': '#4DD0E1',
    'BoundaryCondition': '#AED581',
    'LambWaveMode': '#FFD54F',
    'RULPrediction': '#90A4AE',
    'StrainType': '#BA68C8',
    'Schema': '#78909C',
}

NODE_SHAPES = {
    'Coupon': 'box',
    'Material': 'diamond',
    'Layup': 'hexagon',
    'FatigueCheckpoint': 'dot',
    'StrainMeasurement': 'triangle',
    'DamageObservation': 'star',
    'DamageMode': 'triangleDown',
    'Feature': 'square',
    'BoundaryCondition': 'ellipse',
    'StrainType': 'ellipse',
}


def build_graph_data(ontology_path, mode='schema'):
    """Extract graph data from the populated ontology.
    
    Modes:
        'schema' - Show classes and their relationships (compact)
        'specimen' - Show one specimen's full data
        'overview' - Show specimens + checkpoints (medium detail)
    """
    from rdflib import Graph, Namespace, RDF, RDFS, OWL
    from rdflib.term import URIRef, Literal

    g = Graph()
    g.parse(ontology_path, format='turtle')
    ex = Namespace("http://example.org/cfrp/")

    nodes = []
    edges = []
    node_ids = set()

    def add_node(node_id, label, group, title='', size=20):
        if node_id not in node_ids:
            node_ids.add(node_id)
            nodes.append({
                'id': node_id,
                'label': label[:30],
                'title': title or label,
                'group': group,
                'color': NODE_COLORS.get(group, '#90A4AE'),
                'shape': NODE_SHAPES.get(group, 'dot'),
                'size': size,
            })

    def add_edge(src, tgt, label='', width=1, color='#999999'):
        edges.append({
            'from': src,
            'to': tgt,
            'label': label,
            'width': width,
            'color': color,
            'font': {'size': 9, 'color': '#666666'},
            'arrows': 'to',
        })

    if mode == 'schema':
        # ── Schema-level: show classes, properties, and key instances ──

        # 1. Material
        add_node('CFRP_T700G', 'CFRP T700G', 'Material',
                 'Torayca T700G Carbon Prepreg', 30)

        # 2. Layups
        for layup_uri in g.subjects(RDF.type, ex.Layup):
            name = str(layup_uri).split('/')[-1]
            label_val = ''
            for lbl in g.objects(layup_uri, RDFS.label):
                label_val = str(lbl)
            add_node(name, label_val or name, 'Layup', label_val, 25)
            add_edge(name, 'CFRP_T700G', 'hasMaterial', 1, '#81C784')

        # 3. Coupons
        for coupon_uri in g.subjects(RDF.type, ex.Coupon):
            name = str(coupon_uri).split('/')[-1]
            coupon_id = ''
            for cid in g.objects(coupon_uri, ex.couponID):
                coupon_id = str(cid)
            layup = ''
            for l in g.objects(coupon_uri, ex.hasLayup):
                layup = str(l).split('/')[-1]

            # Count checkpoints
            cp_count = sum(1 for _ in g.objects(coupon_uri, ex.hasCheckpoint))
            title = f"Coupon: {coupon_id}\nLayup: {layup}\nCheckpoints: {cp_count}"
            add_node(name, coupon_id or name, 'Coupon', title, 22)
            if layup:
                add_edge(name, layup, 'hasLayup', 2, '#FFB74D')

        # 4. Boundary Conditions
        for bc_uri in g.subjects(RDF.type, ex.BoundaryCondition):
            name = str(bc_uri).split('/')[-1]
            label_val = ''
            for lbl in g.objects(bc_uri, RDFS.label):
                label_val = str(lbl)
            add_node(name, label_val or name, 'BoundaryCondition', label_val, 18)

        # 5. Damage modes
        for dm_cls in [ex.MatrixCrack, ex.Delamination, ex.Fracture]:
            for dm_uri in g.subjects(RDF.type, dm_cls):
                name = str(dm_uri).split('/')[-1]
                label_val = ''
                for lbl in g.objects(dm_uri, RDFS.label):
                    label_val = str(lbl)
                add_node(name, label_val or name, 'DamageMode', label_val, 22)

        # Damage causal chain
        for s, o in g.subject_objects(ex.precedes):
            sn = str(s).split('/')[-1]
            on = str(o).split('/')[-1]
            if sn in node_ids and on in node_ids:
                add_edge(sn, on, 'precedes', 3, '#FF7043')

        # 6. Features (CI)
        for feat_uri in g.subjects(RDF.type, ex.Feature):
            name = str(feat_uri).split('/')[-1]
            # Skip timestamped measurement instances
            if any(c.isdigit() for c in name.split('_')[-1:][0] if len(name.split('_')[-1:][0]) > 2):
                continue
            label_val = ''
            for lbl in g.objects(feat_uri, RDFS.label):
                label_val = str(lbl)
            add_node(name, label_val or name, 'Feature', label_val, 18)

            # Feature → DamageMode
            for dm in g.objects(feat_uri, ex.correlatesWith):
                dmn = str(dm).split('/')[-1]
                if dmn in node_ids:
                    add_edge(name, dmn, 'correlates', 2, '#4DD0E1')

        # 7. Strain Types
        for st_uri in g.subjects(RDF.type, ex.StrainType):
            name = str(st_uri).split('/')[-1]
            label_val = ''
            for lbl in g.objects(st_uri, RDFS.label):
                label_val = str(lbl)
            add_node(name, label_val or name, 'StrainType', label_val, 16)

        # 8. Lamb Wave Modes
        for lw_uri in g.subjects(RDF.type, ex.LambWaveMode):
            name = str(lw_uri).split('/')[-1]
            label_val = ''
            for lbl in g.objects(lw_uri, RDFS.label):
                label_val = str(lbl)
            add_node(name, label_val or name, 'LambWaveMode', label_val, 18)

    elif mode == 'overview':
        # ── Overview: coupons + their checkpoints + damage ──

        for coupon_id in ['L1S11', 'L1S12']:
            coupon_uri = ex[coupon_id]
            add_node(coupon_id, coupon_id, 'Coupon', f'Specimen {coupon_id}', 35)

            for cp_uri in g.objects(coupon_uri, ex.hasCheckpoint):
                cp_name = str(cp_uri).split('/')[-1]
                cycles = 0
                for c in g.objects(cp_uri, ex.atCycleCount):
                    cycles = int(c)

                cp_label = f"{cycles} cyc"
                cp_title = f"Checkpoint: {cp_name}\nCycles: {cycles}"

                # Count measurements
                n_meas = sum(1 for _ in g.objects(cp_uri, ex.hasMeasurement))
                cp_title += f"\nMeasurements: {n_meas}"

                add_node(cp_name, cp_label, 'FatigueCheckpoint', cp_title, 12)
                add_edge(coupon_id, cp_name, '', 1, '#CE93D8')

                # Damage observations
                for obs_uri in g.objects(cp_uri, ex.hasDamageObservation):
                    obs_name = str(obs_uri).split('/')[-1]
                    obs_text = ''
                    for t in g.objects(obs_uri, ex.observationText):
                        obs_text = str(t)
                    cracks = None
                    for cr in g.objects(obs_uri, ex.crackCount):
                        cracks = int(cr)

                    obs_label = f"{cracks} cracks" if cracks else "damage"
                    add_node(obs_name, obs_label, 'DamageObservation',
                             f"Observation at {cycles} cycles:\n{obs_text}", 14)
                    add_edge(cp_name, obs_name, '', 1, '#EF5350')

    elif mode.startswith('specimen:'):
        # ── Detailed view of one specimen ──
        coupon_id = mode.split(':')[1]
        coupon_uri = ex[coupon_id]
        add_node(coupon_id, coupon_id, 'Coupon', f'Specimen {coupon_id}', 40)

        for cp_uri in g.objects(coupon_uri, ex.hasCheckpoint):
            cp_name = str(cp_uri).split('/')[-1]
            cycles = 0
            for c in g.objects(cp_uri, ex.atCycleCount):
                cycles = int(c)

            add_node(cp_name, f"{cycles} cyc", 'FatigueCheckpoint',
                     f"Checkpoint at {cycles} cycles", 15)
            add_edge(coupon_id, cp_name, '', 2, '#CE93D8')

            # Measurements (grouped by strain type)
            strain_groups = {}
            for m_uri in g.objects(cp_uri, ex.hasMeasurement):
                m_name = str(m_uri).split('/')[-1]
                stype = ''
                for st in g.objects(m_uri, ex.hasStrainType):
                    for lbl in g.objects(st, RDFS.label):
                        stype = str(lbl)
                gauge = ''
                for gid in g.objects(m_uri, ex.gaugeID):
                    gauge = str(gid)
                mean_val = None
                for mv in g.objects(m_uri, ex.strainMean):
                    mean_val = float(mv)

                group_key = f"{cp_name}_{stype.replace(' ', '_')}"
                if group_key not in strain_groups:
                    strain_groups[group_key] = {'type': stype, 'gauges': []}
                strain_groups[group_key]['gauges'].append({
                    'gauge': gauge, 'mean': mean_val
                })

            for gk, gv in strain_groups.items():
                gauge_info = '\n'.join(
                    f"  {g['gauge']}: mean={g['mean']:.8f}" if g['mean'] else f"  {g['gauge']}"
                    for g in gv['gauges']
                )
                title = f"{gv['type']} at {cycles} cycles\n{gauge_info}"
                add_node(gk, gv['type'][:10], 'StrainMeasurement', title, 10)
                add_edge(cp_name, gk, '', 1, '#F48FB1')

            # Damage
            for obs_uri in g.objects(cp_uri, ex.hasDamageObservation):
                obs_name = str(obs_uri).split('/')[-1]
                obs_text = ''
                for t in g.objects(obs_uri, ex.observationText):
                    obs_text = str(t)
                add_node(obs_name, 'damage', 'DamageObservation', obs_text, 14)
                add_edge(cp_name, obs_name, '', 2, '#EF5350')

    return nodes, edges


def generate_html(nodes, edges, title="CFRP Knowledge Graph"):
    """Generate an interactive HTML visualization using vis.js."""

    legend_items = []
    seen_groups = set()
    for n in nodes:
        g = n['group']
        if g not in seen_groups:
            seen_groups.add(g)
            legend_items.append({'group': g, 'color': NODE_COLORS.get(g, '#90A4AE')})

    legend_html = ''.join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="width:14px;height:14px;border-radius:50%;background:{li["color"]};'
        f'display:inline-block;margin-right:5px;"></span>{li["group"]}</span>'
        for li in legend_items
    )

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    height: 100vh;
    overflow: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2a2a4a;
    z-index: 10;
  }}
  .header h1 {{
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(135deg, #4FC3F7, #CE93D8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .header .stats {{
    font-size: 13px;
    color: #888;
  }}
  .controls {{
    background: #1a1a2e;
    padding: 10px 24px;
    display: flex;
    gap: 12px;
    align-items: center;
    border-bottom: 1px solid #2a2a4a;
  }}
  .controls button {{
    padding: 6px 16px;
    border: 1px solid #4FC3F7;
    background: transparent;
    color: #4FC3F7;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
  }}
  .controls button:hover, .controls button.active {{
    background: #4FC3F7;
    color: #0f0f1a;
  }}
  .legend {{
    padding: 8px 24px;
    background: #151528;
    font-size: 12px;
    border-bottom: 1px solid #2a2a4a;
    overflow-x: auto;
    white-space: nowrap;
  }}
  #graph {{
    width: 100%;
    height: calc(100vh - 120px);
  }}
  .info-panel {{
    position: absolute;
    bottom: 20px;
    right: 20px;
    background: rgba(26, 26, 46, 0.95);
    border: 1px solid #4FC3F7;
    border-radius: 10px;
    padding: 16px;
    max-width: 350px;
    font-size: 13px;
    display: none;
    backdrop-filter: blur(10px);
    z-index: 100;
  }}
  .info-panel h3 {{
    color: #4FC3F7;
    margin-bottom: 8px;
    font-size: 15px;
  }}
  .info-panel p {{
    color: #bbb;
    white-space: pre-wrap;
    line-height: 1.5;
  }}
  .info-panel .close {{
    position: absolute;
    top: 8px;
    right: 12px;
    cursor: pointer;
    color: #666;
    font-size: 16px;
  }}
</style>
</head>
<body>
<div class="header">
  <h1>🔬 {title}</h1>
  <div class="stats" id="stats"></div>
</div>
<div class="controls">
  <button id="btn-schema" class="active" onclick="switchView('schema')">Schema View</button>
  <button id="btn-overview" onclick="switchView('overview')">Specimens Overview</button>
  <button id="btn-s11" onclick="switchView('specimen_L1S11')">S11 Detail</button>
  <button id="btn-s12" onclick="switchView('specimen_L1S12')">S12 Detail</button>
  <button onclick="network.fit()">⟲ Reset Zoom</button>
  <button onclick="togglePhysics()">⚡ Toggle Physics</button>
</div>
<div class="legend" id="legend">{legend_html}</div>
<div id="graph"></div>
<div class="info-panel" id="info-panel">
  <span class="close" onclick="document.getElementById('info-panel').style.display='none'">✕</span>
  <h3 id="info-title"></h3>
  <p id="info-body"></p>
</div>

<script>
// Pre-loaded data for each view
const viewData = {{
  schema: {{ nodes: {nodes_json}, edges: {edges_json} }},
  overview: null,
  specimen_L1S11: null,
  specimen_L1S12: null,
}};

let network = null;
let physicsEnabled = true;
let currentView = 'schema';

function initNetwork(nodesData, edgesData) {{
  const container = document.getElementById('graph');
  const data = {{
    nodes: new vis.DataSet(nodesData),
    edges: new vis.DataSet(edgesData)
  }};
  const options = {{
    physics: {{
      enabled: physicsEnabled,
      barnesHut: {{
        gravitationalConstant: -3000,
        centralGravity: 0.3,
        springLength: 120,
        springConstant: 0.04,
        damping: 0.09
      }},
      stabilization: {{ iterations: 150 }}
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 100,
      zoomView: true,
      dragView: true,
    }},
    nodes: {{
      font: {{ color: '#e0e0e0', size: 12 }},
      borderWidth: 2,
      shadow: true,
    }},
    edges: {{
      smooth: {{ type: 'continuous' }},
      font: {{ size: 9, color: '#666', strokeWidth: 0 }}
    }}
  }};
  if (network) network.destroy();
  network = new vis.Network(container, data, options);
  
  document.getElementById('stats').textContent = 
    nodesData.length + ' nodes · ' + edgesData.length + ' edges';

  // Click handler
  network.on('click', function(params) {{
    const panel = document.getElementById('info-panel');
    if (params.nodes.length > 0) {{
      const nodeId = params.nodes[0];
      const node = nodesData.find(n => n.id === nodeId);
      if (node) {{
        document.getElementById('info-title').textContent = node.label;
        document.getElementById('info-body').textContent = node.title || '';
        panel.style.display = 'block';
        panel.style.borderColor = node.color;
      }}
    }} else {{
      panel.style.display = 'none';
    }}
  }});
}}

function switchView(view) {{
  currentView = view;
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  const btnId = 'btn-' + view.replace('specimen_', '').toLowerCase();
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.add('active');

  if (viewData[view]) {{
    initNetwork(viewData[view].nodes, viewData[view].edges);
  }} else {{
    // For dynamically loaded views, we embed all data at generation time
    initNetwork(viewData.schema.nodes, viewData.schema.edges);
  }}
}}

function togglePhysics() {{
  physicsEnabled = !physicsEnabled;
  if (network) {{
    network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
  }}
}}

// Initial render
initNetwork(viewData.schema.nodes, viewData.schema.edges);
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("CFRP Knowledge Graph Visualization")
    print("=" * 60)

    if not os.path.exists(ONTOLOGY_PATH):
        print(f"ERROR: Populated ontology not found: {ONTOLOGY_PATH}")
        print("Run 'python scripts/populate_ontology.py' first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate all views
    print("\nGenerating Schema view...")
    schema_nodes, schema_edges = build_graph_data(ONTOLOGY_PATH, mode='schema')
    print(f"  {len(schema_nodes)} nodes, {len(schema_edges)} edges")

    print("Generating Overview view...")
    overview_nodes, overview_edges = build_graph_data(ONTOLOGY_PATH, mode='overview')
    print(f"  {len(overview_nodes)} nodes, {len(overview_edges)} edges")

    print("Generating S11 Detail view...")
    s11_nodes, s11_edges = build_graph_data(ONTOLOGY_PATH, mode='specimen:L1S11')
    print(f"  {len(s11_nodes)} nodes, {len(s11_edges)} edges")

    print("Generating S12 Detail view...")
    s12_nodes, s12_edges = build_graph_data(ONTOLOGY_PATH, mode='specimen:L1S12')
    print(f"  {len(s12_nodes)} nodes, {len(s12_edges)} edges")

    # Generate HTML with all views embedded
    html = generate_html(schema_nodes, schema_edges, "CFRP Knowledge Graph")

    # Inject additional view data
    overview_json = json.dumps({'nodes': overview_nodes, 'edges': overview_edges})
    s11_json = json.dumps({'nodes': s11_nodes, 'edges': s11_edges})
    s12_json = json.dumps({'nodes': s12_nodes, 'edges': s12_edges})

    html = html.replace(
        'overview: null,',
        f'overview: {overview_json},'
    ).replace(
        'specimen_L1S11: null,',
        f'specimen_L1S11: {s11_json},'
    ).replace(
        'specimen_L1S12: null,',
        f'specimen_L1S12: {s12_json},'
    )

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'=' * 60}")
    print(f"Visualization saved to: {OUTPUT_HTML}")
    print(f"Open in your browser to explore the Knowledge Graph!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
