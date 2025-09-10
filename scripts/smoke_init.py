import importlib.util, importlib.machinery, os, sys

MONOLITH = 'e8_mind_server_M18.7.py'
path = os.path.join(os.path.dirname(__file__), '..', MONOLITH)
path = os.path.abspath(path)
loader = importlib.machinery.SourceFileLoader('e8_mind_smoke', path)
spec = importlib.util.spec_from_loader(loader.name, loader)
if spec is None:
    print('Failed to create spec for monolith', file=sys.stderr)
    sys.exit(1)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)  # type: ignore
E8Mind = getattr(mod, 'E8Mind')
try:
    mind = E8Mind.__new__(E8Mind)  # bypass original heavy __init__
    mind._init_minimal(semantic_domain_val='general', run_id='smoke_init', llm_client_instance=None,
                       client_model='dummy', embedding_model_name='dummy', embed_adapter=None, embed_in_dim=256)
    print('[SMOKE] Minimal init ok. action_dim attr?', hasattr(mind,'action_dim'))
    # Try safe call of apply_manifold_action with dummy vector if layout present
    if hasattr(mod, 'ACTION_LAYOUT'):
        layout = getattr(mod, 'ACTION_LAYOUT')
        if layout:
            size = 0
            for lay in layout:
                size = max(size, int(lay.get('biv_start',0))+int(lay.get('biv_len',0))+1)
                size = max(size, int(lay.get('angle_idx',0))+1)
            import numpy as _np
            dummy = _np.zeros(size, dtype=_np.float32)
            if hasattr(mind, 'apply_manifold_action'):
                mind.dimensional_shells = {4: type('S',(),{'spin_with_bivector':lambda *a,**k: None,'dim':4})(),
                                          6: type('S',(),{'spin_with_bivector':lambda *a,**k: None,'dim':6})()}
                mind.proximity_engine = type('PE',(),{'update_shell_index':lambda *a,**k: None})()
                mind.macro_manager = None
                mind.apply_manifold_action(dummy)
                print('[SMOKE] apply_manifold_action executed on dummy vector')
except Exception as e:
    print('[SMOKE] Minimal init failed:', e)
    sys.exit(2)
