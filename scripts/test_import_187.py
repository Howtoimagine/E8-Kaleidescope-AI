import importlib.util, sys
p = 'e8_mind_server_M18.7.py'
spec = importlib.util.spec_from_file_location('e8mod', p)
mod = importlib.util.module_from_spec(spec)
sys.modules['e8mod']=mod
try:
    spec.loader.exec_module(mod)  # type: ignore
    print('IMPORT_OK')
    print('HAS load_profile', hasattr(mod, 'load_profile'))
    print('DATA_SOURCES keys', list(getattr(mod, 'DATA_SOURCES', {}).keys()))
    # Call a simple function to ensure runtime basics
    if hasattr(mod, 'run_hypothesis_validation'):
        print('run_hypothesis_validation:', mod.run_hypothesis_validation({'text': 'compare models and simulate'}))
except Exception as e:
    import traceback
    print('IMPORT_FAIL:', e)
    traceback.print_exc()
