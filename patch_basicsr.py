"""Patch basicsr/gfpgan to fix torchvision.transforms.functional_tensor import."""
import os

for pkg in ['basicsr', 'gfpgan']:
    try:
        mod = __import__(pkg)
        root = mod.__path__[0]
    except Exception:
        continue
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dp, fn)
            txt = open(fp).read()
            if 'functional_tensor' in txt:
                open(fp, 'w').write(txt.replace(
                    'torchvision.transforms.functional_tensor',
                    'torchvision.transforms.functional'
                ))
                print(f'Patched: {fp}')
