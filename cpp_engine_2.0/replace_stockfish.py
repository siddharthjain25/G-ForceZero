import os

dirs = ['src/eval_backend', 'src/sf_bridge', 'src']
for d in dirs:
    if not os.path.isdir(d): continue
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(('.cpp', '.h', '.hpp', 'CMakeLists.txt', '.md', '.txt')):
                path = os.path.join(root, f)
                with open(path, 'r') as file:
                    content = file.read()
                
                new_content = content.replace('Stockfish', 'GForce').replace('stockfish', 'gforce').replace('STOCKFISH', 'GFORCE')
                
                if new_content != content:
                    with open(path, 'w') as file:
                        file.write(new_content)
                    print(f"Updated {path}")
