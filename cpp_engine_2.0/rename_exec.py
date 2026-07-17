import os

with open('CMakeLists.txt', 'r') as f:
    content = f.read()

content = content.replace('nnue_engine', 'gforce_engine')

with open('CMakeLists.txt', 'w') as f:
    f.write(content)

with open('../render/config.yml', 'r') as f:
    content2 = f.read()

content2 = content2.replace('nnue_engine', 'gforce_engine')

with open('../render/config.yml', 'w') as f:
    f.write(content2)

with open('../Dockerfile', 'r') as f:
    content3 = f.read()
content3 = content3.replace('nnue_engine', 'gforce_engine')
with open('../Dockerfile', 'w') as f:
    f.write(content3)

