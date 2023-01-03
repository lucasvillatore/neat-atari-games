### Instalação das dependências

```bash
pip install -r requirements.txt
```

### Instalação Wrapper
```bash

git clone git://github.com/mila-iqia/atari-representation-learning.git
cd atari-representation-learning
sudo python3 setup.py install
```

### Treinando as redes

```bash
python3 run_all.py
```

### Valores do .env

```
GENERATIONS = Total de gerações pra treinar
EPISODES = Número de treinamentos por geração
MAX_STEPS = Número máximo de steps que um treinamento de geração pode fazer
NUM_CORES = Número workers pra treinar a rede | valores >= 1
RENDER=Renderizar o jogo como modo humano | 0 = Não renderizar | 1 = Renderizar
```