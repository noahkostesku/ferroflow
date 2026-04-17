import ferroflow as ff
import numpy as np

dag = ff.DAG()

# 3-layer MLP: Linear(784->256)->ReLU->Linear(256->128)->ReLU->Linear(128->10)
l1 = dag.matmul([], [256, 784])
r1 = dag.relu([l1], [256])
l2 = dag.matmul([r1], [128, 256])
r2 = dag.relu([l2], [128])
l3 = dag.matmul([r2], [10, 128])

results = ff.run(dag, workers=4)
print(f"Output shape: {len(results[l3])} values")
print("ferroflow MLP:K")
