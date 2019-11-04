import random
import wandb
wandb.init(project="cloud-classification")

for i in range(10):
    val1 = random.random()
    val2 = random.random()
    wandb.log({"nani": val1, "da fak": val2})
