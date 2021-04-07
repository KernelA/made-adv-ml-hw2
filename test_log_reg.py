import torch
from torch import optim
from sklearn import datasets
from sklearn.metrics import roc_auc_score
import numpy as np

from rating_model import LogisticRegressionTorch, Trainer


def train(x, y, model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, em_step=0, optimizer=optimizer, num_iter=400, device=torch.device("cpu"))

    trainer.fit(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)).reshape(-1, 1))


@torch.no_grad()
def validate(x, y, model):
    model.eval()
    predicted = model.predict_proba(torch.from_numpy(x))
    predicted = predicted.cpu()

    return roc_auc_score(y, predicted.numpy())


if __name__ == "__main__":
    x, y = datasets.make_classification(n_samples=500)
    x = x.astype(np.float32)

    model = LogisticRegressionTorch(x.shape[1])

    train(x, y, model)
    print("ROC AUC: ", validate(x, y, model))
