"""The Task 3 architecture, importable without running the notebook.

`task3_build.py` is a linear script: importing it loads the catalogue, builds splits and
trains. Inference needs only the module definitions, so they live here and the script
keeps its own copy for the narrative. The two must stay identical -- the parameter names
are what `torch.load` matches against, so a rename here silently breaks every saved
checkpoint.
"""
from torch import nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class Backbone(nn.Module):
    OUT = 128

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


class Net(nn.Module):
    """One backbone, one linear head per entry in `heads` ({name: n_classes}).

    Design A holds two of these with one head each; B holds one with a single 24-way
    head; C holds one with two heads. Same class, so the comparison cannot be confounded
    by an accidental difference in the body.
    """

    def __init__(self, heads, dropout=0.2):
        super().__init__()
        self.backbone = Backbone()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(Backbone.OUT, n) for k, n in heads.items()})

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return {k: h(f) for k, h in self.heads.items()}
