"""Multiple progressive neural network variations."""
from copy import deepcopy
from math import sqrt, prod
from typing import List, Dict, Union, Optional, Any, Tuple, Callable, Iterable, Set

import torch
from torch import nn, Tensor
from torch.nn import functional as F

def nested_module_dict(module_dict: nn.ModuleDict,
                       name: str,
                       module: nn.Module) -> nn.ModuleDict:
    """Creates a nested ModuleDict.

    The name is split by "." and each part of the split defines a new level in
    the nested dict.

    Args:
        module_dict: A ModuleDict
        name: a name containing arbitrary many "."
        module: the module that should be added in the final level

    Returns:
        The nested ModuleDict

    Examples:
        >>> name = 'test.1.test'
        >>> module = nn.Linear(10, 20)
        >>> nested_module_dict(nn.ModuleDict({}), name, module)
        ModuleDict(
            (test): ModuleDict(
                (1): ModuleDict(
                    (test): Linear(in_features=10, out_features=20, bias=True)
                    )
                )
            )
    """
    names = name.split('.')
    current_module_dict = module_dict
    for name in names[:-1]:
        new_module_dict = nn.ModuleDict({})
        current_module_dict[name] = new_module_dict
        current_module_dict = new_module_dict
    current_module_dict[names[-1]] = module
    return module_dict


class LateralConnection(nn.Module):
    """The lateral connection.

    This module aggregates and combines the records from the source layers.
    """

    def __init__(self,
                 source_layer_names: List[str],
                 forward_record: Dict[nn.Module, Tensor],
                 lateral_layer_names: List[str],
                 name_to_module: Callable[[str], nn.Module]):
        """LateralConnection initializer.

        Args:
            source_layer_names: The names of the source layers
            forward_record: A dict mapping modules on their input to their
                forward method
            lateral_layer_names: The names of the lateral layers
            name_to_module: A function mapping a name to its module
        """
        super().__init__()
        self.source_layer_names = source_layer_names
        self.forward_record = forward_record
        self.lateral_layer_names = lateral_layer_names
        self.name_to_module = name_to_module

    def forward(self, x):
        num_connections = len(self.source_layer_names) + 1
        x = x / num_connections
        for source_layer_name, lateral_layer_name in zip(self.source_layer_names, self.lateral_layer_names):
            source_layer = self.name_to_module(source_layer_name)
            assert source_layer in self.forward_record.keys()
            lateral_layer = self.name_to_module(lateral_layer_name)
            x = x + lateral_layer(self.forward_record[source_layer]) / num_connections
        return x


def replace_module(old_module_name: str,
                   new_module: nn.Module,
                   network: nn.Module):
    """ Replaces old_module with new_module in network.

    Args:
        old_module_name: The name of the old module
        new_module: The new module
        network: The network (should contain the old_module)
    """
    name_to_module = dict(network.named_modules())
    assert old_module_name in name_to_module.keys()
    old_modules = old_module_name.split('.')
    module_container_name, module_name = '.'.join(old_modules[:-1]), old_modules[-1]
    module_container = name_to_module[module_container_name] if module_container_name else network
    module_container.__setattr__(module_name, new_module)


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Networks (PNN).

    This network architecture adds weighted layers (lateral connections) between
    layers of previously trained and newly added columns that are jointly
    trained with the new column.

    We use forward hooks to implement PNN for arbitrary network structures.
    """

    def __init__(self,
                 base_network: nn.Module,
                 backbone: nn.Module = None,
                 last_layer_name: Optional[str] = None,
                 lateral_connections: Optional[List[str]] = None):
        """ProgressiveNeuralNetwork initializer.

        Args:
            base_network: A PyTorch model
            backbone: A PyTorch model that is used as a backbone for the
                Progressive Neural Network. If not provided, no backbone is used.
            last_layer_name: The name of the last layer
            lateral_connections: The names of the layers that should have
                lateral connections
        """
        super().__init__()
        self.backbone = backbone if backbone is not None else nn.Sequential()
        self.backbone.eval()
        self.base_network = base_network

        if lateral_connections is None:
            lateral_connections = []
        all_names = [name for name, _ in base_network.named_modules()]
        assert set(lateral_connections).issubset(all_names), \
            f'All lateral connections should be in the base network. ' \
            f'Given {lateral_connections} but only {all_names} are available.'

        self.lateral_connections = lateral_connections

        self.is_classification: List[bool] = []

        self.networks: nn.ModuleList[nn.ModuleList[Union[nn.Module, nn.ModuleDict]]] \
            = nn.ModuleList([])
        
        self.networks_maxlogits: List[float] = []
        self.networks_labels: List[str] = []

        self.last_layer_name = last_layer_name

        self.forward_record = {}
        self.lateral_forward_pre_hooks = []
        self.name_to_module = {}
        self._update_base_network()

    @property
    def previous_tasks(self):
        """The number of previously finished tasks.

        Returns:
            The number of previously finished tasks.
        """
        return len(self.networks) + 1

    def _update_base_network(self):
        base_network_name_to_module = dict(self.base_network.named_modules())
        for lateral_connection in self.lateral_connections:
            lateral_module = base_network_name_to_module[lateral_connection]
            replace_module(lateral_connection,
                           nn.Sequential(
                               lateral_module,
                               LateralConnection([], {}, [], None)
                           ),
                           self.base_network)

    def full_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the whole state.

        Returns:
            A dictionary containing the whole state
        """
        full_state_dict = {
            'last_layer_name': self.last_layer_name,
            'networks': self.networks.state_dict(),
            'base_network': self.base_network.state_dict(),
            'base_network_string': str(self.base_network),
            'backbone': self.backbone.state_dict(),
            'backbone_string': str(self.backbone),
            'lateral_connections': self.lateral_connections,
            'is_classification': self.is_classification,
            'networks_maxlogits': self.networks_maxlogits,
            'networks_labels': self.networks_labels
        }
        return full_state_dict

    def load_full_state_dict(self, full_state_dict: Dict[str, Any]):
        """Copies the whole state from full_state_dict.

        Args:
            full_state_dict: A dict containing the full state
        """
        self.lateral_connections = full_state_dict['lateral_connections']
        self.last_layer_name = full_state_dict['last_layer_name']
        if not any(isinstance(module, LateralConnection) for module in self.base_network.modules()):
            self._update_base_network()
        self.base_network.load_state_dict(full_state_dict['base_network'])
        self.backbone.load_state_dict(full_state_dict['backbone'])
        if full_state_dict['networks']:
            num_columns = max(int(key[0]) for key in full_state_dict['networks'].keys()) + 1
            for i in range(num_columns):
                ending = '.0' if self.last_layer_name in self.lateral_connections else ''
                output_size = full_state_dict['networks'][f'{i}.{i}.{self.last_layer_name}{ending}.weight'].shape[0]
                self.add_new_column(full_state_dict['is_classification'][i], output_size, False, False)
            self.networks.load_state_dict(full_state_dict['networks'])
            self.networks_maxlogits = full_state_dict['networks_maxlogits']
            self.networks_labels = full_state_dict['networks_labels']
    def train(self, mode: bool = True):
        """Sets the module in training mode.

        This means that the last column is trained.

        Args:
            mode: whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.

        Returns:
            self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.networks:
            self.networks[-1].train(mode)
        return self      

    def named_modules(self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        backbone_modules = set(self.backbone.modules())
        memo = memo | backbone_modules
        return super().named_modules(memo, prefix, remove_duplicate)

    def _lateral_forward_pre_hook(self, module, input):
        self.forward_record[module] = input[0]

    def _name_to_module(self, name):
        if not self.name_to_module:
            self.name_to_module = dict(self.networks.named_modules())
        return self.name_to_module[name]

    def add_new_column(self,
                       is_classification: bool = True,
                       output_size: Optional[int] = None,
                       differ_from_previous: bool = False,
                       resample_base_network: bool = False):
        """Adds a new column.

        Args:
            is_classification: Whether the new column is a classification or a
                regression task.
            output_size: The dimension of the output of the last layer of the
                new column (is usually the number of classes in the
                corresponding dataset)
            differ_from_previous: Whether the new weights should be altered
                slightly
            resample_base_network: Whether the weights should be resampled
                completely
        """
        self.is_classification.append(is_classification)

        self.base_network.requires_grad_(False)
        self.networks.requires_grad_(False)
        self.networks.eval()

        intra_column = deepcopy(self.base_network)
        if output_size is not None:
            assert self.last_layer_name not in self.lateral_connections
            last_layer = dict(intra_column.named_modules())[self.last_layer_name]
            assert isinstance(last_layer, nn.Linear)
            new_last_layer = nn.Linear(last_layer.in_features, output_size, bias=last_layer.bias is not None) \
                .requires_grad_(False)
            replace_module(self.last_layer_name, new_last_layer, intra_column)

        if resample_base_network:
            intra_column.apply(lambda module: module.reset_parameters()
            if hasattr(module, 'reset_parameters') else None)

        inter_column = [nn.ModuleDict({}) for _ in self.networks]
        for hook in self.lateral_forward_pre_hooks:
            hook.remove()
        for lateral_connection in self.lateral_connections:
            name = f'{lateral_connection}.0'

            self.lateral_forward_pre_hooks.append(
                dict(intra_column.named_modules())[name].register_forward_pre_hook(
                    self._lateral_forward_pre_hook))

            for inter_column_connections, column_previous in zip(inter_column, self.networks):
                column_previous_name_to_module = dict(column_previous[-1].named_modules())
                new_module = deepcopy(column_previous_name_to_module[name])

                self.lateral_forward_pre_hooks.append(
                    column_previous_name_to_module[name].register_forward_pre_hook(
                        self._lateral_forward_pre_hook))

                nested_module_dict(inter_column_connections, name, new_module)
            replace_module(f'{lateral_connection}.1',
                           LateralConnection(
                               [f'{i}.{i}.{lateral_connection}.0' for i in range(len(self.networks))],
                               self.forward_record,
                               [f'{len(self.networks)}.{i}.{lateral_connection}.0' for i in range(len(self.networks))],
                               self._name_to_module),
                           intra_column)

        new_network = nn.ModuleList([*inter_column, intra_column])

        if differ_from_previous:
            for param in new_network.parameters():
                param += (torch.randn_like(param) / sqrt(prod(param.shape))) \
                         * torch.linalg.norm(param) * torch.finfo(param.dtype).eps

        self.networks.append(new_network.requires_grad_(True))
        self.name_to_module = dict(self.networks.named_modules())
        torch.cuda.empty_cache()

    def forward(self, x: Tensor):
        # assert self.networks, 'no column is available, please call add_new_column before forward_once'
        # x = self.backbone(x)
        # out = [column[-1](x) for column in self.networks]
        # self.forward_record.clear()
        # return out

        assert self.networks, 'no column is available, please call add_new_column before forward_once'

        if not isinstance(self.backbone, nn.Sequential):
            x = self.backbone(x) # [num_img, 384]

            # class_tokens = x[:,0].unsqueeze(1)
            # patch_tokens = x[:,1:]
            # avg_patch_tokens = torch.mean(patch_tokens, dim=1).unsqueeze(1)
            # # x is now [num_img, 2] with class_tokens and avg_patch_tokens
            # x = torch.cat((class_tokens, avg_patch_tokens), dim=1)

        out = [column[-1](x) for column in self.networks]
        self.forward_record.clear()
        return out

    # Reset parameters of given column
    def reset_parameters(self, label: str):
        """Resets the parameters of the given column.

        Args:
            column: The column to reset
        """
        column = self.networks_labels.index(label)
        self.networks[column].apply(lambda module: module.reset_parameters()
        if hasattr(module, 'reset_parameters') else None)

class ClassifierNet(nn.Module): # input size: 384, output size: 1 -> self.fc = nn.Linear(384, 1)
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(in_features=input_size, out_features=288)
        self.hidden_1 = nn.Linear(in_features=288, out_features=192)
        self.hidden_2 = nn.Linear(in_features=192, out_features=144)
        self.hidden_3 = nn.Linear(in_features=144, out_features=96)
        self.hidden_4 = nn.Linear(in_features=96, out_features=64)
        self.hidden_5 = nn.Linear(in_features=64, out_features=32)
        self.hidden_6 = nn.Linear(in_features=32, out_features=16)
        self.hidden_7 = nn.Linear(in_features=16, out_features=8)
        self.fc = nn.Linear(in_features=8, out_features=output_size)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = F.relu(self.hidden_6(x))
        x = F.relu(self.hidden_7(x))
        return self.fc(x)