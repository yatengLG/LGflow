# -*- coding: utf-8 -*-
# @Author  : LG


from collections import OrderedDict
from ..parameter import Parameter


class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("_parameters cannot assign after __init__")

            self.register_parameter(key, value)

        elif isinstance(value, Module):

            if modules is None:
                raise AttributeError("_modules cannot assign after __init__")

            self.register_module(key, value)
        else:

            if value is None:
                self.register_parameter(key, value)
            elif not isinstance(value, OrderedDict):
                self.register_buffer(key, value)

    def register_buffer(self, key, value):
        assert isinstance(key, str)
        self._buffers[key] = value

    def register_module(self, key, value):
        assert isinstance(key, str)
        self._modules[key] = value


    def register_parameter(self, key, value):
        assert isinstance(key, str)

        if "." in key:
            raise ValueError("参数名不允许'.'存在")

        if value is None:
            self._parameters[key] = None

        elif not isinstance(value, Parameter):
            raise TypeError("参数必须是 LG_flow.Parameter 类型,但现在类型是: {}".format(type(value)))

        elif value.grad_fn:
            raise ValueError("参数必须不是计算得到的张量,出错: {}".format(key))

        else:
            self._parameters[key] = value

    def parameters(self):
        params_dic = {}

        def get_params(module_name, module):
            params = module._parameters
            for name, param in params.items():
                params_dic["{}.{}".format(module_name, name)] = param
            modules = module._modules
            for module in modules.items():
                get_params(module[0], module[1])

        get_params("",self)

        return params_dic

    def train(self, mode: bool=True):
        def set_module_train(module, mode):
            module.training = mode
            modules = module._modules
            for _, module in modules.items():
                set_module_train(module, mode)

        set_module_train(self, mode)

