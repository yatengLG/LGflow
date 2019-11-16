# -*- coding: utf-8 -*-
# @Author  : LG

import LG_flow
import LG_flow.functional as f
from LG_flow import Parameter
from collections import OrderedDict

class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()

    def forward(self, x):
        raise NotImplementedError

    def __setattr__(self, key, value):

        # def remove_from(*dicts):
        #     for d in dicts:
        #         if key in d:
        #             del d[key]
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

            # self.register_parameter(value.)
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



    def __str__(self):
        cont = ""
        for module_name, module in self._modules.items():
            cont += "{} : {}\n".format(module_name, module)
            for param in module._parameters.items():
                cont += "   {}\n".format(param)
        return cont

    def print_struct(self):
        cont = ""
        for module_name, module in self._modules.items():
            cont += "{} : {}\n".format(module_name, module)
            for param in module._parameters.items():
                cont += "   {}\n".format(param)
        print(cont)

    def print_detail(self):
        cont = ""
        for module in self._modules.items():
            cont += "{}\n".format(module)

            module_name, module = module
            for param_name, param in module._parameters.items():
                cont += "{} : {}\n".format(param_name, param)
            cont+="----"*20+"\n"
        print(cont)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(LG_flow.randn(shape=(in_features, out_features), requires_grad=True))

        if bias:
            self.bias = Parameter(LG_flow.zeros(out_features,requires_grad=True))
        else:
            self.bias = None

    def forward(self, x:LG_flow.Tensor):
        return f.linear(x, self.weights, self.bias)

    def __str__(self):
        return "(LG_flow.nn.Linear)"
