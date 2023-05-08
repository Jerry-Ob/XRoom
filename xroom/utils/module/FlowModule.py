class FlowModule:
    
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, flow_stream, *args, **kwargs):
        raise NotImplementedError
    
class FlowModuleList(FlowModule):
    
    def __init__(self, modules):
        self.modules = modules
        
    def forward(self, *args, **kwargs):
        data = self.modules[0].forward(*args, **kwargs)
        for module in self.modules[1:]:
            data = module.forward(*data)
        return data
    
class FlowModuleBranch(FlowModule):
    
    def __init__(self, branches):
        self.branches = branches
    
    def forward(self, *args, **kwargs):
        data = {}
        for module in self.branches:
            data[module] = (self.branches[module].forward(*args, **kwargs))
            
        return data