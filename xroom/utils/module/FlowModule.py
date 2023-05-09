import concurrent.futures
import time

class FlowModule:
    
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, flow_stream, *args, **kwargs):
        raise NotImplementedError
    
class FlowModuleList(FlowModule):
    
    def __init__(self, modules):
        self.modules = modules
        self.time_count = {}
    
    def run_time(self):
        return self.time_count
    
    def forward(self, *args, **kwargs):
        data = self.modules[0].forward(*args, **kwargs)
        self.time_count = {}
        for i, module in enumerate(self.modules[1:]):
            start_time = time.time()
            
            data = module.forward(*data)
            
            if isinstance(module, FlowModuleBranch) or isinstance(module, FlowModuleList):
                self.time_count[str(i)+str(type(module))] = (module.run_time())
            else:
                self.time_count[str(i)+str(type(module))] = (time.time()-start_time)
                
        return data
    
class FlowModuleBranch(FlowModule):
    
    def __init__(self, branches, multi_process=0):
        self.branches = branches
        self.multi_process = multi_process
        self.time_count = {}
    
    def run_time(self):
        return self.time_count
    
    def forward(self, *args, **kwargs):
        data = {}
        self.time_count = {}
        if self.multi_process > 1:
            task = {}
            start_time = time.time()
            self.pool = concurrent.futures.ThreadPoolExecutor(self.multi_process)
            for module in self.branches:
                task[module] = self.pool.submit(self.branches[module].forward, *args, **kwargs)
            self.pool.shutdown(True)
            
            for module in task:
                data[module] = task[module].result()
            self.time_count = time.time()-start_time
                
        else:
            self.time_count = {}
            for module in self.branches:
                start_time = time.time()
                
                data[module] = (self.branches[module].forward(*args, **kwargs))
                
                if isinstance(self.branches[module], FlowModuleBranch) or isinstance(self.branches[module], FlowModuleList):
                    self.time_count[module] = (self.branches[module].run_time())
                else:
                    self.time_count[module] = (time.time()-start_time)
            
        return data
    
class FlowModuleAsync(FlowModule):
    
    def __init__(self, module):
        self.module = module
        self.queue = []
        self.parallel = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
    def forward(self, *args, **kwargs):
        task = self.parallel.submit(self.module.forward, *args, **kwargs)
        self.queue.append(task)
        
        for i in range(0, len(self.queue)):
            if self.queue[i].done():
                result = self.queue[i].result()
                self.queue = self.queue[:i] + self.queue[i+1:]
                return [result, len(self.queue)+1-i]
        return [None, -1]