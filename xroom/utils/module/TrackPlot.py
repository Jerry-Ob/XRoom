from .FlowModule import FlowModule
import plotly.graph_objects as go
import numpy as np

class TrackPlot(FlowModule):
    
    def __init__(self, max_length = 20, plot_size=10):
        self.previous_path = {}
        self.counter = 0
        self.max_length = max_length
        self.plot_size = plot_size
    
    def forward(self, images, masks, anchors, *args, **kwargs):
        
        points = [((anchor[0]+anchor[2])/2, (anchor[1]+anchor[3])/2)
                  for anchor in anchors]
        
        for i, point in enumerate(points):
            if i not in self.previous_path:
                self.previous_path[i] = []
            self.previous_path[i].append(point)
        
        fig = go.Figure()
        for path in self.previous_path:
            x = [i[0] for i in self.previous_path[path]]
            x.reverse()
            x = x[:self.max_length]
            y = [1-i[1] for i in self.previous_path[path]]
            y.reverse()
            y = y[:self.max_length]
            
            size = np.array(list(range(self.max_length+1, 0, -1)))
            size = (size-np.min(size))/(np.max(size)-np.min(size))
            size = size*self.plot_size
            fig.add_scatter(x=x, y=y, mode='markers', marker_size=size)
        
        fig.update_layout(
            width=800,
            height=400,
            yaxis_range = [0, 1],
            xaxis_range = [0, 1],
            showlegend=False
        )
        return [fig]
        
        
        