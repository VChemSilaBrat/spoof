import os
from comet_ml import Experiment


class CometMLWriter:
    """Wrapper for Comet.ml experiment tracking"""
    
    def __init__(self, project_name="asvspoof-lightcnn", experiment_name=None):
        """
        Initialize Comet.ml experiment
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
        """
        api_key = os.environ.get('COMET_API_KEY')
        workspace = os.environ.get('COMET_WORKSPACE')
        
        if not api_key:
            print("Warning: COMET_API_KEY not set. Logging disabled.")
            self.experiment = None
            return
        
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace
        )
        
        if experiment_name:
            self.experiment.set_name(experiment_name)
    
    def log_parameters(self, params):
        """Log hyperparameters"""
        if self.experiment:
            self.experiment.log_parameters(params)
    
    def log_metric(self, name, value, step=None, epoch=None):
        """Log a metric"""
        if self.experiment:
            self.experiment.log_metric(name, value, step=step, epoch=epoch)
    
    def log_metrics(self, metrics, step=None, epoch=None):
        """Log multiple metrics"""
        if self.experiment:
            self.experiment.log_metrics(metrics, step=step, epoch=epoch)
    
    def end(self):
        """End the experiment"""
        if self.experiment:
            self.experiment.end()
