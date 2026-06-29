import wandb
import logging
from typing import Dict, Any

logger = logging.getLogger("DrishtiLogger")

class ExperimentTracker:
    """
    Wrapper for Weights & Biases (wandb) experiment tracking.
    """
    
    def __init__(self, project_name: str, config: Dict[str, Any]):
        """
        Initializes the W&B tracker.
        
        Args:
            project_name (str): The W&B project name.
            config (Dict[str, Any]): Hyperparameters and configurations to log.
        """
        self.project_name = project_name
        self.config = config
        self.run = None

    def start_run(self, run_name: str = None):
        """Starts a W&B run."""
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=self.config,
                reinit=True
            )
            logger.info(f"Successfully initialized W&B run: {self.run.name}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Logs metrics to W&B."""
        if self.run:
            wandb.log(metrics, step=step)
            
    def finish_run(self):
        """Ends the current W&B run."""
        if self.run:
            wandb.finish()
            logger.info("W&B run finished.")
