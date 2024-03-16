from textSummarizer.logging import logger

from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

def run_stage(stage, stage_name):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        stage.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

class TrainingPipeline:
    def __init__(self, stages: list) -> None:
        self.stages = stages

    def train(self) -> None:
        for (stage_name, stage) in self.stages:
            run_stage(stage, stage_name)

def main():
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline()),
        ("Data Validation stage", DataValidationTrainingPipeline()),
        ("Data Transformation stage", DataTransformationTrainingPipeline()),
        ("Model Trainer stage", ModelTrainerTrainingPipeline())
    ]
    
    training_pipeline = TrainingPipeline(stages)
    training_pipeline.train()
