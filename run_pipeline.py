from pipelines.training_pipeline import training_pipelines

if __name__ == "__main__":
    # run the pipeline
    training_pipelines(train_data_path="data\\train.csv")