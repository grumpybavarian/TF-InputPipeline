from mnist_feeddict import ModelFeeddict
from mnist_pipeline import ModelPipeline
import time


def main():
    # measure performance of feeddict implementation
    print("Evaluating feeddict...")
    feeddict_start = time.time()
    feeddict = ModelFeeddict(num_epochs=10)
    feeddict.train()
    feeddict_duration = time.time() - feeddict_start

    # measure performance of dataset implementation
    print("Evaluating pipeline...")
    pipeline_start = time.time()
    pipeline = ModelPipeline(num_epochs=10)
    pipeline.train()
    pipeline_duration = time.time() - pipeline_start

    print("Feeddict: ", feeddict_duration)
    print("Pipeline: ", pipeline_duration)


if __name__ == '__main__':
    main()
