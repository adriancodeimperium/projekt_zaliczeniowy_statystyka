import csv
import os
import neat
from typing import List, Dict

from neat.reporting import BaseReporter

import pdb

import visualize


class MyReporter(BaseReporter):
    def start_generation(self, generation):
        print("Starting generation {}".format(generation))

    def post_evaluate(self, config, population, species, best_genome):
        print("Current best fitness: {}".format(best_genome.fitness))


class Record:
    def __init__(self, flower_dimensions: List[float], species_name: str):
        self.flower_dimensions: List[float] = flower_dimensions
        self.species_vector: List[float] = self.__species_name_to_vector(species_name)

    @staticmethod
    def __species_name_to_vector(name: str) -> List[float]:
        return [float(name == "setosa"), float(name == "versicolor"), float(name == "virginica")]


class Dataset:
    def __init__(self, iris_file_path: str):
        self.path: str = iris_file_path
        self.records: Dict[int, Record] = {}
        self.__load_data()

    def __load_data(self):
        with open(self.path, mode="r") as f:
            reader = csv.reader(f)
            _ = next(reader)
            self.records = {
                i: Record(
                    flower_dimensions=[float(v) for v in line[:-1]],
                    species_name=line[-1]
                )
                for i, line in enumerate(reader)
            }

    def get_records_for_evaluation(self):
        return [(r.flower_dimensions, r.species_vector) for r in self.records.values()]


class NeatNNPresentation:
    def __init__(self, dataset: Dataset, neat_config: neat.Config, max_generations: int = 100):
        self.dataset: Dataset = dataset
        self.neat_config: neat.Config = neat_config
        self.max_generations: int = max_generations

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 450
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for input_data, expected_output in self.dataset.get_records_for_evaluation():
                genome_output = net.activate(input_data)
                genome.fitness -= (sum(abs(expected_output[k] - genome_output[k]) for k in range(len(expected_output))))

    def run(self):
        population = neat.Population(self.neat_config)
        population.add_reporter(MyReporter())
        best_genome = population.run(self.evaluate_genomes, self.max_generations)
        winner_net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_config)
        visualize.draw_net(self.neat_config, best_genome, True)
        for input_data, expected_output in self.dataset.get_records_for_evaluation():
            net_output = winner_net.activate(input_data)
            print("input {!r}, expected output {!r}, got {!r}".format(input_data, expected_output, net_output))


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)

    cfg = neat.Config(
        genome_type=neat.DefaultGenome,
        reproduction_type=neat.DefaultReproduction,
        species_set_type=neat.DefaultSpeciesSet,
        stagnation_type=neat.DefaultStagnation,
        filename=os.path.join(this_dir, "config", "neat.ini")
    )

    nnp = NeatNNPresentation(
        dataset=Dataset(iris_file_path=os.path.join(this_dir, "data", "iris.csv")),
        neat_config=cfg,
        max_generations=3000
    )

    nnp.run()
