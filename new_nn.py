import argparse
import os
import pickle
import time

import neat

from neatnn.iris import NeatNNPresentation, Dataset


def make_nnp(max_generations: int, population_size: int) -> NeatNNPresentation:
    this_dir = os.path.dirname(__file__)

    cfg = neat.Config(
        genome_type=neat.DefaultGenome,
        reproduction_type=neat.DefaultReproduction,
        species_set_type=neat.DefaultSpeciesSet,
        stagnation_type=neat.DefaultStagnation,
        filename=os.path.join(this_dir, "config", "neat.ini")
    )

    return NeatNNPresentation(
        dataset=Dataset(iris_file_path=os.path.join(this_dir, "data", "iris.csv")),
        neat_config=cfg,
        max_generations=max_generations,
        population_size=population_size
    )


def default_name(max_generations: int, population_size: int) -> str:
    return "g_{}_p_{}_{}".format(max_generations, population_size, int(time.time()))


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--max-generations", required=True, type=int,
                        help="Maximum number of generations to process")
    parser.add_argument("-p", "--population-size", required=True, type=int,
                        help="Population size")
    parser.add_argument("-n", "--name", required=False, type=str,
                        help="Custom name")

    args = vars(parser.parse_args())
    name = "{}_{}".format(args['name'], int(time.time())) if args['name'] is not None else default_name(
        args['max_generations'], args['population_size'])

    nnp = make_nnp(args['max_generations'], args['population_size'])

    print("Finding a best neural network genome using {} generations of {} specimen with the NEAT algorithm...")
    genome = nnp.run()
    print("Found genome with fitness {}".format(genome.fitness))
    print("Saving genome to genomes/{}".format(name))
    with open(os.path.join(this_dir, "neatnn", "genomes", name), mode="wb") as f:
        pickle.dump({
            'genome': genome,
            'config': nnp.neat_config,
            'dataset': nnp.dataset
        }, f)
