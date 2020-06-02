import pickle
import neat
from app import app
from flask import render_template
import os


def get_genomes_path():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, "..", "neatnn", "genomes")


@app.route('/')
@app.route('/index')
def index():
    genomes_path = get_genomes_path()
    genomes = [f for f in os.listdir(genomes_path)
               if os.path.isfile(os.path.join(genomes_path, f)) and not f.startswith(".")]
    return render_template('index.html', genomes=genomes)


@app.route('/genome/<genome_filename>')
def genome(genome_filename):
    with open(os.path.join(get_genomes_path(), genome_filename), mode="rb") as f:
        genome_data = pickle.load(f)
        nn = neat.nn.FeedForwardNetwork.create(genome_data['genome'], genome_data['config'])
        results = []
        for input_data, expected_output in genome_data['dataset'].get_records_for_evaluation():
            genome_output = nn.activate(input_data)
            results.append({
                'input': input_data,
                'expected': expected_output,
                'actual': genome_output,
                'is_correct': sum([abs(expected_output[i] - genome_output[i]) for i in range(len(expected_output))]) / len(expected_output) <= 0.25
            })
        return render_template(
            'genome.html',
            results=results,
            total=len(results),
            correct=len([x for x in results if x['is_correct']]),
            genome=genome_data['genome'],
            genome_name=genome_filename
        )