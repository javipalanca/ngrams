import sys

import click

from cat_dist import initialize_data, write_distance_commands, write_similarity_matrix, compute_embedding


@click.group()
def cli():
    pass


@cli.command(help="Computes the distance matrix for the corpus.")
@click.option('-k', '--size', default=2, help='Context size.')
@click.option('-c', '--corpus', default="brown", help='Corpus name.')
def corpus_distance(size, corpus):
    log_quotient_counts, middle_words = initialize_data(k=size, corpus_name=corpus)
    write_distance_commands(log_quotient_counts, middle_words, k=size)


@cli.command(help="Computes the similarity matrix for the corpus.")
@click.option('-k', '--size', default=2, help='Context size.')
def similarity(size):
    _, middle_words = initialize_data(k=size)
    write_similarity_matrix(middle_words)


@cli.command(help="Computes the embedding.")
def embedding():
    compute_embedding()


if __name__ == '__main__':
    sys.exit(cli())  # pragma: no cover
