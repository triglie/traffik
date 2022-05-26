import click

from traffik.dataset import DatasetGenerator

@click.command()
@click.option('--source',   default='./data',       help='Path of the directory that contains videos and CSVs')
@click.option('--save',     default='./save',       help='Path where to store the hdf5 dataset file')
@click.option('--out',      default='traffikds.h5', help='Name of the generated HDF5 dataset file.')
def generate(source, save, out):
    dg = DatasetGenerator(sourcepath=source, savepath=save, h5filename=out)
    dg.generate()


if __name__ == '__main__': generate()
