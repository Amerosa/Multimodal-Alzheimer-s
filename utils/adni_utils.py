from utils.basic_utils import get_project_root
import csv

def parse_adni_subject(pth):
    """Takes a standard adni nii.gz file name, splits it up and returns the
    subject id and session id information
    """
    return pth.split('_')[:2]


def generate_mappings():
    """
    Parses the labels file and makesa dictionary mapping the subject and session
    to a numerical encoding of the 4 classes
    """
    encoding = {'CN': 0, 'nMCI': 1, 'cMCI': 2, 'AD': 3}
    mappings = {}
    labels_path = get_project_root() / 'data/labels.tsv'
    with open(labels_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)
        for line in tsvreader:
            subject_id, session_id, _, _, label = line
            mappings['_'.join( [subject_id, session_id] )] = encoding[label]
    return mappings
