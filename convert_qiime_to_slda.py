#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
from ml_utils.parse import load_dataset 

def interface():
    args = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description='Simple wrapper for machine_learning.py.',
        epilog='Supply the following:\n' + \
                  '* OTU table (.biom)\n\n' + \
                  'And either of these two options:\n' + \
                  '1) Mapping file + Metadata category + (optional) Metadata value\n' + \
                  '2) Separate labels file')
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value') 
    args.add_argument('--dm', action='store_true', help='Input ' + \
        'matrix is a distance matrix', default=False)
    args.add_argument('-o', '--output-prefix', help='Prefix for output files (default: ./out_)', \
                            default='./out_', type=str)
    args = args.parse_args()
    return args

def create_slda_dataset(data_matrix, sample_ids, labels, output_prefix):
    """ Reformat matrix+labels for the SLDA C implementation.  
        An extra file is created to track sample_ids but is not needed by SLDA
    """ 
    
    # 1)  Create labels file: one label per line 
    unique_labels = list(set(labels))
    print unique_labels
    exit()

    print len(sample_ids)
    print data_matrix.shape
    print labels

if __name__=="__main__":
    args = interface()
    data_matrix, sample_ids, labels = load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file, args.dm)
    create_slda_dataset(data_matrix, sample_ids, labels, args.output_prefix)

