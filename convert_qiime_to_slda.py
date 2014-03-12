#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
from ml_utils.parse import load_dataset 
from ml_utils.util import convert_labels_to_int

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
    unique_labels, label_indices = convert_labels_to_int(labels)
    output_name = output_prefix + 'labels.txt'
    output = open(output_name, 'w')
    output.write('\n'.join([str(l) for l in label_indices]))
    output.close()

    # 2) Create sample id file: one per line (NOT NEEDED BY SLDA)
    output_name = output_prefix + 'sample_ids.txt'
    output = open(output_name, 'w')
    output.write('\n'.join(sample_ids))
    output.close() 
    
    # 3) Create data file for SLDA Format: 
    #    <M> <term_1>:<count> <term_2>:<count> ... <term_N>:<count>
    # Where M is the number of unique, non-zero elements in the matrix
    output_name = output_prefix + 'data.txt' 
    output = open(output_name, 'w')
    N = data_matrix.shape[1] 
    for row in data_matrix:
        to_write = [ '%d:%d'%(k, row[k]) for k in xrange(N) if row[k] > 0 ] 
        output.write('%d %s\n' % (len(to_write), to_write))
    output.close()

if __name__=="__main__":
    args = interface()
    data_matrix, sample_ids, labels = load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file, args.dm)
    create_slda_dataset(data_matrix, sample_ids, labels, args.output_prefix)

