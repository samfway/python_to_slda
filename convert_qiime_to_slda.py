#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
from ml_utils.parse import load_dataset 
from ml_utils.util import convert_labels_to_int
from ml_utils.cross_validation import get_test_sets, get_test_train_set, load_test_sets_from_file
from ml_utils.slda import write_matrix_to_slda_file

""" 
    This code can be used to format python matrices/labels into
    the input format required by the C implementation of the SLDA 
    algorithm, the one found on David Blei's website.  

    Using the --validation flag, a set of matrices training/testing matrices is
    created.  Otherwise, by default, the program converts the matrix and all 
    labels into a format suitable for the C program. 
"""

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
    args.add_argument('-o', '--output-prefix', help='Prefix for output files (default: ./out_)', \
                            default='./out_', type=str)
    args.add_argument('-t', '--test-sets-file', help='File of training/test indices', default=None)

    args.add_argument('--dm', action='store_true', help='Input ' + \
        'matrix is a distance matrix', default=False)

    args.add_argument('--validation', action='store_true', help='Create ' + \
        '10foldCV dataset', default=False)

    args = args.parse_args()
    return args

def create_slda_cv_dataset(data_matrix, labels, output_prefix, test_sets=None):
    """ Create cross validation sets to evaluation """ 

    unique_labels, labels = convert_labels_to_int(labels)

    if test_sets is None:
        test_sets = get_test_sets(labels, kfold=10, stratified=True)

    for idx, test_set in enumerate(test_sets):
        train_matrix, train_labels, test_matrix, test_labels = \
            get_test_train_set(data_matrix, labels, test_set)
        # Create training set & test set
        output_name = '%s_%d_%s_' % (output_prefix, idx, 'train')
        create_slda_dataset(train_matrix, train_labels, output_name)
        output_name = '%s_%d_%s_' % (output_prefix, idx, 'test')
        create_slda_dataset(test_matrix, test_labels, output_name)

    # Labels were converted to integers, save a legend 
    output = open(output_name + '_label_info.txt')
    for idx, label in enumerate(unique_labels):
        output.write('%d: %s\n' % (idx, label))
    output.close() 

def create_slda_dataset(data_matrix, labels, output_prefix, sample_ids=None):
    """ Reformat matrix+labels for the SLDA C implementation.  
        Creates a labels file, a data file, and an (optional) sample_ids file.
    
        Data file is of the format:
            <M> <term_1>:<count> <term_2>:<count> ... <term_N>:<count>
        NOTE: M is the number of unique, non-zero elements in the row
              In other words, the number of <term>:<count> pairs to follow.
    """ 
    # 1)  Create labels file: one label per line 
    output_name = output_prefix + 'labels.txt'
    output = open(output_name, 'w')
    labels_file = output_name
    if labels is not None:
        unique_labels, label_indices = convert_labels_to_int(labels)
        output.write('\n'.join([str(l) for l in label_indices]))
    else:
        fake_labels = [0]*len(data_matrix)
        output.write('\n'.join([str(l) for l in fake_labels]))
    output.close()
    
    # 2) Create data file for SLDA Format: 
    output_name = output_prefix + 'data.txt' 
    write_matrix_to_slda_file(data_matrix, output_name)
    data_file = output_name

    # 3) Create sample id file: one per line (NOT NEEDED BY SLDA)
    if sample_ids is not None:
        output_name = output_prefix + 'sample_ids.txt'
        output = open(output_name, 'w')
        output.write('\n'.join(sample_ids))
        output.close() 
    
    return data_file, labels_file


if __name__=="__main__":
    args = interface()
    data_matrix, sample_ids, labels = load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file, args.dm)

    if not args.validation:
        create_slda_dataset(data_matrix, labels, args.output_prefix, sample_ids)
    else:
        if args.test_sets_file is not None:
            test_sets = load_test_sets_from_file(args.test_sets_file)
        create_slda_cv_dataset(data_matrix, labels, args.output_prefix, test_sets)

