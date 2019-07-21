import os
import numpy as np
import pandas as pd
import h5py
import pickle as pkl
# import glob
from pathlib import Path

_DISABLE_MOM = False

# %% Hyperparameters
data_file_name = 'model_data'
run_name = 'run'

# data_filetype = 'pkl'

# %% Helper functions
def __unique_to_set(a, b):
    """
    Return elements that are unique to container a and items that are unique to container b among the union of a and b.

    Args:
        a (container):
        b (container):

    Returns:
        a_unique (list): elements that are contained in container a but not container b
        b_unique (list): elements that are contained in container b but not container a

    """

    def overlap(a, b):
        return list(set(a) & set(b))

    def difference(a, b):
        return list(set(a) ^ set(b))

    dif = difference(a, b)
    a_unique = overlap(a, dif)
    b_unique = overlap(b, dif)
    return a_unique, b_unique

# %% Methods for outputing data

def update_output_table(table_params, table_path='output/param_table.csv', compare_exclude=[], column_labels=None,
                        overwrite_existing=True):
    """
    Add row to output table using entries of param_dict.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        table_path (string): The filepath for the table (including that table name, i.e. 'output/param_table.csv').
            Windows style paths are okay.
        column_labels (list): Contains the keys of params_table in the order in which they should be written in the
            output table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_number.

    Returns:
        run_id (int): Unique identifier for the run.

    """
    if _DISABLE_MOM:
        return -1

    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    Path.mkdir(table_dir, exist_ok=True)

    for key in table_params:
        table_params[key] = str(table_params[key])

    run_id, run_number, param_df_updated, merge_indices = _get_updated_table(compare_exclude,
                                                                             table_params,
                                                                             table_path,
                                                                             column_labels)

    if run_number == 0 or not overwrite_existing:
        param_df_updated.to_csv(table_path)
    else:
        run_id = np.max(merge_indices)  # id for most recent run that matches table_params
    return run_id

def make_dir_for_run(table_params, table_path='output/param_table.csv', compare_exclude=[],
                     overwrite_existing=True):
    """
    Creates a directory for the run as well as the corresponding row in the parameter table.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        table_path (string): The filepath for the table.
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        column_labels (list): Contains the keys of params_table in the order in which they should be written in the
            output table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_number.

    Returns:
        run_id (int): The unique identifier for the run
        run_dir (str): The path to the output directory for the run

    """
    if _DISABLE_MOM:
        return -1, Path.parent[0]

    run_id = update_output_table(table_params, table_path, compare_exclude, [], overwrite_existing)

    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    run_dir = Path(table_dir / (run_name + '_' + str(run_id) + '/'))
    Path.mkdir(run_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def write_output(output, params, table_params, output_path, overwrite=False, data_filetype='pickle'):
    """

    Args:
        output (dict): Dictionary that holds the output data
        params (dict): Dictionary that holds the parameters
        table_params (dict): Dictionary that holds the parameters in the output table
        output_path (string): Filepath for output file

    Returns:

    """
    if _DISABLE_MOM:
        return

    output_path = Path(output_path)
    output_dir = output_path.parents[0]
    print()
    print("Attempting to write data to " + str(Path.cwd() / output_path))
    print()
    do_write_output = True
    try:
        os.makedirs(output_dir, exist_ok=False)
    except (OSError, FileExistsError):
        if overwrite:
            print("Warning: existing data directory overwritten.")
        else:
            print("Data directory already exists. Not writing output.")
            do_write_output = False
    if do_write_output:
        if data_filetype == 'hdf5':
            with h5py.File(output_path, "w") as fid:
                param_grp = fid.create_group("parameters")
                param_table_grp = fid.create_group("table_parameters")
                out_grp = fid.create_group("output")
                for key in params:
                    if params[key] is not None:
                        param_grp.create_dataset(key, data=params[key])
                for key in table_params:
                    if table_params[key] is not None:
                        param_table_grp.create_dataset(key, data=table_params[key])
                for key in output:
                    if output[key] is not None:
                        out_grp.create_dataset(key, data=output[key])
        elif data_filetype == 'pickle':
            data = dict(parameters=params, table_parameters=table_params, output=output)
            with open(output_path, "wb") as fid:
                pkl.dump(data, fid)

        print("Done. Data written.")

def save_model(table_params, table_path, model_output, params, compare_exclude=[], columns=None,
               overwrite_existing=False, data_filetype='pickle'):
    """
    Creates an entry in the output_table and saves the output in the corresponding directory. Basically just a
    wrapper to call update_output_table and then write_output.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        table_path (string): The filepath for the table.
        model_output (dict): Dictionary that holds the output data
        params (dict): Dictionary that holds the parameters
        columns (list): Contains the keys of params_table in the order in which they should be written in the
            output table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_number.
        output_path (string): Filepath for output file
        data_filetype (str): Filetype for data to be written in. Currently only hdf5 is supported.

    Returns:

    """
    if _DISABLE_MOM:
        return -1, 'MOM_DISABLED/'

    run_id = update_output_table(table_params, table_path, compare_exclude, columns, overwrite_existing)
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    if data_filetype == 'hdf5':
        file_name = data_file_name + '.h5'
    elif data_filetype == 'pickle':
        file_name = data_file_name + '.pkl'
    else:
        raise ValueError('data_filetype option not recognized.')
    output_dir = table_dir / (run_name + '_' + str(run_id))
    output_path = output_dir / file_name
    params.update(dict(table_path=table_path, output_dir=output_dir, run_id=run_id))
    write_output(model_output, params, table_params, output_path, overwrite_existing, data_filetype)

    return run_id, output_path

# %% Methods for loading data
# Todo: Build in support for nested dictionaries / groups
def hdf5group_to_dictionary(h5grp):
    d = {}
    for key in h5grp:
        d[key] = h5grp[key].value
    return d

def run_with_id_exists(run_id, table_dir='output', only_directory_ok=False):
    """
    Given the name of the run, the ID of the run, and the directory of the output table, checks to see if the run
    exists.

    Args:
        run_id ():
        table_dir ():
        only_directory_ok (bool): If True, this method returns True if the output directory exists, even if the output
            files haven't been written to the output directory. If False, this method only returns True if the
            corresponding output directory and output files exist.

    Returns:
        bool

    """
    if _DISABLE_MOM:
        return False
    table_dir = Path(table_dir)
    run_dir = Path(table_dir / (run_name + '_' + str(run_id)))
    if only_directory_ok:
        return Path.exists(run_dir)
    else:
        filename_no_ext = Path(run_dir / data_file_name)
        filelist = list(filename_no_ext.glob('.*'))
        return len(filelist) > 0

def get_dirs_for_run(table_params, table_path='output/param_table.csv', compare_exclude=[]):
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    out = _get_updated_table(compare_exclude, table_params, table_path)
    merge_ids = out[-1]
    return [str(table_dir / f"run_{x}") for x in merge_ids]

def run_with_params_exists(table_params, table_path='output/param_table.csv', compare_exclude=[],
                           only_directory_ok=False):
    """
    Given a set of parameters, check if a run matching this set exists.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        table_path (string): The filepath for the table.
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.

    Returns:

    """
    if _DISABLE_MOM:
        return False

    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    out = _get_updated_table(compare_exclude, table_params, table_path)
    merge_ids = out[-1]

    if only_directory_ok:
        return len(merge_ids) > 1
    else:
        run_id, run_number, param_df_updated, merge_ids

def _get_updated_table(compare_exclude, table_params, table_path, column_labels=None):
    """
    Core method for updating a parameter table.

    Args:
        compare_exclude (List): Parameters that should be excluded for comparisons with the run table
        table_params (Dict-like): Parameters for the run
        table_path (str): Path to the run table
        column_labels (List[str]): Labels for the columns. Used to assert an order.

    Returns:
        run_id (int): Unique identifier for the run
        run_number (int): Indexes runs with the same parameters.
        param_df_updated (DataFrame): The updated run table.
        merge_ids (List[int]): List of unique identifiers of runs that correspond with table_params.
    """

    if not os.path.isfile(table_path):  # If the table hasn't been created yet.
        run_id = 0
        param_df_updated = pd.DataFrame(table_params, index=[run_id], columns=column_labels, dtype=object)
        param_df_updated['run_number'] = 0
        param_df_updated = param_df_updated.fillna('na')
        run_number = 0
        merge_ids = [0]
        return run_id, run_number, param_df_updated, merge_ids

    if column_labels is None:
        column_labels = list(table_params.keys()).copy()
    if 'run_number' not in column_labels:
        column_labels.append('run_number')  # To make sure run_number is the last column, unless otherwise specified
    param_df = pd.read_csv(table_path, index_col=0, dtype=str)
    new_cols = __unique_to_set(param_df.columns, column_labels)[1]  # param_keys that don't yet belong to param_df
    for key in new_cols:
        param_df[key] = pd.Series('na', index=param_df.index)
    unique_to_param_df = __unique_to_set(param_df.columns, column_labels)[0]
    if not unique_to_param_df:  # If column_labels is comprehensive
        param_df = param_df[column_labels]  # Reorder colums of param_df based on column_labels

    run_id = np.max(np.array(param_df.index)) + 1
    new_row = pd.DataFrame(table_params, index=[run_id], dtype=str)
    for e1 in unique_to_param_df:  # Add placeholders to new row for items that weren't in param_dict
        new_row[e1] = 'na'
    # new_row = new_row[column_labels]
    compare_exclude2 = compare_exclude.copy()
    compare_exclude2.append('run_number')
    temp1 = param_df.drop(compare_exclude2, axis=1, errors='ignore')
    temp2 = new_row.drop(compare_exclude, axis=1, errors='ignore')
    # temp_merge = pd.merge(temp1, temp2)
    temp_merge = temp1.reset_index().merge(temp2).set_index('index')  # This merges while preserving the index

    # This is needed to ensure proper order in some cases (if table_params has less items than the table has columns)
    column_labels = list(temp_merge.columns)
    column_labels.append('run_number')
    run_number = temp_merge.shape[0]
    new_row['run_number'] = run_number
    new_row = new_row[column_labels]

    param_df_updated = param_df.append(new_row)
    merge_ids = list(temp_merge.index)

    return run_id, run_number, param_df_updated, merge_ids

def load_from_id(run_id, table_path='output/param_table.csv', data_filetype='pickle'):
    # Todo: get it working with more filetypes
    """
    Given the name of the run, the ID of the run, and the directory of the output table, load the data.

    Args:
        run_id ():
        table_path (): Name for the directory of the table. Cannot be inside another directory other than the current
            working one.

    Returns:

    """
    if _DISABLE_MOM:
        raise Exception('shouldnt get here')

    # md = io.loadmat(basedir + 'output/' + str(run_id) + '/collected_data.mat')
    # md = pkl.load(open(basedir + output_dir + '/' + run_name + '_' + str(run_id) + '/output.pkl', 'rb'))
    # params = io.loadmat(basedir + 'output/' + str(run_id) + '/PARAMS.mat')
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    filename_no_ext = Path(table_dir / (run_name + '_' + str(run_id) + '/' + data_file_name))
    if data_filetype == "hdf5":
        try:
            hf = h5py.File(filename_no_ext.with_suffix('.h5'), 'r')
        except OSError:
            hf = h5py.File(filename_no_ext.with_suffix('.hdf5'), 'r')
    elif data_filetype == "pickle":
        try:
            with open(filename_no_ext.with_suffix('.pkl'), 'rb') as fid:
                hf = pkl.load(fid)
        except FileNotFoundError:
            return -1, None
    else:
        print("Error: data_filetype option not recognized.")

    output = hf['output']
    params = hf['parameters']
    return output, params

def load_data(compare_exclude, table_params, table_path='output/param_table.csv', ret_as_dict=True,
              data_filetype='pickle'):
    """

    Args:
        table_params (dict): Dictionary of parameters of interest (doesn't need to be comprehensive, but
        should uniquely determine the run).

    Returns:
        output: output data that has been collected
        params: parameters for the run
        run_id:
            TODO: put more info here
        nonunique_params (dict): Dictionary of parameters that have non-unique values.

    Exceptions:
        If the parameters in param_dict don't uniquely determine the run, then an error message will be
        output to say this. The function will then return nonunique_params.

    """
    if _DISABLE_MOM:
        raise Exception('shouldnt get here')

    table_path = Path(table_path)
    table_dir = table_path.parents[0]

    run_id, run_number, param_df_updated, merge_ids = _get_updated_table(compare_exclude, table_params,
                                                                         table_path=table_path)

    if len(matched_ids) == 1:
        run_id = matched_ids[0]
        output, params = load_from_id(run_id, table_path, data_filetype)
        if output == -1:
            return -1, None, None, None
        if data_filetype == 'hdf5' and ret_as_dict:
            output = hdf5group_to_dictionary(output)
            params = hdf5group_to_dictionary(params)
        run_dir = Path(table_dir / (run_name + '_' + str(run_id)))
        return output, params, run_id, run_dir
    elif len(matched_ids) > 1:
        print("The parameters in table_params don't uniquely determine the run.")
        # nonunique_params = {}
        # for cind in param_df_updated[merge_ids[:-1]].columns:
        #     c = param_df_updated[cind].values
        #     cd = set(c)
        #     if len(cd) > 1:
        #         nonunique_params[cind] = cd
        # str1 = "The parameters in param_dict don't uniquely determine the run."
        # str2 = "Here are the nonunique parameters: {}".format(nonunique_params)
        # raise KeyError(str1 + str2)
    elif len(matched_ids) == 0:
        raise KeyError("Error: run matching parameters {} not found".format(table_params))

# %% Untested

# def _get_updated_table(output_dir='output'):
#     basedir = get_base_dir()
#     out_dir = basedir + '/' + output_dir
#     run_ids = [name[-1] for name in os.listdir(out_dir) if os.path.isdir(out_dir + '/' + name)]
#     run_names = [name[:-2] for name in os.listdir(out_dir) if os.path.isdir(out_dir + '/' + name)]
#     table_dir = './' + out_dir + '/param_table.csv'
#     param_df = pd.DataFrame()
#     for it, run_id in enumerate(run_ids):
#         name = run_names[it]
#         with h5py.File(out_dir + '/' + name + '_' + run_id + '/data.hdf5', 'r') as hf:  # Todo: resolve hdf5 vs h5 ext
#             tbl_params = hdf5group_to_dictionary(hf['table_parameters'])
#         new_row = pd.DataFrame(tbl_params, index=[int(run_id)])
#         if it == 0:
#             param_df = new_row.copy()
#             param_df['run_number'] = 0
#         elif it > 0:
#             temp1 = param_df.drop(['ic_seed', 'run_number', 'run_time'], axis=1)
#             temp2 = new_row.drop(['ic_seed', 'run_time'], axis=1)
#             run_number = pd.merge(temp1, temp2).shape[0]
#             new_row['run_number'] = run_number
#             param_df = param_df.append(new_row)
#
#     param_df.to_csv(table_dir)
#
# def delete_from_id(run_name, run_id, table_path='output/param_table.csv'):
#     import shutil
#
#     # table_dir = output_dir + '/' + 'param_table.csv'
#     table_dir = table_path.split('/')[0]
#     data_dir = table_dir + '/' + run_name + '_' + str(run_id)
#
#     shutil.rmtree(data_dir)
#
#     full_df = pd.read_csv(table_dir, index_col=0)
#     full_df = full_df.drop(run_id)
#     full_df.to_csv(table_dir)
