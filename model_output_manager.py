import os
import numpy as np
import pandas as pd
import h5py
import pickle as pkl
# import glob
from pathlib import Path

# %% Hyperparameters
DATA_FILE_NAME = 'model_data'
RUN_NAME = 'run'

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
    Add row to run tracker table using entries of param_dict.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        table_path (string): The filepath for the table (including that table name, i.e. 'output/param_table.csv').
            Windows style paths are okay.
        column_labels (list): Contains the keys of params_table in the order in which they should be written in the
            run tracker table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_number.

    Returns:
        run_id (int): Unique identifier for the run.

    """
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
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_number.

    Returns:
        run_id (int): The unique identifier for the run
        run_dir (str): The path to the output directory for the run

    """
    run_id = update_output_table(table_params, table_path, compare_exclude, [], overwrite_existing)
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    run_dir = Path(table_dir/(RUN_NAME+'_'+str(run_id)+'/'))
    Path.mkdir(run_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def write_output(output, params, table_params, output_dir, overwrite=False, data_filetype='pickle'):
    """

    Args:
        output (dict): Dictionary that holds the output data
        params (dict): Dictionary that holds the parameters
        table_params (dict): Dictionary that holds the parameters in the output table
        output_dir (string): Parent directory for output file. The output file name is DATA_FILE_NAME.pkl or
            DATA_FILE_NAME.h5

    Returns:

    """
    output_dir = Path(output_dir)
    output_dir = output_dir.parents[0]
    output_file = (output_dir/DATA_FILE_NAME).with_suffix('.pkl')
    print()
    print("Attempting to write data to "+str(Path.cwd() / output_file ))
    print()
    try:
        output_dir.mkdir(parents=True)
    except (OSError, FileExistsError):
        if overwrite:
            print("Warning: existing data directory overwritten.")
        else:
            print("Data directory already exists. Not writing output.")
            return
    if data_filetype == 'hdf5':
        with h5py.File(output_dir, "w") as fid:
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
        with open(output_file, "wb") as fid:
            pkl.dump(data, fid, protocol=4)

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
    run_id = update_output_table(table_params, table_path, compare_exclude, columns, overwrite_existing)
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    if data_filetype == 'hdf5':
        file_name = DATA_FILE_NAME+'.h5'
    elif data_filetype == 'pickle':
        file_name = DATA_FILE_NAME+'.pkl'
    else:
        raise ValueError('data_filetype option not recognized.')
    output_dir = table_dir / (RUN_NAME+'_'+str(run_id))
    output_path = output_dir / file_name
    params.update(dict(table_path=table_path, output_dir=output_dir, run_id=run_id))
    write_output(model_output, params, table_params, output_path, overwrite_existing, data_filetype)

    return run_id, output_path

# Todo: Build in support for nested dictionaries / groups
def hdf5group_to_dictionary(h5grp):
    d = {}
    for key in h5grp:
        d[key] = h5grp[key].value
    return d

# %% Methods for checking for run existence and getting location

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
    table_dir = Path(table_dir)
    run_dir = Path(table_dir/(RUN_NAME+'_'+str(run_id)))
    if only_directory_ok:
        return Path.exists(run_dir)
    else:
        filename_no_ext = Path(run_dir/DATA_FILE_NAME)
        filelist = list(filename_no_ext.glob('.*'))
        return len(filelist) > 0

def get_dirs_and_ids_for_run(run_params, table_path='output/param_table.csv', compare_exclude=[]):
    """

    Parameters
    ----------
    run_params : dict
        Dictionary holding the parameters specifying the run
    table_path : str
        Path to the run tracker table
    compare_exclude : list
        list holding the parameters that should be excluded in specifying the run

    Returns
    -------
    List[str]
        Directories that match run_params and compare_exclude.
    List[int]
        Run Ids that match run_params and compare_exclude.
    List[bool]
        List of bools that correspond with the other two returned lists, with an entry being True if the output data
        file is in the directory and False otherwise.
    """
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    out = _get_updated_table(compare_exclude, run_params, table_path)
    run_ids = out[0]
    merge_ids = out[-1]
    dirs = [str(table_dir / f"run_{x}") for x in merge_ids]
    ids = [x for x in merge_ids]
    output_exists = [Path.exists((Path(d)/DATA_FILE_NAME).with_suffix('.pkl')) for d in dirs]
    return dirs, ids, output_exists

# def run_with_params_exists(table_params, table_path='output/param_table.csv', compare_exclude=[],
#                            check_output_exist=True):
#     """
#     Given a set of parameters, check if a run matching this set exists.
#
#     Args:
#         table_params (dict, OrderedDict): Parameters that will be put into the table
#         table_path (string): The filepath for the table.
#         compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
#             run. For instance, if runs are identical except for the date when the run was done, then it might be
#             reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
#             one may want to put the date parameter key in compare_exclude.
#
#     Returns:
#
#     """
#
#
#     table_path = Path(table_path)
#     table_dir = table_path.parents[0]
#     out = _get_updated_table(compare_exclude, table_params, table_path)
#     merge_ids = out[-1]
#
#     dirs = get_dirs_for_run(run_params, table_path='output/param_table.csv', compare_exclude=[])
#
#     file_check = True
#     if check_output_exist:
#         for dir in dirs:
#             if not Path.exists(table_dir)
#     return len(merge_ids) > 1

# %% Methods for loading data
def _get_updated_table(compare_exclude, table_params, table_path, column_labels=None):
    """
    Core method for updating a parameter table.

    Parameters
    ----------
    compare_exclude : list
        Parameters that should be excluded for comparisons with the run table
    table_params : Dict-like
        Parameters for the run
    table_path : str
        Path to the run table
        column_labels (List[str]): Labels for the columns. Used to assert an order.

    Returns
    -------
    run_id : int
        Unique identifier for the run
    run_number : int
        Indexes runs with the same parameters.
    param_df_updated : DataFrame
        The updated run table.
    merge_ids : List[int]
        List of unique identifiers of runs that corresponded with table_params (not incluing the new row)
    """
    table_path = Path(table_path)
    if not table_path.exists():  # If the table hasn't been created yet.
        run_id = 0
        if not column_labels:
            param_df_updated = pd.DataFrame(table_params, index=[run_id], dtype=object)
        else:
            param_df_updated = pd.DataFrame(table_params, index=[run_id], columns=column_labels, dtype=object)

        param_df_updated['run_number'] = 0
        param_df_updated = param_df_updated.fillna('na')
        run_number = 0
        merge_ids = [0]
        return run_id, run_number, param_df_updated, merge_ids

    if not column_labels:
        column_labels = list(table_params.keys()).copy()
    if 'run_number' not in column_labels:
        column_labels.append('run_number')  # To make sure run_number is the last column, unless otherwise specified
    param_df = pd.read_csv(table_path, index_col=0, dtype=str)
    new_cols = __unique_to_set(param_df.columns, column_labels)[1]  # param_keys that don't yet belong to param_df
    for key in new_cols:
        param_df[key] = pd.Series('na', index=param_df.index)
    unique_to_param_df = __unique_to_set(param_df.columns, column_labels)[0]
    if not unique_to_param_df:  # If column_labels is comprehensive
        param_df = param_df[column_labels]  # Reorder columns of param_df based on column_labels

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

    # Debug code
    # agree = temp1.reset_index().merge(temp2, how).set_index('index')
    # for key in temp1:
    #     print()
    #     print(key, '\t\t', temp1[key], temp2[key])
    #     print()

    # This is needed to ensure proper order in some cases (if table_params has less items than the table has columns)
    column_labels = list(temp_merge.columns)
    column_labels.append('run_number')
    run_number = temp_merge.shape[0]
    new_row['run_number'] = run_number
    new_row = new_row[column_labels]

    param_df_updated = param_df.append(new_row, sort=True)
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

    # md = io.loadmat(basedir + 'output/' + str(run_id) + '/collected_data.mat')
    # md = pkl.load(open(basedir + output_dir + '/' + run_name + '_' + str(run_id) + '/output.pkl', 'rb'))
    # params = io.loadmat(basedir + 'output/' + str(run_id) + '/PARAMS.mat')
    table_path = Path(table_path)
    table_dir = table_path.parents[0]
    filename_no_ext = Path(table_dir/(RUN_NAME+'_'+str(run_id)+'/'+DATA_FILE_NAME))
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
    table_path = Path(table_path)
    table_dir = table_path.parents[0]

    run_id, run_number, param_df_updated, merge_ids = _get_updated_table(compare_exclude, table_params,
                                                                         table_path=table_path)

    if len(merge_ids) == 1:
        run_id = merge_ids[0]
        output, params = load_from_id(run_id, table_path, data_filetype)
        if output == -1:
            return -1, None, None, None
        if data_filetype == 'hdf5' and ret_as_dict:
            output = hdf5group_to_dictionary(output)
            params = hdf5group_to_dictionary(params)
        run_dir = Path(table_dir/(RUN_NAME+'_'+str(run_id)))
        return output, params, run_id, run_dir
    elif len(merge_ids) > 1:
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
    elif len(merge_ids) == 0:
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
