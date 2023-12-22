import numpy as np
import re, glob
import numpy as np
import math


def is_valid_template(path):
    '''* for all'''
    subs = path.split('*')
    return all(len(sub) > 0 for sub in subs)


def get_param_value(path, name):
    '''name_L{*}'''
    pattern = re.compile(rf'[\w/]+{name}[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    try: 
        matched, = pattern.findall(path)
    except ValueError:
        return None

    num = matched[matched.index(name)+len(name):]
    return float(num)


def get_data(path, col_name):
    '''Extract column'''
    with open(path) as lines:
        header = next(lines).strip().split()

        col = header.index(col_name)
        data = []
        for line in lines:
            try:
                data.append(float(line.strip().split()[col]))
            except ValueError:
                continue

        return data

    
def contains(iterable, num, tol=1E-14):
    '''Is num in iterable?'''
    for i, item in enumerate(iterable):
        if abs(item-num) < tol:
            return i
    return None


def extract_data(template, variable_ranges, xcol, ycol):
    '''Look for data in tables and align'''
    assert len(variable_ranges) == template.count('*')

    paths = dict()
    variables = tuple(variable_ranges.keys())
    for path in glob.glob(template):
        if any(w in path for w in ('tikz', 'pvd', 'vtu', 'tikz_txt', 'lstsq_tikz_txt')): continue

        key = ()
        is_valid = True
        for variable in variables:
            value = get_param_value(path, variable)
            if value is None:
                break
            variable_range = variable_ranges[variable]
            print(variable, value, variable_range)
            is_valid = not variable_range or contains(variable_range, value) is not None
            key = key + (value, )
            if not is_valid:
                break
        # Only valid can asssign
        if is_valid:
            paths[key] = path

    print(paths)
    # Want to align data for union of all indep
    if xcol is not None:
        X = set()
        for key in paths:
            path = paths[key]
            X.update(get_data(path, xcol))
    else:
        X = 0
        for key in paths:
            print(key)
            path = paths[key]
            X = max(X, len(get_data(path, ycol)))
        X = np.arange(X)
    X = np.array(sorted(X))
    
    aligned_y = dict()
    # Now we insert NaNs as indicator of missing data
    for key in paths:
        path = paths[key]
        if xcol is not None:
            x = get_data(path, xcol)
        else:
            x = np.arange(len(get_data(path, ycol)))
        # Make room for larger
        Y = np.nan*np.ones_like(X)
        y = get_data(path, ycol)
        for xi, yi in zip(x, y):
            # Look for where to insert
            idx = contains(X, xi)
            if idx is not None:
                Y[idx] = yi
        # Final data
        aligned_y[key] = Y

    return X, aligned_y, variables

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import glob, os, itertools, argparse
    # We want to verify the dependence of Poincare constant on domain parameters
    # NOTE: in the eigenvalue problem we obtain the estimate of constant**2 so 
    # that is why we take the square root below

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # How to get data
    parser.add_argument('-y_column', type=str, default=['cond'], help='', nargs='+')        
    parser.add_argument('-domain', type=str, default='honeycomb', help='')
    parser.add_argument('-rescale', type=int, default=1, choices=(0, 1))
    
    args, _ = parser.parse_known_args()

    path = f'results/{args.domain}/kappa*_spectra.txt'
    
    template = path
    Ns = [1E8, 1E4, 1E2, 1E0, 1E-2, 1E-4, 1E-8]
    value_ranges = {'kappa': Ns}
    sort_by = 'kappa'

    nrefs = [0, 1, 2, 3]
    template = f'results/{args.domain}/rescaleFalse_nrefs*_spectra.txt'
    value_ranges = {'nrefs': nrefs}
    sort_by = 'nrefs'

    xcol = 'N'

    big_table = []
    big_table_headers = ()
    for y_column in args.y_column:
        X, aligned_y, variables = extract_data(template=template,
                                               variable_ranges=value_ranges,
                                               xcol=xcol,
                                               ycol=y_column)

        # FIXME: - check converged
        #        - we loop in order to do fit and then have a result result
        #          to fit again (overkill?)
        #        - how good is the fit?
        #        - reduced should be tikzable
        #        - plot final constants

        # Self inspection
        prev = None
        for key in sorted(aligned_y):
            print(key, '->', (aligned_y[key])[np.isfinite(aligned_y[key])])


        data = np.column_stack([X] + [aligned_y[key] for key in aligned_y])
        header = ' '.join(['x'] + [f'{variables[0]}{key[0]:.4E}' for key in aligned_y])

        tikz_path, ext = os.path.splitext(template.replace('*', 'varied'))
        tikz_path = '_'.join([tikz_path, 'paramRobust',
                              f'y{y_column.upper()}',
                              'tikz'])
        tikz_path = f'{tikz_path}.txt'

        tikz_path = os.path.join(tikz_path)
        with open(tikz_path, 'w') as out:
            out.write('%s\n' % header)
            np.savetxt(out, data)
        print(tikz_path)

        import tabulate
        headers = sorted(aligned_y)

        print(tabulate.tabulate(data, headers=(xcol, ) + tuple(headers)))

        keys = sorted(tuple(aligned_y.keys()), reverse=False)
        table = np.vstack([key for key in keys])

        ultima = np.array([aligned_y[key][np.where(~np.isnan(aligned_y[key]))[0][-1]] for key in keys])
        disp_ultima = np.where(np.abs(ultima) > 50, np.round(ultima, 5), np.round(ultima, 5))

        if not big_table:
            big_table.append(table.flatten())
            big_table_headers = big_table_headers + variables

        big_table.append(disp_ultima)
        big_table_headers = big_table_headers + (y_column, )
        
        if all(len(aligned_y[key]) > 3 for key in keys):
            penultima = np.array([aligned_y[key][np.where(~np.isnan(aligned_y[key]))[0][-2]] for key in keys])
            rel = np.abs(ultima-penultima)/ultima

            table = np.c_[table, disp_ultima, np.abs(ultima-penultima), rel]

            headers = variables + ('data', 'diff', 'rel')
            print()
            print(tabulate.tabulate(table, headers=headers))
            print()
        else:
            table = np.c_[table, disp_ultima]

            headers = variables + ('data', )
            print()
            print(tabulate.tabulate(table, headers=headers))
            print()            

    big_table = np.array(big_table).T
    print()
    print(tabulate.tabulate(big_table, headers=big_table_headers, tablefmt='latex'))
    print()
