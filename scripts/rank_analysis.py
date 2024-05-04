import marimo

__generated_with = "0.4.11"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import re
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors


    def make_extractor():
        # GEQP3 call: (m, n) = (711, 5376), numerical rank = 571
        match_m = '(?P<m>[0-9]+)'
        match_n = '(?P<n>[0-9]+)'
        match_r = '(?P<r>[0-9]+)'
        specstr = r'\(' + match_m + ', ' +  match_n + r'\), numerical rank = ' + match_r
        pattern = re.compile(specstr)

        def extractor(s):
            match = pattern.search(s)
            if match is None:
                return None
            else:
                m = int(match.group('m'))
                n = int(match.group('n'))
                r = int(match.group('r'))
                return m, n, r

        return extractor
    return make_extractor, mcolors, np, plt, re


@app.cell
def __(make_extractor, np, re):
    fn = '/Users/rjmurr/Documents/randnla/spand-repo/tests/logs/bump2911/lvl20_skip10_tol1e-2_rank_info_log.txt'

    tol = float(re.search(r'_tol(?P<t>[0-9]+[e][-][0-9]+)', fn).group('t'))
    with open(fn, 'r') as f:
        extractor = make_extractor()
        qp3data = []
        level = 0
        for ell in f.readlines():
            if 'Level ' in ell:
                match = re.search(r'Level (?P<lev>[0-9]+), [0-9]+ dofs left, [0-9]+ clusters left', ell)
                if match is not None:
                    level = int(match.group('lev'))
            if 'GEQP3 call' in ell:
                qp3data.append((level,) + extractor(ell))
            else:
                pass

    qp3 = np.array(qp3data)
    qp3 = qp3[~np.any(qp3[:,1:3] == 0, axis=1),:]
    min_mns = np.min(qp3[:,1:3], axis=1)
    frac_numerical_ranks = qp3[:,3] / min_mns
    return (
        ell,
        extractor,
        f,
        fn,
        frac_numerical_ranks,
        level,
        match,
        min_mns,
        qp3,
        qp3data,
        tol,
    )


@app.cell
def __():
    #  'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'
    return


@app.cell
def __(frac_numerical_ranks, plt, tol):
    plt.hist(frac_numerical_ranks, bins='auto')
    plt.title(f'(Numerical ranks using reltol = {tol}) / min(m, n)')
    plt.show()
    return


@app.cell
def __(min_mns, qp3):
    fullrank_selector = min_mns == qp3[:,3]
    qp3f = qp3[fullrank_selector,:]
    qp3l = qp3[~fullrank_selector,:]
    return fullrank_selector, qp3f, qp3l


@app.cell
def __(mcolors, plt, qp3f, tol):
    _matrix_cols_hist_xaxis = qp3f[:,2] 
    _matrix_rows_hist_yaxis = qp3f[:,1]
    _color_gamma = 0.5
    _ax = plt.hist2d(
        _matrix_cols_hist_xaxis,
        _matrix_rows_hist_yaxis,
        range=[[1000,8000],[0,500]],
        bins=(40,40),
        norm=mcolors.PowerNorm(_color_gamma)
    )
    _line1 = 'Heatmap of column count (x-axis) and row count (y-axis)'
    _line2 = f'\nof matrices that are numerically full-rank, up to reltol = {tol}.'
    plt.title(_line1 + _line2)
    plt.show()
    return


@app.cell
def __(np, plt, qp3l, tol):
    aspectsl = qp3l[:,2] / qp3l[:,1]
    _ax = plt.figure().add_subplot()
    histout = _ax.hist(aspectsl, bins='auto', range=(1,200))
    _num_not_wide = np.count_nonzero(aspectsl < 1)
    _pct_not_wide = 100 * _num_not_wide / aspectsl.size
    _pct_txt = '%.2f' % _pct_not_wide
    _line1 = '(Num cols) / (num rows) for matrices that are'
    _line2 = f'\nnumerically rank-deficient, up to reltol = {tol}'
    _ax.text(50,300,f' Excludes the {_pct_txt}% of matrices \nthat had (num rows) > (num cols)')
    plt.title(_line1 + _line2)
    return aspectsl, histout


@app.cell
def __(mcolors, min_mns, plt, qp3, tol):
    def dimension_heatmap_for_rank_threshold(_rank_frac):
        _selector = qp3[:,3]/min_mns < _rank_frac
        _matrix_cols_hist_xaxis = qp3[_selector,2] 
        _matrix_rows_hist_yaxis = qp3[_selector,1]
        _color_gamma = 0.5
        _fig = plt.figure()
        _ax = _fig.add_subplot()
        _ax.hist2d(
            _matrix_cols_hist_xaxis,
            _matrix_rows_hist_yaxis,
            range=[[1000,8000],[0,1000]],
            bins=(40,40),
            norm=mcolors.PowerNorm(_color_gamma)
        )
        _fracstr = '%.2f' % _rank_frac
        _tolstr  = '%.2f' % tol
        _line1 = 'Heatmap of column count and row count of matrices whose'
        _line2 = f'\nnumerical rank (QP3, cutoff reltol {_tolstr}) is < {_fracstr} min(m,n).'
        _ax.set_title(_line1 + _line2)
        _ax.set_xlabel('column count (n)')
        _ax.set_ylabel('row count (m)')
        return _ax, _fig
    return dimension_heatmap_for_rank_threshold,


@app.cell
def __(dimension_heatmap_for_rank_threshold, plt):
    _ax, _fig = dimension_heatmap_for_rank_threshold(0.05)
    plt.show()
    return


@app.cell
def __(dimension_heatmap_for_rank_threshold, np, tol):
    _tolstr  = '%.2f' % tol
    _foldername = '/Users/rjmurr/Documents/randnla/spand-repo/scripts/plots'
    for _rankfrac in np.arange(start=0.05, stop=1.01, step=0.01):
        _ax, _fig = dimension_heatmap_for_rank_threshold(_rankfrac)
        _rf = '%.2f' % _rankfrac
        _fig.savefig(f'{_foldername}/dimheatmaps_for_rankfrac_{_rf}_tol_{_tolstr}.png')
        del _fig
    return


@app.cell
def __(imageio, np):
    images = []
    _foldername = '/Users/rjmurr/Documents/randnla/spand-repo/scripts/plots'
    for _rankfrac in np.arange(start=0.05, stop=1.01, step=0.01):
        _rf = '%.2f' % _rankfrac
        _filename = f'{_foldername}/dimheatmaps_for_rankfrac_{_rf}_tol{_tolstr}.png'
        images.append(imageio.imread(_filename))
    imageio.mimsave(f'{_foldername}/dimheatmaps_for_rankfracs_tol_{_tolstr}.gif', images)
    return images,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
