# this script is made by Valentin Flunkert
import numpy as np
import pylab as pl

from pydelay import dde23

eqns = { 
    'x1' : '(x1 - pow(x1,3)/3.0 - y1 + C*(x2(t-tau) - x1))/eps',
    'y1' : 'x1 + a',
    'x2' : '(x2 - pow(x2,3)/3.0 - y2 + C*(x1(t-tau) - x2))/eps',
    'y2' : 'x2 + a'
}

params = { 
    'a'   : 1.3,
    'eps' : 0.01,
    'C'   : 0.5,
    'tau' : 3.0
}


class HistoryCache:
    def __init__(self, dde, maxinterval, resolution):
        tfinal = dde.sol['t'][-1]
        self.vals = dde.sample(tfinal - maxinterval, tfinal, resolution)

    def initializeHistory(self, newdde):
        newdde.hist_from_arrays(self.vals)



def simulateInPieces(eqns, params, histfuncs, tfinal, num_pieces, sample_dt): 
    """
    Simulate until `tfinal` in `num_pieces` steps to avoid memory
    overflow. Return the final solution sampled with a resolution of
    `sample_dt`.
    """

    piece_length = tfinal * 1.0 / num_pieces
    n = 0

    # simulatie first piece with the given `histfuncs`
    dde = dde23(eqns=eqns, params=params)
    dde.set_sim_params(tfinal=piece_length)
    dde.hist_from_funcs(histfuncs)
    dde.run()

    maxinterval = 4
    resolution = 1E-4

    # set the history array for the next runs
    hist = HistoryCache(dde, maxinterval, resolution)

    # Sample the final result.
    # You may choose to write something to disc isntead
    result = dde.sample(0, piece_length, sample_dt)

    n += 1
    while n < num_pieces:
        dde = dde23(eqns=eqns, params=params)
        dde.set_sim_params(tfinal=piece_length)
        hist.initializeHistory(dde)
        dde.run()

        # extend the previous result with the new piece
        tmp = dde.sample(0, piece_length, sample_dt)
        for key, value in tmp.iteritems():
            # By definition the simulation always starts at t=0.
            # So we have to shift the time array for the new piece 
            if key == 't':  
                result[key] = np.append(result[key], value + n * piece_length)
            else:
                result[key] = np.append(result[key], value)
        n += 1
        hist = HistoryCache(dde, maxinterval, resolution)
    return result


histfuncs = { 'x1': lambda t: 1.0 }
result = simulateInPieces(eqns, params, histfuncs, 1000, 20, 0.01)

pl.plot(result['t'], result['x1'], 'x-')

# Comparison with single run until t = 1000
dde = dde23(eqns=eqns, params=params)
dde.set_sim_params(tfinal=1000)
dde.hist_from_funcs(histfuncs)
dde.run()
sol = dde.sample(0, 1000, 0.01)

pl.plot(sol['t'], sol['x1'], 'x--')

pl.show()




