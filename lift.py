"""
Implementation of a lift-based recommender.  This is written for
readability and reasonable performance; there are probably faster
ways to implement it.

It illustrates two things: the basic principles of writing a
LensKit recommender, and tricks for processing large data sets
efficiently and within memory bounds.  This can process the 
MovieLens 25M data set in less than 16GB of RAM.

It can be your starting point for your genome-based recommender.
You should not need as many tricks as I have performed here for
your recommender - computing the lift matrix in reasonable memory
is difficult, but if you are strategic in your choice of data
structures, you do shouldn't need anything so large as an item
co-occurrence matrix.
"""

import logging
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings
from lenskit.util import Stopwatch

# I want a logger for information
_log = logging.getLogger(__name__)


# LensKit algorithms generally extend the Predictor base class.
# This is true even when they do not generate rating predictions:
# in SciKit-Learn terminology, we use 'predict' to generate scores
# for data points, regardless of what those scores are.  An actual
# rating prediction is just a specific kind of prediction.
class Lift(Predictor):
    """
    Implementation of Lift.
    """

    def fit(self, ratings):
        """
        Train the recommender.  This method can also take additional
        keyword parameters if it requires data besides the ratings.

        The result of calling this method is that the model now has
        learned everything it needs from the input data to generate
        predictions (which will later be used for recommendations),
        and stored the results of that in instance fields that end
        with undescores (e.g. user_ratings_).

        Review the A1 solution for some of the ideas here.
        """
        timer = Stopwatch()

        # let's only consider items with at least 2 ratings. if an
        # item only has 1 rating, it will have very high lift with
        # the other items the user has rated, but on the basis of a
        # single vote, and also be useless as a seed item for other
        # users' recommendations.
        i_counts = ratings['item'].value_counts()
        keep = i_counts.index.values[i_counts > 1]
        _log.info('[%s] keeping %d of %d items', timer, len(keep), len(i_counts))
        ratings = ratings[ratings['item'].isin(keep)]

        # now let's make a sparse matrix.
        # LensKit provides a utility function for that:
        matrix, users, items = sparse_ratings(ratings, scipy=True)
        _log.info('[%s] training from %d ratings (%d users, %d items)',
                  timer, matrix.nnz, len(users), len(items))
        # make an array of popularities that matches the matrix columns
        pops = i_counts.reindex(items).values

        # our matrix should be the right size - users by items:
        assert matrix.shape == (len(users), len(items))

        # now, the matrix is filled with the rating *values*. we're
        # doing lift, so we want replace that with a bunch of 1s.
        # I'll make them float16s to save space - they're just 1s.
        matrix.data = np.ones_like(matrix.data, dtype=np.float16)

        # in order to compute lift, we need co-occurrances, individual
        # item occurrance counts, and the total number of items.
        n_items = len(items)
        _log.info('[%s] counting co-occurrances', timer)
        # now, computing the co-occurrance matrix takes a _lot_ of memory.
        # so we're going to compute it in blocks.  each block is one piece
        # of the final lift matrix, and we will reassemble in the end
        # to do that efficiently, we want *rows* of a transposed matrix.
        # so we'll make an efficiently transposable matrix:
        left = matrix.T.tocsr()

        lift_block_rows = []
        lift_block_cols = []
        lift_block_data = []
        tot_nnz = 0
        for start in tqdm(range(0, n_items, 1000)):
            end = min(start + 1000, n_items)
            _log.debug('computing block %d:%d', start, end)
            subleft = left[start:end, :]
            # actually do the co-occurrence multiplication
            lbk = subleft @ matrix
            tot_nnz += lbk.nnz

            # now, we can do some vectorized things. first, the derivation of
            # the arithmetic:
            #     lift = P(i|j)/P(i) = P(i,j) / (P(i) P(j))
            #     P(i,j) = count(i, j) / n
            #     P(i) = count(i) / n
            #     P(j) = count(j) / n
            #     lift = count(i, j) * n / (count(i) * count(j))
            # to do this, we're going to convert to a COO matrix, and then extract
            # the actual inner arrays. 
            # we're then going to get the upper triangle, to cut intermediate memory
            # use in half.
            # and finally. lift less than 1 is a *negative* correlation, effectively.
            # let's get rid of those. we'll mask, filter our row/col/value
            # arrays, and add them to their respective lists for reassembly
            lbk = lbk.tocoo()
            row, col, data = lbk.row, lbk.col, lbk.data
            del subleft, lbk
            
            # rows are offset by the start of the matrix
            row += start

            # compute our filter
            mask = (col > row) & (data > 1)
            row = row[mask]
            col = col[mask]
            data = data[mask]

            # now multiply all entries by the number of items to get the numerator
            data *= n_items
            # now, we can modify its data by dividing by each denominator
            # component in turn.
            data /= pops[row]  # count(i)
            data /= pops[col]  # count(j)

            lift_block_rows.append(row)
            lift_block_cols.append(col)
            lift_block_data.append(data.astype('f4'))

        # we don't need that transposed matrix anymore
        del left

        # now we have lists of arrays that will hold our final matrix
        # it is time to assemble it, one array at a time to control memory use
        # since we only saved half the matrices before, we need to double up
        # the results when reassembling them so we have a full symmetric matrix
        # this takes quite a bit of memory, but makes the end computations more
        # efficient.
        _log.info('[%s] assembling matrix with %d entries', timer, tot_nnz)
        lift_rows = np.concatenate(lift_block_rows + lift_block_cols, dtype='i4')
        lift_cols = np.concatenate(lift_block_cols + lift_block_rows, dtype='i4')
        del lift_block_rows, lift_block_cols
        lift_data = np.concatenate(lift_block_data + lift_block_data, dtype='f4')
        del lift_block_data

        lift = csr_matrix((lift_data, (lift_rows, lift_cols)))
        
        # now we can save the results, and we're done
        _log.info('[%s] training finished, saving %d results', timer, lift.nnz)
        self.user_index_ = users
        self.item_index_ = items
        self.user_ratings_ = matrix
        self.lift_ = lift

    def predict_for_user(self, user, items, ratings=None):
        """
        Generate item scores for a user.

        This needs to do two things:

        1. Look up the user's ratings (because ratings is usually none)
        2. Score the items using them

        Note that user and items are both user and item IDs, not positions.
        """

        # convert user and items into rows and columns
        u_row = self.user_index_.get_loc(user)
        i_cols = self.item_index_.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]

        # now get the user's ratings
        u_rates = self.user_ratings_[u_row, :]
        
        # we want to score based on the mean lift between it and items the
        # user has rated. u_rates is a 1×n sparse matrix of user items, where
        # a column is 1 if the user rated it, and 0 otherwise.
        # If we multiply this by an n×n matrix of lifts, we will get the *total*
        # lift for each item.  We can then divide by the sum of the user's values 
        # eq. to the # of 1s!) to get the mean lift! And then pick the lifts
        # we actually care about (it's faster here to compute all lifts than a
        # subset of them)
        scores = u_rates @ self.lift_
        scores = scores.toarray().ravel()

        # now we divide
        scores = scores / np.sum(u_rates)

        # and we can finally put in a series to return
        results = pd.Series(scores[i_cols], index=scorable)
        return results.reindex(items)  # fill in missing values with nan

# This class can be used just like any other class!
# You can wrap it in Recommender.adapt to generate top-N recommendations