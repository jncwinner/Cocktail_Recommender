"""
Implementation of a PyTorch-based recommender.
"""

import logging
from dataclasses import dataclass
from typing import NamedTuple, Union
from tqdm.auto import tqdm
import math
import numpy as np
import pandas as pd
from numba import njit

from csr import CSR

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util

# I want a logger for information
_log = logging.getLogger(__name__)


# function to get item tags usable in EmbeddingBag
# it would be hopelessly slow in pure Python, compile w/ Numba
@njit
def _tag_bag_inputs(tag_mat: CSR, items: np.ndarray):
    """
    Get the inputs for a :py:class:`torch.nn.EmbeddingBag` --- the input, offset, and
    value arrays --- from an item-tag matrix.

    Args:
        tag_mat:
            The item-tag matrix.
        items:
            The items whose tags are needed. May contain repeats.
    """
    # how many item-tag pairs do we have?
    # this is equivalent to np.sum(tag_mat.row_nnzs()[items]) (which would be fast in Python),
    # but uses less memory
    n_it = 0
    for i in items:
        sp, ep = tag_mat.row_extent(i)
        rl = ep - sp
        n_it += rl
    
    # allocate result arrays
    offsets = np.empty(len(items), dtype=np.int32)
    tagids = np.empty(n_it, dtype=np.int32)
    pos = 0

    # copy each item's rows to result array
    for ii, i in enumerate(items):
        # get row start/end
        sp, ep = tag_mat.row_extent(i)
        # how many tags for this item?
        itc = ep - sp
        end = pos + itc
        # copy values
        tagids[pos:end] = tag_mat.colinds[sp:ep]
        # set offset
        offsets[ii] = pos
        # update position for storing results
        pos = end
    
    return tagids, offsets


# named tuples are a quick way to make classes that are tuples w/ named fields
class ItemTags(NamedTuple):
    """
    Item tags suitable for input to an EmbeddingBag.  This is used for both
    ingredients and number of ingredients.
    """

    tag_ids: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_items(cls, matrix, items):
        if isinstance(items, torch.Tensor):
            items = items.numpy()
        tids, offs = _tag_bag_inputs(matrix, items)
        return cls(torch.from_numpy(tids), torch.from_numpy(offs))

    def to(self, dev):
        return ItemTags(self.tag_ids.to(dev), self.offsets.to(dev))


class Batch(NamedTuple):
    """
    A single batch of training data.
    """

    users: torch.Tensor
    items: torch.Tensor
    neg_items: torch.Tensor

    def to(self, dev):
        "Convert the batch data to a tensor, if necessary, and copy to the specified device."
        u, i, ni = self

        return Batch(u.to(dev), i.to(dev), ni.to(dev))


@dataclass
class DrinkData:
    """
    Capture data about drinks that is saved after training.
    """
    # user and item indices
    users: pd.Index
    items: pd.Index

    # item-ingredient matrix
    ingredient_mat: CSR
    # item-ingredientCount matrix
    ingredientCount_mat: CSR

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)
    
    @property
    def n_ingredients(self):
        return self.ingredient_mat.ncols if self.ingredient_mat else 0
    
    @property
    def n_ingredientCounts(self):
        return self.ingredientCount_mat.ncols if self.ingredientCount_mat else 0

    def drink_ingredients(self, drinks) -> ItemTags:
        return ItemTags.from_items(self.ingredient_mat, drinks)
    
    def drink_ingredientCounts(self, drinks) -> ItemTags:
        return ItemTags.from_items(self.ingredientCount_mat, drinks)


@dataclass
class DrinkTrainData:
    """
    Class capturing MF training data/context
    """
    # user and item indices
    users: pd.Index
    items: pd.Index

    matrix: CSR

    # consumption data
    r_users: np.ndarray
    r_items: np.ndarray

    batch_size: int

    @property
    def n_samples(self):
        return len(self.r_users)

    @property
    def batch_count(self):
        return math.ceil(self.n_samples / self.batch_size)

    def batch(self, rows):
        # get the ratings for this batch
        ub = self.r_users[rows]
        ib = self.r_items[rows]
        jb, jsc = sampling.neg_sample(self.matrix, ub, sampling.sample_unweighted)

        ub = torch.from_numpy(ub)
        ib = torch.from_numpy(ib)
        jb = torch.from_numpy(jb)

        return Batch(ub, ib, jb)


class DrinkNet(nn.Module):
    """
    Torch module that defines the matrix factorization model.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_ingredients(int): the number of ingredients
        n_ingredientCounts(int): the number of ingredientCounts
        n_feats(int): the embedding dimension
    """

    drink_data: DrinkData
    n_feats: int

    i_full = None

    def __init__(self, data, n_feats, reg, user_bias=True):
        super().__init__()
        self.drink_data = data
        self.n_feats = n_feats

        if isinstance(reg, float):
            self.ub_reg = self.ib_reg = self.p_reg = self.q_reg = reg
        elif len(reg) == 2:
            ureg, ireg = reg
            self.ub_reg = self.p_reg = ureg
            self.ib_reg = self.q_reg = ireg
        elif len(reg) == 4:
            self.ub_reg, self.ib_reg, self.p_reg, self.q_reg = reg
        else:
            raise ValueError('invalid regularization term')
        
        # user and item bias terms
        if user_bias:
            self.u_bias = nn.Embedding(data.n_users, 1)
        else:
            self.u_bias = None
        self.i_bias = nn.Embedding(data.n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(data.n_users, n_feats)
        self.i_embed = nn.Embedding(data.n_items, n_feats)

        # ingredient and ingredientCount embeddings
        if data.n_ingredients:
            self.a_embed = nn.EmbeddingBag(data.n_ingredients, n_feats)
        else:
            self.a_embed = None
        if data.n_ingredientCounts:
            self.g_embed = nn.EmbeddingBag(data.n_ingredientCounts, n_feats)
        else:
            self.g_embed = None

        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        if self.u_bias is not None:
            self.u_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.mul_(0.05)
        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)
        if self.use_ingredients:
            self.a_embed.weight.data.mul_(0.05)
        if self.use_ingredientCounts:
            self.g_embed.weight.data.mul_(0.05)

    @property
    def device(self):
        return self.i_bias.weight.data.device

    @property
    def use_ingredients(self):
        return self.a_embed is not None

    @property
    def use_ingredientCounts(self):
        return self.g_embed is not None

    def forward(self, user, item, negative=None, *, include_reg=False):
        B = len(user)

        ub, uvec = self._user_rep(user)
        ib, ivec = self._item_rep(item)

        score = ib + ub + torch.sum(uvec * ivec, 1)

        if include_reg:
            if isinstance(ub, torch.Tensor):
                ub_w = torch.linalg.norm(ub) * self.ub_reg / B
            else:
                ub_w = 0.0
            ib_w = torch.linalg.norm(ib) * self.ib_reg / B
            ue_w = torch.linalg.norm(uvec) * self.p_reg / B
            ie_w = torch.linalg.norm(ivec) * self.q_reg / B
            reg = ub_w + ib_w + ue_w + ie_w
        
        if negative is not None:
            nib, nivec = self._item_rep(negative)
            neg_score = nib + ub + torch.sum(uvec * nivec, 1)
            score = score - neg_score

            if include_reg:
                reg = reg + torch.linalg.norm(nib) * self.ib_reg / B \
                    + torch.linalg.norm(nivec) * self.q_reg / B
        
        # we're done
        assert score.shape == user.shape
        
        if include_reg:
            return score, reg
        else:
            return score

    def _user_rep(self, user):
        ub = 0.0

        ut = user.to(self.device)

        if self.u_bias is not None:
            ub = self.u_bias(ut).reshape(-1)
        
        uvec = self.u_embed(ut)

        return ub, uvec

    def _item_rep(self, item):
        it = item.to(self.device)
        ib = self.i_bias(it).reshape(-1)

        if self.i_full is not None:
            ivec = self.i_full(it)
        else:
            ivec = self.i_embed(it)
            if self.use_ingredients:
                ingredients = self.drink_data.drink_ingredients(item).to(self.device)
                avec = self.a_embed(ingredients.tag_ids, ingredients.offsets)
                ivec = ivec + avec
            if self.use_ingredientCounts:
                ingredientCounts = self.drink_data.drink_ingredientCounts(item).to(self.device)
                gvec = self.g_embed(ingredientCounts.tag_ids, ingredientCounts.offsets)
                ivec = ivec + gvec

        return ib, ivec

    def compact(self, *, init_only=False):
        """
        Collapse item feature embeddings into integrated item embeddings
        for fast recommendations.
        """
        if init_only:
            self.i_full = nn.Embedding(self.drink_data.n_items, self.n_feats)
            return

        iw = self.i_embed.weight.data
        n, k = iw.shape
        
        if self.a_embed is not None:
            amat = self.drink_data.ingredient_mat
            ainput = torch.from_numpy(amat.colinds).to(self.device)
            aoffset = torch.from_numpy(amat.rowptrs[:-1]).to(self.device)
            aw = self.a_embed(ainput, aoffset)
            assert aw.shape == iw.shape
            iw = iw + aw

        if self.g_embed is not None:
            gmat = self.drink_data.ingredientCount_mat
            ginput = torch.from_numpy(gmat.colinds).to(self.device)
            goffset = torch.from_numpy(gmat.rowptrs[:-1]).to(self.device)
            gw = self.g_embed(ginput, goffset)
            assert gw.shape == iw.shape
            iw = iw + gw

        self.i_full = nn.Embedding(n, k, _weight=iw)


class DrinkTagMF(Predictor):
    """
    Implementation of a tag-aware hybrid MF in PyTorch.
    """

    _device = None
    _train_dev = None

    drink_data_: DrinkData
    _train_data: DrinkTrainData

    def __init__(self, n_features, *, batch_size=4096, epochs=5, reg=0.01, components='all', loss='bpr', device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            batch_size(int):
                The batch size for training.  Since this model is relatively simple,
                large batch sizes work well.
            reg(float):
                The regularization term to apply to embeddings and biases.
            epochs(int):
                The number of training epochs to run.
            rng_spec:
                The random number specification.
        """
        self.n_features = n_features
        self.batch_size = batch_size
        if components == 'all':
            self.components = ['ingredients', 'ingredientCounts']
        elif components == 'none':
            self.components = []
        elif isinstance(components, str):
            self.components = [components]
        else:
            self.components = components
        self.epochs = epochs
        self.reg = reg
        self.rng_spec = rng_spec
        self.loss = loss

        self._device = device

    def fit(self, ratings, *, ingredients, ingredientCounts, **kwargs):
        # run the iterations
        timer = util.Stopwatch()
        
        _log.info('[%s] preparing input data set', timer)
        self._prepare_data(ratings, ingredients, ingredientCounts)

        dev = self._device
        if dev is None:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._prepare_model(dev)

        # now _data has the training data, and __model has the trainable model

        for epoch in range(self.epochs):
            _log.info('[%s] beginning epoch %d of %d', timer, epoch + 1, self.epochs)
        
            self._fit_iter()

            unorm = torch.linalg.norm(self._model.u_embed.weight.data).item()
            inorm = torch.linalg.norm(self._model.i_embed.weight.data).item()
            _log.info('[%s] epoch %d finished (|P|=%.3f, |Q|=%.3f)',
                      timer, epoch + 1, unorm, inorm)

        _log.info('finished training')
        self._finalize()
        self._cleanup()
        return self

    def _prepare_data(self, ratings, ingredients, ingredientCounts):
        "Set up a training data structure for the MF model"
        # index users and items
        _log.info('creating matrix for %d ratings', len(ratings))
        matrix, users, items = sparse_ratings(ratings[['user', 'item']])

        r_users = np.require(matrix.rowinds(), 'i4')
        r_items = np.require(matrix.colinds, 'i4')

        # set up ingredient tags
        if 'ingredients' in self.components:
            _log.info('creating ingredient tag matrix')
            au_idx = pd.Index(np.unique(ingredients['ingredient']))
            ingredients = ingredients[['item', 'ingredient']]
            ingredients = ingredients[ingredients['item'].isin(items)]
            ingredients = ingredients.drop_duplicates()
            ba_ino = items.get_indexer(ingredients['item']).astype('i4')
            ba_ano = au_idx.get_indexer(ingredients['ingredient']).astype('i4')
            ba_mat = CSR.from_coo(ba_ino, ba_ano, None, (len(items), len(au_idx)))
        else:
            ba_mat = None

        # set up ingredient tags
        if 'ingredients' in self.components:
            _log.info('creating ingredient tag matrix')
            ingredients = ingredients[ingredients['item'].isin(items)]
            bgs = ingredients['ingredient'].astype('category')
            bg_ino = items.get_indexer(ingredients['item']).astype('i4')
            assert np.all(bg_ino >= 0)
            ngs = len(bgs.cat.categories)
            ba_mat = CSR.from_coo(bg_ino, bgs.cat.codes.values.astype('i4'), None, (len(items), ngs))
        else:
            ba_mat = None
            
        # set up ingredientCount tags
        if 'ingredientCounts' in self.components:
            _log.info('creating ingredientCount tag matrix')
            ingredientCounts = ingredientCounts[ingredientCounts['item'].isin(items)]
            bgs = ingredientCounts['n_steps'].astype('category')
            bg_ino = items.get_indexer(ingredientCounts['item']).astype('i4')
            assert np.all(bg_ino >= 0)
            ngs = len(bgs.cat.categories)
            bg_mat = CSR.from_coo(bg_ino, bgs.cat.codes.values.astype('i4'), None, (len(items), ngs))
        else:
            bg_mat = None

        _log.info('data ready to go')
        drink_data = DrinkData(users, items, ba_mat, bg_mat)
        train_data = DrinkTrainData(users, items, matrix, r_users, r_items, self.batch_size)

        self.drink_data_ = drink_data
        self._train_data = train_data

    def _prepare_model(self, train_dev=None):
        self._rng = util.rng(self.rng_spec)
        ub = self.loss == 'logistic'
        model = DrinkNet(self.drink_data_, self.n_features, self.reg, user_bias=ub)
        self._model = model
        if train_dev:
            _log.info('preparing to train on %s', train_dev)
            self._train_dev = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # set up training features
            self._opt = Adam(self._model.parameters())

    def _finalize(self):
        "Finalize model training, moving back to the CPU"
        self._model.compact()
        self._model = self._model.to('cpu')
        self._model.eval()
        del self._train_dev

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._train_data
        del self._opt
        del self._rng

    def _fit_iter(self):
        """
        Run one iteration of the recommender training.
        """
        n = self._train_data.n_samples
        # permute the training data
        perm = self._rng.permutation(n)
        loop = tqdm(range(self._train_data.batch_count))

        if self.loss == 'bpr':
            batch = self._fit_batch_bpr
        elif self.loss == 'logistic':
            batch = self._fit_batch_logistic
        else:
            raise ValueError('unknown loss ' + self.loss)

        for i in loop:
            # get the batch - we do this manually, our data is so simple it's faster
            b_start = i * self.batch_size
            b_end = min(b_start + self.batch_size, n)
            # get training rows for this batch
            b_rows = perm[b_start:b_end]

            batch(b_rows)

            # loop.set_postfix_str('loss: {:.3f}'.format(loss))
        
        loop.clear()

    def _fit_batch_logistic(self, rows):
        """
        Fit a single batch of training data with logistic loss.
        """
        # create input tensors from the data
        batch = self._train_data.batch(rows)
        
        # p_as = self.drink_data_.drink_ingredients(batch.items).to(self._train_dev)
        # p_gs = self.drink_data_.drink_ingredientCounts(batch.items).to(self._train_dev)

        # n_as = self.drink_data_.drink_ingredients(batch.neg_items).to(self._train_dev)
        # n_gs = self.drink_data_.drink_ingredientCounts(batch.neg_items).to(self._train_dev)

        # compute scores and loss
        pos_pred, pp_reg = self._model(batch.users, batch.items, include_reg=True)
        neg_pred, np_reg = self._model(batch.users, batch.neg_items, include_reg=True)
        # _log.debug('total scores: pos=%f, neg=%f', torch.sum(pos_pred).item(), torch.sum(neg_pred).item())
        # _log.debug('total reg: pos=%f, neg=%f', torch.sum(pp_reg).item(), torch.sum(np_reg).item())
        pp_loss = -(pos_pred - 2 * torch.log(1 + torch.exp(pos_pred))).sum()
        np_loss = torch.log(1 + torch.exp(neg_pred)).sum()
        pred_loss = pp_loss + np_loss
        pred_loss = pred_loss / (len(rows) * 2)
        # _log.debug('loss: %f (pos: %f, neg: %f)', pred_loss.item(), pp_loss.item(), np_loss.item())
        assert np.isfinite(pred_loss.item())

        # add regularization loss
        reg_loss = pp_reg + np_reg
        loss = pred_loss + reg_loss

        # update model
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        # return loss.item()

    def _fit_batch_bpr(self, rows):
        """
        Fit a single batch of training data with logistic loss.
        """
        # create input tensors from the data
        batch = self._train_data.batch(rows)
        
        # p_as = self.drink_data_.drink_ingredients(batch.items).to(self._train_dev)
        # p_gs = self.drink_data_.drink_ingredientCounts(batch.items).to(self._train_dev)

        # n_as = self.drink_data_.drink_ingredients(batch.neg_items).to(self._train_dev)
        # n_gs = self.drink_data_.drink_ingredientCounts(batch.neg_items).to(self._train_dev)

        # compute scores and loss
        scores, reg = self._model(batch.users, batch.items, batch.neg_items, include_reg=True)
        pair_loss = (-F.logsigmoid(scores)).mean()

        # add regularization loss
        loss = pair_loss + reg

        # update model
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        # return loss.item()
        
    def predict_for_user(self, user, items, ratings=None):
        """
        Generate item scores for a user.

        This needs to do two things:

        1. Look up the user's ratings (because ratings is usually none)
        2. Score the items using them

        Note that user and items are both user and item IDs, not positions.
        """

        # convert user and items into rows and columns
        try:
            u_row = self.drink_data_.users.get_loc(user)
        except KeyError:
            _log.warn('user %s unknown', user)
            return pd.Series(np.nan, index=items)

        i_cols = self.drink_data_.items.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]

        u_tensor = torch.from_numpy(np.repeat(u_row, len(i_cols)))
        i_tensor = torch.from_numpy(i_cols)
        
        # get scores
        with torch.no_grad():
            scores = self._model(u_tensor, i_tensor).to('cpu')
        
        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    def to_device(self, dev):
        _log.info('moving model to %s', dev)
        self._device = dev
        self._model = self._model.to(dev)
    
    def __str__(self):
        return 'HyDrink(features={}, reg={})'.format(self.n_features, self.reg)

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_device' in state:
            del state['_device']

        if '_model' in state:
            del state['_model']
            state['_model_weights_'] = self._model.state_dict()
        
        if '_train_data' in state:
            _log.warn('attempted to serialize training data')
            del state['_train_data']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_drink_data' in state:  # ugly hack for cross-version compatibility
            self.drink_data_ = self._drink_data
            del self._drink_data
        if '_model_weights_' in state:
            self._prepare_model()
            self._model.compact(init_only=True)
            self._model.load_state_dict(self._model_weights_)
            self._model.eval()
            del self._model_weights_


# if __name__ == "__main__":
#     train3 = pd.read_parquet('my-gr-interact-train3.parquet', engine='pyarrow')
#     test_set = pd.read_parquet('my-gr-interact-test-set.parquet', engine='pyarrow')
    # import torchmf_mod
    # from torchmf_mod import TorchMF2

#     model = DrinkTagMF(50)
#     model.fit(train3)
#     from lenskit import batch

#     batch.predict(model, test_set)