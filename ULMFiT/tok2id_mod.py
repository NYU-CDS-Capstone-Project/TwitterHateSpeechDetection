from fastai.text import *
import fire
import pickle
from os.path import join


def tok2id(dir_path, max_vocab=30000, min_freq=1):
    print(f'dir_path {dir_path} max_vocab {max_vocab} min_freq {min_freq}')
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    tmp_path = p / 'tmp'
    assert tmp_path.exists(), f'Error: {tmp_path} does not exist.'

    trn_tok = np.load(tmp_path / 'tok_trn.npy')
    val_tok = np.load(tmp_path / 'tok_val.npy')

    print (tmp_path)
    itos = pickle.load(open(join(tmp_path, 'itos.pkl'), 'rb'))
    print (join(tmp_path, 'itos.pkl'))
    print ('itos loaded')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))

    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

    np.save(tmp_path / 'trn_ids.npy', trn_lm)
    np.save(tmp_path / 'val_ids.npy', val_lm)

if __name__ == '__main__': fire.Fire(tok2id)
