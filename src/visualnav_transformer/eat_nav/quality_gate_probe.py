"""Small deterministic nested trajectory probes for Quality information."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class DeterministicProbe(nn.Module):
    """Both modes use this exact graph; Probe 0 supplies q=0.5."""
    def __init__(self):
        super().__init__()
        self.dem_cnn = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(64, 64))
        self.goal_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
        self.quality_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        self.position = nn.Parameter(torch.zeros(1, 6, 64))
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=0.0, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 16))

    def forward(self, dem, goal, quality):
        if dem.shape[1:] != (4, 2, 20, 20) or goal.shape != (len(dem), 2) or quality.shape != (len(dem), 1):
            raise ValueError('probe input shape mismatch')
        frames = self.dem_cnn(dem.reshape(-1, 2, 20, 20)).reshape(len(dem), 4, 64)
        tokens = torch.cat((frames, self.goal_mlp(goal).unsqueeze(1),
                            self.quality_mlp(quality).unsqueeze(1)), dim=1) + self.position
        return self.head(self.transformer(tokens).mean(dim=1)).reshape(len(dem), 8, 2)


def shuffle_q_within_terrain(q, terrain, seed: int):
    q = np.asarray(q, dtype=np.float32)
    terrain = np.asarray(terrain)
    shuffled = q.copy()
    rng = np.random.default_rng(seed)
    for value in np.unique(terrain):
        indices = np.flatnonzero(terrain == value)
        shuffled[indices] = rng.permutation(q[indices])
    return shuffled


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError('trajectory/mask shape mismatch')
    valid = mask.unsqueeze(-1)
    return ((prediction - target).square() * valid).sum() / (2 * mask.sum().clamp_min(1))


def aggregate_masked_mse(prediction, target, mask) -> float:
    return float(masked_mse(prediction, target, mask).detach().cpu())


def tensorize_windows(windows, stats, device):
    """Keep every valid-window tensor on one device for reproducible paired runs."""
    rows = [w for w in windows if np.asarray(w['action_mask']).any()]
    dem = torch.from_numpy(np.stack([w['dem'] for w in rows])).to(device)
    goal = torch.from_numpy(np.stack([w['goal'] for w in rows])).to(device)
    q = torch.tensor([w['quality'] for w in rows], dtype=torch.float32, device=device).reshape(-1,1)
    action = torch.from_numpy(np.stack([(w['action']-stats.mean)/stats.std for w in rows])).to(device)
    mask = torch.from_numpy(np.stack([w['action_mask'] for w in rows])).to(device)
    return {'dem': dem, 'goal': goal, 'q': q, 'action': action, 'mask': mask,
            'terrain': np.asarray([w['terrain_type'] for w in rows]),
            'episode': [w['episode_key'] for w in rows],
            'index': np.asarray([w['window_index'] for w in rows], dtype=np.int32)}


def _validation_loss(model, data, q, batch_size):
    model.eval()
    sum_error = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(q), batch_size):
            sl = slice(start, start + batch_size)
            prediction = model(data['dem'][sl], data['goal'][sl], q[sl])
            mask = data['mask'][sl]
            sum_error += float((((prediction-data['action'][sl]).square())*mask.unsqueeze(-1)).sum())
            count += int(mask.sum()) * 2
    return sum_error / max(count,1)


def train_probe(data, validation, *, kind, training_seed, shuffle_seed, settings, output):
    """Train one scratch deterministic probe; aggregate validation chooses checkpoint."""
    from pathlib import Path
    import json
    torch.manual_seed(training_seed)
    model = DeterministicProbe().to(data['dem'].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.probe_learning_rate)
    if kind == 'probe_0':
        train_q = torch.full_like(data['q'], settings.fixed_probe_q)
        val_q = torch.full_like(validation['q'], settings.fixed_probe_q)
    elif kind == 'probe_q':
        train_q, val_q = data['q'], validation['q']
    elif kind == 'probe_q_shuffled':
        if shuffle_seed is None:raise ValueError('shuffle seed required')
        shuffled = shuffle_q_within_terrain(data['q'].cpu().numpy().flatten(), data['terrain'], shuffle_seed)
        train_q = torch.from_numpy(shuffled).to(data['q'].device).reshape(-1,1)
        val_q = validation['q']
    else:
        raise ValueError('unknown probe kind')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf');best_epoch=0;bad=0
    for epoch in range(1,settings.max_probe_epochs+1):
        generator=torch.Generator(device=data['dem'].device).manual_seed(training_seed*100000+epoch)
        order=torch.randperm(len(train_q),generator=generator,device=data['dem'].device)
        model.train()
        losses=[]
        for indices in order.split(settings.probe_batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(data['dem'][indices],data['goal'][indices],train_q[indices])
            loss=masked_mse(prediction,data['action'][indices],data['mask'][indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val=_validation_loss(model,validation,val_q,settings.probe_batch_size)
        improved=val < best-settings.probe_min_delta
        if improved:
            best=val;best_epoch=epoch;bad=0
            torch.save({'model':model.state_dict(),'kind':kind,'training_seed':training_seed,
                        'shuffle_seed':shuffle_seed,'validation_loss':val,'epoch':epoch},output/'best.pt')
        else:
            bad+=1
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':val,'best':improved}
        history.append(row)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if epoch==1 or epoch%10==0 or improved:
            print('PROBE_EPOCH',kind,training_seed,shuffle_seed,epoch,round(val,6),flush=True)
        if bad>=settings.probe_patience:break
    return {'kind':kind,'training_seed':training_seed,'shuffle_seed':shuffle_seed,
            'best_validation_loss':best,'best_epoch':best_epoch,'epochs_ran':len(history),
            'validation_improving_at_cap':len(history)==settings.max_probe_epochs and best_epoch>=len(history)-2,
            'checkpoint':str(output/'best.pt')}


def predict_probe(checkpoint, data, q_kind, batch_size, fixed_q=0.5):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=DeterministicProbe().to(data['dem'].device).eval()
    model.load_state_dict(payload['model'])
    q=torch.full_like(data['q'],fixed_q) if q_kind=='probe_0' else data['q']
    with torch.no_grad():
        return torch.cat([model(data['dem'][i:i+batch_size],data['goal'][i:i+batch_size],q[i:i+batch_size])
                          for i in range(0,len(q),batch_size)],dim=0)
