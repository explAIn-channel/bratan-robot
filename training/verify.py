"""Check the portable training kit without starting a training run or accessing hardware."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/sandbox/rosnav_shadow_v1'))
import train_rl
import numpy as np
import onnxruntime as ort
import torch
from policy_artifacts import current_policy_contract, contract_digest, promotion_quality_gate, PROMOTION_SCENARIOS
from rl_env import FastFlat, DriveEnv
from symmetrize import SymmetricActor


def main():
    from training_lock import exclusive_guard, HandoffError
    with tempfile.TemporaryDirectory() as temporary:
        lock = Path(temporary)/'run'
        with exclusive_guard(lock):
            try:
                with exclusive_guard(lock):
                    raise AssertionError('duplicate writer accepted')
            except HandoffError:
                pass
        with exclusive_guard(lock):
            pass
    evidence = json.loads((ROOT/'export-evidence.json').read_text())
    contract = current_policy_contract()
    assert contract_digest(contract) == evidence['contract_digest']
    model = ROOT.parent/'brain/assets/navigation/actor.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == evidence['files']['actor.onnx']
    checkpoint = ROOT/'original-run/best.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == evidence['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    actor = train_rl.Actor(state['feature_mean'].numpy(), state['feature_std'].numpy())
    actor.load_state_dict(state)
    policy = SymmetricActor(actor).eval()
    vectors = np.load(ROOT/'parity.npz')
    with torch.no_grad():
        expected = policy(torch.from_numpy(vectors['features'])).numpy()
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    result = session.run(None, {'features': vectors['features']})[0]
    assert np.allclose(result, expected, atol=1e-6, rtol=0)
    assert np.allclose(result, vectors['expected'], atol=1e-6, rtol=0)
    flat = FastFlat(ROOT/'map/map.yaml')
    for scenario in train_rl.SCENARIOS:
        env = DriveEnv(flat, np.random.default_rng(260906), scenario=scenario, encoding='log')
        observation = env.reset()
        assert observation.shape == (228,) and np.isfinite(observation).all()
        for _ in range(20):
            observation, reward, done, info = env.step(np.array([0.1, 0.0]))
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if done:
                observation = env.reset()
    loss = policy(torch.from_numpy(vectors['features'][:4])).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    # The final training report uses its declared three-scenario suite.
    metrics = {name: {'learned': {'episodes': 120, 'arrived': 1.0, 'collision': 0.0, 'timeout': 0.0},
                      'expert': {'episodes': 120, 'arrived': 1.0, 'collision': 0.0, 'timeout': 0.0}}
               for name in train_rl.SCENARIOS}
    verdict = promotion_quality_gate({name: metrics[name] for name in PROMOTION_SCENARIOS})
    assert verdict['passed']
    print('PASS: model/checkpoint identity, contract, 128 ONNX/PyTorch vectors, six simulator scenarios, gradients, final-report scenario selection.')


if __name__ == '__main__':
    main()
