"""인과(causal) 단조 디코더: 행별 클래스 확률 → 라벨(1~6).

모든 디코더는 (proba, relative_time_s, experiment_id) 만의 순수 함수이고,
같은 실험의 '현재 + 과거' 행만 사용한다 (실험별로 시간순 정렬 후 왼쪽→오른쪽 한 번 훑기).
→ 미래 행을 빼도(앞 60%), 행 순서를 섞어도, 다른 실험이 있든 없든 각 행의 결과가 같다.
시간 의존 파라미터는 모두 '초' 단위 (Test는 500행 다운샘플로 10/15초 간격이 섞여 있음).

사용 예
    params = fit_params(train_label, train_time, train_exp, train_prior=None)   # 학습 fold 라벨로만 추정
    pred = decode(proba, time_s, exp_id, params, method='hmm', alpha=0.5, clamp=True)

method
  'argmax'  : 행별 argmax (기준)
  'cummax'  : argmax의 누적 최댓값 (단조 클램프). 한 번 튄 값이 영구히 남는 lock-in 위험
  'persist' : 최근 persist_s 초 동안 계속 L보다 높게 예측되어야 L을 올림 (= 누적최댓값(최근 persist_s초 argmax의 최솟값))
  'ordinal' : P(단계 >= s) > theta 인 가장 큰 s, 누적 최댓값 (확률 임계값 히스테리시스)
  'hmm'     : 좌→우 HMM 전방 필터 (머무르거나 앞으로만, 4·5 건너뛰기 허용).
              연속시간 마르코프 생성행렬 Q(초 단위) → 간격 Δt초면 expm(QΔt) 전이.
              방출 = (p(y|x) / 학습 사전확률)^alpha (분류기 사후확률 → 우도 변환, alpha<1은 행 간 상관 보정)
              예측 = argmax(필터 사후확률 × 클래스 가중치), clamp=True면 누적 최댓값
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

K = 6


@dataclass
class DecoderParams:
    Q: np.ndarray                 # (6,6) 생성행렬, 초당 비율
    init: np.ndarray              # (6,) 실험 첫 관측의 단계 분포
    prior: np.ndarray             # (6,) 분류기 학습 데이터의 클래스 비율 (사후→우도 변환용)
    info: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def trans(self, dt: float) -> np.ndarray:
        key = round(float(dt), 6)
        if key not in self._cache:
            self._cache[key] = np.eye(K) if key <= 0 else np.clip(expm(self.Q * key), 0.0, None)
        return self._cache[key]


def _groups(exp_id, time_s, key=None):
    """실험별 시간순(동시각이면 key, 없으면 입력 순) 인덱스 목록."""
    exp_id = np.asarray(exp_id).astype(str)
    t = np.asarray(time_s, dtype=float)
    out = []
    uniq, inv = np.unique(exp_id, return_inverse=True)
    for g in range(len(uniq)):
        idx = np.where(inv == g)[0]
        order = np.lexsort((np.asarray(key)[idx],) + (t[idx],)) if key is not None else np.argsort(t[idx], kind='mergesort')
        out.append(idx[order])
    return out


def fit_params(label, time_s, exp_id, train_prior=None, smooth: float = 0.5) -> DecoderParams:
    """학습 fold 라벨 시퀀스만으로 CTMC 파라미터 추정 (MLE: 이탈률 = 이탈 횟수 / 체류 시간).

    train_prior: 분류기가 학습한 행들의 클래스 비율. None이면 주어진 라벨 행의 비율.
    smooth: 다음 단계 전이 횟수에 더할 의사 카운트 (s→s+1, s→s+2 에만; 관측되지 않은 건너뛰기도 아주 작게 허용).
    """
    label = np.asarray(label, dtype=int)
    t = np.asarray(time_s, dtype=float)
    dwell = np.zeros(K)
    nxt = np.zeros((K, K))
    first = np.zeros(K)
    for idx in _groups(exp_id, t):
        y, tt = label[idx] - 1, t[idx]
        assert (np.diff(y) >= 0).all(), '학습 라벨이 실험 안에서 단조 증가가 아님'
        first[y[0]] += 1
        dt = np.diff(tt)
        np.add.at(dwell, y[:-1], dt)            # 구간 [t_i, t_{i+1}) 은 y_i 상태에 머문 시간으로 계산
        ch = np.where(y[1:] != y[:-1])[0]
        np.add.at(nxt, (y[ch], y[ch + 1]), 1)
    Q = np.zeros((K, K))
    rates = np.zeros(K)
    for s in range(K - 1):
        cnt = nxt[s].copy()
        cnt[s + 1] += smooth
        if s + 2 < K:
            cnt[s + 2] += smooth * 0.2
        n_out = nxt[s].sum()
        rates[s] = (n_out + smooth) / max(dwell[s], 1.0)
        Q[s] = rates[s] * cnt / cnt.sum()
        Q[s, s] = -rates[s]
    init = (first + 0.1) / (first + 0.1).sum()
    if train_prior is None:
        train_prior = np.bincount(label - 1, minlength=K) / len(label)
    prior = np.asarray(train_prior, dtype=float) + 1e-6
    return DecoderParams(Q=Q, init=init, prior=prior / prior.sum(),
                         info={'mean_dwell_s': (1 / np.where(rates > 0, rates, np.nan)).round(1).tolist(),
                               'transitions': nxt.astype(int).tolist()})


def _rolling_min_causal(t, a, w):
    """(t_i - w, t_i] 에 있는 값의 최솟값. w<=0 이면 a 그대로."""
    if w <= 0:
        return a.copy()
    start = np.searchsorted(t, t - w, side='right')
    return np.array([a[start[i]:i + 1].min() for i in range(len(a))])


def decode(proba, time_s, exp_id, params: DecoderParams | None = None, method: str = 'hmm',
           alpha: float = 1.0, clamp: bool = True, beta: float = 0.0, persist_s: float = 0.0,
           theta: float = 0.5, eps: float = 1e-4, key=None) -> np.ndarray:
    """확률 (n,6) → 라벨 (n,) 1~6. 입력 행 순서대로 돌려준다.

    beta: Macro F1용 클래스 가중치 = prior^-beta (사후확률에 곱함; hmm/argmax/cummax/persist 공통)
    eps : 방출 확률 하한 (분류기가 0에 가깝게 준 단계로도 회복 가능하게)
    """
    P = np.asarray(proba, dtype=float)
    t = np.asarray(time_s, dtype=float)
    n = len(P)
    out = np.zeros(n, dtype=int)
    w = np.ones(K) if (params is None or beta == 0) else params.prior ** (-beta)
    for idx in _groups(exp_id, t, key):
        p, tt = P[idx], t[idx]
        if method in ('argmax', 'cummax', 'persist'):
            a = np.argmax(p * w, axis=1)
            if method == 'cummax':
                a = np.maximum.accumulate(a)
            elif method == 'persist':
                a = np.maximum.accumulate(_rolling_min_causal(tt, a, persist_s))
        elif method == 'ordinal':
            ge = np.cumsum((p * w / (p * w).sum(1, keepdims=True))[:, ::-1], axis=1)[:, ::-1]   # P(>= s)
            a = (ge > theta).sum(1) - 1
            a = np.maximum.accumulate(a)
        elif method == 'hmm':
            e = (np.clip(p, eps, None) / params.prior) ** alpha
            a = np.zeros(len(idx), dtype=int)
            post = None
            for i in range(len(idx)):
                pred = params.init if i == 0 else post @ params.trans(tt[i] - tt[i - 1])
                post = pred * e[i]
                post = post / post.sum()
                a[i] = np.argmax(post * w)
            if clamp:
                a = np.maximum.accumulate(a)
        else:
            raise ValueError(method)
        out[idx] = a + 1
    return out
