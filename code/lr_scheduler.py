# lr_scheduler.py
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(
    optimizer,
    total_epochs=200,
    warmup_epochs=10,
    T_0=50,  # 첫 번째 restart 주기 (warmup 이후)
    T_mult=2,  # restart 시 주기를 몇 배로 늘릴지 결정
    initial_lr=1e-4,
    eta_min=1e-6,
    start_epoch=0,
):
    """
    Warmup 후 Cosine Annealing with Warmup/Restarts 스케줄러.

    학습률 스케줄은 두 단계로 구성됩니다:

    1. Warmup (epoch < warmup_epochs):
       lr = initial_lr * (epoch / warmup_epochs)

    2. Cosine Annealing with Warm Restarts (epoch >= warmup_epochs):
       - warmup 이후의 epoch을 epoch' = epoch - warmup_epochs로 치환합니다.
       - 현재 restart cycle을 결정합니다.
         각 cycle의 길이는 T_i = T_0 * (T_mult^i)입니다.
       - cycle 내 상대적 위치(epoch_in_cycle)에 따라 학습률은 아래와 같이 계산됩니다.

         cosine_decay = 0.5 * (1 + cos(pi * (epoch_in_cycle) / T_i))
         lr = eta_min + (initial_lr - eta_min) * cosine_decay

       LambdaLR의 lr_lambda는 optimizer의 lr에 곱해지는 비율을 반환해야 하므로,
       위 식을 initial_lr으로 나누어 반환합니다.

    Args:
        optimizer (Optimizer): 학습에 사용되는 optimizer.
        total_epochs (int): 전체 에폭 수 (참고용).
        warmup_epochs (int): 초기 warmup 기간.
        T_0 (int): 첫 번째 cosine annealing 주기 (warmup 이후).
        T_mult (int): 각 restart마다 주기를 늘리는 배수.
        initial_lr (float): optimizer에 지정한 초기 학습률.
        eta_min (float): 최소 학습률.
        start_epoch (int): 시작 에폭 (보통 0).

    Returns:
        scheduler: LambdaLR 스케줄러 인스턴스.
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # warmup: 선형 증가
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # cosine annealing with restarts
            epoch_prime = epoch - warmup_epochs  # warmup 이후의 epoch 수
            cycle = 0
            T_i = T_0
            # 현재 cycle 찾기: epoch_prime가 T_i보다 크면 남은 값을 다음 cycle로 넘김
            while epoch_prime >= T_i:
                epoch_prime -= T_i
                cycle += 1
                T_i = T_0 * (T_mult**cycle)
            # 현재 cycle 내에서의 비율에 따른 cosine decay 계산
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch_prime / T_i))
            # lr_lambda는 optimizer의 lr에 곱해지는 비율이어야 하므로,
            # initial_lr에서 eta_min까지 선형 보간하는 비율 반환
            return eta_min / initial_lr + (1 - eta_min / initial_lr) * cosine_decay

    # start_epoch에 맞춰 last_epoch 설정 (보통 -1로 시작)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch - 1)
    return scheduler
