"""Compatibility launcher for the shared online SAC runner."""

import hydra

if __package__:
    from .rl_online import train
else:
    from rl_online import train


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='sac_taco_state',
)
def run(cfg) -> None:
    train(cfg)


if __name__ == '__main__':
    run()
