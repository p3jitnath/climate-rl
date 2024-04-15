from ray import tune

config = {}

config["reinforce"] = {"learning_rate": tune.uniform(1e-4, 1e-2)}

config["ddpg"] = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    "tau": tune.uniform(0.001, 0.10),
    "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "exploration_noise": tune.uniform(0.1, 0.3),
    "policy_frequency": tune.choice([x for x in range(1, 11)]),
    "noise_clip": tune.choice([x * 0.1 for x in range(1, 11)]),
}

config["dpg"] = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    # "tau": tune.uniform(0.001, 0.10),
    # "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "exploration_noise": tune.uniform(0.1, 0.3),
    "policy_frequency": tune.choice([x for x in range(1, 11)]),
}

config["td3"] = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    "tau": tune.uniform(0.001, 0.10),
    "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "policy_noise": tune.uniform(0.1, 0.3),
    "exploration_noise": tune.uniform(0.1, 0.3),
    "policy_frequency": tune.choice([x for x in range(1, 11)]),
    "noise_clip": tune.choice([x * 0.1 for x in range(1, 11)]),
}

config["ppo"] = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    "num_minibatches": tune.choice([64, 128, 256, 512, 1024]),
    "update_epochs": tune.choice([x for x in range(1, 11)]),
    "clip_coef": tune.uniform(0.1, 0.3),
    "vf_coef": tune.uniform(0.5, 0.8),
}

config["trpo"] = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    "num_minibatches": tune.choice([64, 128, 256, 512, 1024]),
    "update_epochs": tune.choice([x for x in range(1, 11)]),
    "clip_coef": tune.uniform(0.1, 0.3),
    "vf_coef": tune.uniform(0.5, 0.8),
}

config["sac"] = {
    "tau": tune.uniform(0.001, 0.10),
    "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "policy_lr": tune.uniform(1e-4, 1e-2),
    "q_lr": tune.uniform(1e-4, 1e-2),
    "exploration_noise": tune.uniform(0.1, 0.3),
    "policy_frequency": tune.choice([x for x in range(1, 11)]),
    "target_network_frequency": tune.choice([x for x in range(1, 11)]),
    "noise_clip": tune.choice([x * 0.1 for x in range(1, 11)]),
    "alpha": tune.choice([x * 0.1 for x in range(1, 6)]),
}

config["tqc"] = {
    "tau": tune.uniform(0.001, 0.10),
    "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "n_quantiles": tune.choice([5 * x for x in range(1, 10)]),
    "n_critics": tune.choice([5 * x for x in range(1, 10)]),
    "actor_adam_lr": tune.uniform(1e-4, 1e-2),
    "critic_adam_lr": tune.uniform(1e-4, 1e-2),
    "alpha_adam_lr": tune.uniform(1e-4, 1e-2),
    "policy_frequency": tune.choice([x for x in range(1, 11)]),
    "target_network_frequency": tune.choice([x for x in range(1, 11)]),
}
