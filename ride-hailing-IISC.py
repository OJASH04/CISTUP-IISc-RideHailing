from stable_baselines3 import A2C
import matplotlib.pyplot as plt
from pyhailing import RidehailEnv
from ray.rllib.agents.ppo import PPOTrainer

def main(render:bool=False):

    env_config = RidehailEnv.DIMACS_CONFIGS.SUI
    env_config["nickname"] = "testing"
    
    env = RidehailEnv(**env_config)

    # Stable Baseline
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # RL-Lib
    # config = {
    #     # Environment (RLlib understands openAI gym registered strings).
    #     "env": "RidehailEnv",
    #     # Use 2 environment workers (aka "rollout workers") that parallelly
    #     # collect samples from their own environment clone(s).
    #     "num_workers": 2,
    #     # Change this to "framework: torch", if you are using PyTorch.
    #     # Also, use "framework: tf2" for tf2.x eager execution.
    #     "framework": "tf",
    #     # Tweak the default model provided automatically by RLlib,
    #     # given the environment's observation- and action spaces.
    #     "model": {
    #         "fcnet_hiddens": [64, 64],
    #         "fcnet_activation": "relu",
    #     },
    #     # Set up a separate evaluation worker set for the
    #     # `trainer.evaluate()` call after training (see below).
    #     "evaluation_num_workers": 1,
    #     # Only for evaluation runs, render the env.
    #     "evaluation_config": {
    #         "render_env": True,
    #     }
    # }   
    
    # trainer = PPOTrainer(config=config)

    # # Run it for n training iterations. A training iteration includes
    # # parallel sample collection by the environment workers as well as
    # # loss calculation on the collected batch and a model update.
    # for _ in range(3):
    #     print(trainer.train())

    # # Evaluate the trained Trainer (and render each timestep to the shell's
    # # output).
    # trainer.evaluate()
    
    all_eps_rewards = []
    for episode in range(RidehailEnv.DIMACS_NUM_EVAL_EPISODES):

        obs = env.reset()
        terminal = False
        reward = 0

        if render:
            rgb = env.render()
            plt.imshow(rgb)
            plt.show()

        while not terminal:

            #action,  _states = model.predict(obs)
            action = env.get_random_action()
            next_obs, new_rwd, terminal, _ = env.step(action)

            reward += new_rwd
            obs = next_obs

            if render:
                rgb = env.render()
                plt.imshow(rgb)
                plt.show()

        print(f"Episode {episode} complete. Reward: {reward}")
        all_eps_rewards.append(reward)

    mean_reward = sum(all_eps_rewards)/len(all_eps_rewards)
    print(f"All episodes complete. Average reward: {mean_reward}")


if __name__ == "__main__":
    main()