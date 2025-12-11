import numpy as np
from scipy.integrate import odeint
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
import dask.dataframe as dd



# SIR Models

def sir_basic(y, t, alpha, gamma, d, v):
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dIdt, dRdt]


def sir_delayed(y, t, alpha, gamma, d, v, tau):
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dIdt, dRdt]


def sir_latent(y, t, alpha, gamma, d, v, delta):
    S, L, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dLdt = gamma * S * I - delta * L - d * L
    dIdt = delta * L - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dLdt, dIdt, dRdt]


def sir_reinfection(y, t, alpha, gamma, d, v, e):
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S + e * R
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - (d + e) * R
    return [dSdt, dIdt, dRdt]

# Simulation functions with noise
def simulate_sir_basic(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v = theta
    y0 = [20, 10, 0]  # Initial conditions: S, I, R
    sol = odeint(sir_basic, y0, t_obs, args=(alpha, gamma, d, v))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()


def simulate_sir_delayed(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, tau = theta
    y0 = [20, 10, 0]
    sol = odeint(sir_delayed, y0, t_obs, args=(alpha, gamma, d, v, tau))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

def simulate_sir_latent(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, delta = theta
    y0 = [20, 0, 10, 0]  # S, L, I, R
    sol = odeint(sir_latent, y0, t_obs, args=(alpha, gamma, d, v, delta))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol[:, [0,2,3]].flatten()  # Observe S, I, R

def simulate_sir_reinfection(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, e = theta
    y0 = [20, 10, 0]
    sol = odeint(sir_reinfection, y0, t_obs, args=(alpha, gamma, d, v, e))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()



# Distance function
def distance(sim_data, observed_data):
    return np.sum((sim_data - observed_data) ** 2)


# Normalize weights
def normalize_weights(weights):
    total_weight = np.sum(weights)
    return np.array(weights) / total_weight if total_weight > 0 else np.zeros_like(weights)


# Sample initial particles
def sample_initial_particles(N, models, model_prior, epsilons, observed_data, t_obs):
    particles, weights = {m: [] for m in range(len(models))}, {m: [] for m in range(len(models))}
    num_particles, attempt_count, max_attempts = 0, 0, N * 100000

    while num_particles < N and attempt_count < max_attempts:
        m = np.random.choice(len(models), p=model_prior)
        params = [prior.rvs() for prior in models[m]['priors']]
        sim_data = models[m]['simulate'](params, t_obs)
        attempt_count += 1

        if distance(sim_data, observed_data) <= epsilons[0]:
            particles[m].append(params)
            weights[m].append(1.0)
            num_particles += 1
            print(f"Accepted {num_particles}/{N} particles (Model {m})")

    if num_particles < N:
        raise RuntimeError(f"Stopped after {attempt_count} attempts. Only {num_particles} particles accepted.")

    for m in particles:
        weights[m] = normalize_weights(weights[m]) if particles[m] else []
    return particles, weights


# Perturb parameters
def perturb_parameters(prev_params, model):
    return [prev + sigma*np.random.uniform(-1, 1) for prev, sigma in zip(prev_params, model['kernel_sigma'])]


# Check prior bounds
def is_within_prior_bounds(params, priors):
    return all(prior.ppf(0) <= param <= prior.ppf(1) for param, prior in zip(params, priors))


# Compute weight
def compute_weight(proposed_params, particles, weights, model):
    prior_density = np.prod([prior.pdf(p) for prior, p in zip(model['priors'], proposed_params)])
    denominator = sum(
        w * np.prod([norm(p_old, sigma).pdf(p_new) for p_old, p_new, sigma in zip(p, proposed_params, model['kernel_sigma'])])
        for p, w in zip(particles, weights)
    )
    return prior_density / (denominator + 1e-12)


# Generate new population
def generate_population(N, T, t, particles, weights, models, model_prior, epsilons, observed_data, t_obs):
    new_particles, new_weights, num_accepted = {m: [] for m in range(len(models))}, {m: [] for m in
                                                                                     range(len(models))}, 0
    total_runs=0
    while num_accepted < N:
        total_runs+=1
        m = np.random.choice(len(models), p=model_prior)
        if not particles[t - 1][m]:
            continue
        idx = np.random.choice(len(particles[t - 1][m]), p=weights[t - 1][m])
        prev_params = particles[t - 1][m][idx]
        proposed_params = perturb_parameters(prev_params, models[m])

        if not is_within_prior_bounds(proposed_params, models[m]['priors']):
            continue

        sim_data = models[m]['simulate'](proposed_params, t_obs)
        if distance(sim_data, observed_data) <= epsilons[t]:
            weight = compute_weight(proposed_params, particles[t - 1][m], weights[t - 1][m], models[m])
            new_particles[m].append(proposed_params)
            new_weights[m].append(weight)
            num_accepted += 1
            print(f"Population {t}: Accepted {num_accepted}/{N} particles (Model {m})")

    acceptance_rate=num_accepted/total_runs
    print(f"Acceptance rate: {acceptance_rate:.2%} for ")


    for m in new_particles:
        new_weights[m] = normalize_weights(new_weights[m]) if new_particles[m] else []
    return new_particles, new_weights








# Main ABC SMC function
def abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs):
    particles, weights = sample_initial_particles(N, models, model_prior, epsilons, observed_data, t_obs)
    all_particles, all_weights = [particles], [weights]

    for t in range(1, T):
        print(f"Processing population {t}/{T - 1}")
        new_particles, new_weights = generate_population(N, T, t, all_particles, all_weights, models, model_prior,
                                                         epsilons, observed_data, t_obs)
        all_particles.append(new_particles)
        all_weights.append(new_weights)


    return all_particles, all_weights


if __name__ == "__main__":
    # Define SIR models
    models = [
        {   # Model 0: Basic SIR (alpha, gamma, d, v,S0)
            'simulate': simulate_sir_basic,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(37, 100)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05]
        },
        {   # Model 1: Delayed SIR (placeholder)
            'simulate': simulate_sir_delayed,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(37, 100)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.1]
        },
        {   # Model 2: Latent phase SIR
            'simulate': simulate_sir_latent,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(37, 100)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.1]
        },
        {   # Model 3: Reinfection SIR
            'simulate': simulate_sir_reinfection,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(37, 100)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.05]
        }
    ]
    model_prior = [0.25, 0.25, 0.25, 0.25]  # Uniform prior

    # Generate observed data from true model (Basic SIR)
    t_obs = np.linspace(0, 11, 12)
    true_params = [0.01, 0.005, 0.001, 0.1]  # alpha, gamma, d, v
    observed_data = simulate_sir_basic(true_params, t_obs, noise_std=0.2)

    # ABC SMC parameters
    N = 1000  # Particles per population
    T = 15   # Populations
    epsilons = [ 100, 90, 80, 73, 70, 60, 50, 40, 30, 25, 20, 16, 15, 14, 13.8]  # Tolerance schedule
    print("\nbefore abc")

    # Run ABC SMC
    particles, weights = abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs)

    # Analyze results
    final_particles = particles[-1]
    final_weights = weights[-1]
    model_counts = {m: len(final_particles.get(m, [])) for m in range(len(models))}
    total = sum(model_counts.values())
    print("\nModel posterior probabilities:")
    for m in range(len(models)):
        print(f"Model {m}: {model_counts[m]/total:.2%}")

    # Plot results
    plt.figure()
    for m in range(len(models)):
        probs = [len(p.get(m, []))/N for p in particles]
        plt.plot(probs, label=f'Model {m}')
    plt.xlabel('Population')
    plt.ylabel('Model Probability')
    plt.title('SIR Model Selection Across Populations')
    plt.legend()
    plt.show()

    population = final_particles
    model_counts = {m: len(population[m]) for m in population}

    models = list(model_counts.keys())  # Model indices
    counts = list(model_counts.values())  # Particle counts

    plt.figure(figsize=(8, 5))
    plt.bar(models, counts, tick_label=[f"Model {m}" for m in models], color='skyblue', edgecolor='black')
    plt.xlabel("Model Index")
    plt.ylabel("Number of Particles")
    plt.title(f"Particle Count per Model")
    plt.show()
    