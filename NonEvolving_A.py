# To activate virtual environment
# source 776_env/bin/activate
# Experiment with Tit-for-Tat as one of 8 opponents 
# Average score is average score per match against all 8 fixed representatives
# Average best score is average score of the best individual in the population

import random
import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Constants
# -------------------------
POPULATION_SIZE = 20
GENERATIONS = 50
NUM_MATINGS = 10
GAME_ROUNDS = 151
MUTATION_RATE = 0.01
CHROMOSOME_LENGTH = 70

# Payoff matrix
PAYOFF = {
    ('C', 'C'): 3,
    ('C', 'D'): 0,
    ('D', 'C'): 5,
    ('D', 'D'): 1
}

# -------------------------
# Player Class
# -------------------------
class AxelrodGAPlayer(axl.Player):
    """
    Player using 70-bit chromosome:
    - Bits 0-63: Lookup table for 64 possible 3-move histories (2^6)
    - Bits 64-69: Premise (6 bits representing assumed history before game starts)
    """
    name = "Axelrod GA Player"
    classifier = {"memory_depth": 3, "stochastic": False}

    def __init__(self, chromosome):
        super().__init__()
        if len(chromosome) != CHROMOSOME_LENGTH:
            raise ValueError(f"Chromosome must be {CHROMOSOME_LENGTH} bits")
        self.chromosome = chromosome
        self.lookup_table = chromosome[:64]
        self.premise = chromosome[64:]

    def strategy(self, opponent):
        # Get the relevant history (last 3 moves from each player)
        my_history = list(self.history)
        opp_history = list(opponent.history)
        
        # If we don't have 3 moves yet, use the premise
        if len(my_history) < 3:
            # Premise gives us the assumed previous 6 moves (3 mine, 3 theirs)
            # Fill in from premise what we don't have
            moves_needed = 3 - len(my_history)
            my_premise = self.premise[3-moves_needed:3]
            opp_premise = self.premise[6-moves_needed:6]
            
            my_last_3 = [axl.Action.C if bit == 0 else axl.Action.D for bit in my_premise] + my_history
            opp_last_3 = [axl.Action.C if bit == 0 else axl.Action.D for bit in opp_premise] + opp_history
        else:
            my_last_3 = my_history[-3:]
            opp_last_3 = opp_history[-3:]
        
        # Convert to index (0-63)
        # Index = my_moves (3 bits) + opponent_moves (3 bits)
        index = 0
        for i, move in enumerate(my_last_3):
            if move == axl.Action.D:
                index += (1 << (5 - i))
        for i, move in enumerate(opp_last_3):
            if move == axl.Action.D:
                index += (1 << (2 - i))
        
        # Look up the move in the lookup table
        gene = self.lookup_table[index]
        return axl.Action.C if gene == 0 else axl.Action.D

# -------------------------
# Genetic Algorithm Functions
# -------------------------
def random_chromosome():
    """Generate a random 70-bit chromosome"""
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]

def evaluate_individual(chromosome, representatives):
    """
    Evaluate an individual against all representatives.
    Returns average score per game.
    """
    player = AxelrodGAPlayer(chromosome)
    total_score = 0
    
    for rep in representatives:
        opponent = rep.__class__()  # Create fresh instance
        match = axl.Match((player, opponent), turns=GAME_ROUNDS)
        match.play()
        score, _ = match.final_score()
        total_score += score
    
    # Return average score per game
    return total_score / len(representatives)

def scaling_function(score, mean, std):
    """
    Axelrod's scaling function:
    - Average score -> 1 mating
    - One std above average -> 2 matings
    - Linear scaling
    """
    if std == 0:
        return 1.0
    return max(0, 1.0 + (score - mean) / std)

def select_parent(population, mating_weights):
    """Select a parent based on mating weights (fitness proportional)"""
    total = sum(mating_weights)
    if total == 0:
        return random.choice(population)
    
    pick = random.uniform(0, total)
    cumulative = 0
    for individual, weight in zip(population, mating_weights):
        cumulative += weight
        if cumulative >= pick:
            return individual
    return population[-1]

def crossover(parent1, parent2):
    """Single-point crossover"""
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome, rate=MUTATION_RATE):
    """Flip each bit with probability = mutation_rate"""
    return [1 - bit if random.random() < rate else bit for bit in chromosome]

# -------------------------
# Main GA Loop
# -------------------------
def run_axelrod_ga(seed=42):
    """Run Axelrod's genetic algorithm experiment"""
    random.seed(seed)
    
    # Initialize representatives (fixed opponents)
    representatives = [
        axl.TitForTat(),
        axl.Defector(),
        axl.Cooperator(),
        axl.FirstByGrofman(),
        axl.Random(),
        axl.FirstByDavis(),
        axl.FirstByJoss(),
        axl.FirstByGraaskamp()
    ]
    
    print("Axelrod 1980s Genetic Algorithm Replication")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Matings per generation: {NUM_MATINGS}")
    print(f"Game rounds: {GAME_ROUNDS}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print(f"Representatives: {len(representatives)}")
    print("=" * 60)
    
    # Initialize population with random chromosomes
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]
    
    # Track statistics
    best_scores = []
    avg_scores = []
    std_scores = []
    
    # Evolution loop
    for generation in range(GENERATIONS):
        # Evaluate all individuals
        scores = [evaluate_individual(chrom, representatives) for chrom in population]
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        best_score = max(scores)
        
        best_scores.append(best_score)
        avg_scores.append(mean_score)
        std_scores.append(std_score)
        
        print(f"Gen {generation+1:2d}: Best={best_score:6.1f}  Avg={mean_score:6.1f}  Std={std_score:5.1f}")
        
        # Calculate mating weights using scaling function
        mating_weights = [scaling_function(s, mean_score, std_score) for s in scores]
        
        # Create next generation through mating
        new_population = []
        for _ in range(NUM_MATINGS):
            parent1 = select_parent(population, mating_weights)
            parent2 = select_parent(population, mating_weights)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population
    
    # Final evaluation
    final_scores = [evaluate_individual(chrom, representatives) for chrom in population]
    best_idx = final_scores.index(max(final_scores))
    best_chromosome = population[best_idx]
    
    # Calculate representative average scores
    print("\n" + "=" * 60)
    print("REPRESENTATIVE STRATEGY PERFORMANCE")
    print("=" * 60)
    rep_avg_scores = []
    for rep in representatives:
        total = 0
        for chrom in population:
            player = AxelrodGAPlayer(chrom)
            opponent = rep.__class__()
            match = axl.Match((player, opponent), turns=GAME_ROUNDS)
            match.play()
            _, opp_score = match.final_score()
            total += opp_score
        avg_score = total / len(population)
        rep_avg_scores.append((rep.__class__.__name__, avg_score))
        print(f"{rep.__class__.__name__:20s}: {avg_score:6.1f}")
    
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print(f"Final best score (avg per game): {final_scores[best_idx]:.1f}")
    print(f"Final avg score (avg per game): {np.mean(final_scores):.1f}")
    print("=" * 60)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    generations_x = range(1, GENERATIONS + 1)
    
    plt.plot(generations_x, best_scores, 'b-o', label='Best Score', linewidth=2)
    plt.plot(generations_x, avg_scores, 'k-o', label='Average Score', linewidth=2)
    
    plt.xlabel('Generations', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Score', fontsize=12, fontweight='bold')
    plt.title("Axelrod's Genetic Algorithm\nNon-Evolving Environment", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    return best_chromosome, population, representatives

# -------------------------
# Analysis Functions
# -------------------------
def analyze_best_chromosome(chromosome, representatives):
    """Print analysis of the best evolved chromosome"""
    print("\n" + "=" * 60)
    print("BEST CHROMOSOME ANALYSIS")
    print("=" * 60)
    
    # Cooperation rates against each representative
    print("\nCooperation Rates vs Each Representative:")
    print("-" * 60)
    player = AxelrodGAPlayer(chromosome)
    overall_coop_count = 0
    overall_total_moves = 0
    
    for rep in representatives:
        opponent = rep.__class__()
        match = axl.Match((player, opponent), turns=GAME_ROUNDS)
        match.play()
        
        coop_count = sum(1 for move in player.history if move == axl.Action.C)
        total_moves = len(player.history)
        coop_rate = (coop_count / total_moves * 100) if total_moves > 0 else 0
        
        overall_coop_count += coop_count
        overall_total_moves += total_moves
        
        print(f"  {rep.__class__.__name__:20s}: {coop_rate:5.1f}% ({coop_count}/{total_moves} moves)")
    
    overall_coop_rate = (overall_coop_count / overall_total_moves * 100) if overall_total_moves > 0 else 0
    print("-" * 60)
    print(f"  {'OVERALL':20s}: {overall_coop_rate:5.1f}% ({overall_coop_count}/{overall_total_moves} moves)")
    
    # Premise (assumed initial history)
    print("\nPremise (assumed initial 6 moves):")
    print(f"  My assumed moves:       {['C' if bit==0 else 'D' for bit in chromosome[64:67]]}")
    print(f"  Opponent assumed moves: {['C' if bit==0 else 'D' for bit in chromosome[67:70]]}")
    
    # Sample of lookup table
    print("\nLookup Table (sample of 64 entries):")
    print("  Format: [My last 3] vs [Opp last 3] -> Action")
    
    moves = ['C', 'D']
    for i in range(8):
        # Extract my 3 moves from index i (bits 2, 1, 0)
        my_moves = ''.join([moves[(i >> (2-k)) & 1] for k in range(3)])
        
        for j in range(8):
            # Extract opponent's 3 moves from index j (bits 2, 1, 0)
            opp_moves = ''.join([moves[(j >> (2-k)) & 1] for k in range(3)])
            
            index = (i << 3) + j
            action = 'C' if chromosome[index] == 0 else 'D'
            print(f"  [{my_moves}] vs [{opp_moves}] -> {action}")

    
    print("\nFull chromosome (70 bits):")
    print(f"  {chromosome}")

if __name__ == "__main__":
    best_chrom, final_pop, reps = run_axelrod_ga()
    analyze_best_chromosome(best_chrom, reps)