# To activate virtual environment
# source 776_env/bin/activate
# Co-evolutionary version: Population plays against itself
# Creates dynamic environment where strategies evolve in response to each other

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
            moves_needed = 3 - len(my_history)
            my_premise = self.premise[3-moves_needed:3]
            opp_premise = self.premise[6-moves_needed:6]
            
            my_last_3 = [axl.Action.C if bit == 0 else axl.Action.D for bit in my_premise] + my_history
            opp_last_3 = [axl.Action.C if bit == 0 else axl.Action.D for bit in opp_premise] + opp_history
        else:
            my_last_3 = my_history[-3:]
            opp_last_3 = opp_history[-3:]
        
        # Convert to index (0-63)
        index = 0
        for i, move in enumerate(my_last_3):
            if move == axl.Action.D:
                index += (1 << (5 - i))
        for i, move in enumerate(opp_last_3):
            if move == axl.Action.D:
                index += (1 << (2 - i))
        
        gene = self.lookup_table[index]
        return axl.Action.C if gene == 0 else axl.Action.D

# -------------------------
# Genetic Algorithm Functions
# -------------------------
def random_chromosome():
    """Generate a random 70-bit chromosome"""
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]

def evaluate_individual_coevolution(chromosome, population):
    """
    Evaluate an individual against ALL other members of the population.
    Returns average score per game.
    """
    player = AxelrodGAPlayer(chromosome)
    total_score = 0
    num_opponents = 0
    
    for opponent_chrom in population:
        # Skip playing against itself
        if opponent_chrom == chromosome:
            continue
            
        opponent = AxelrodGAPlayer(opponent_chrom)
        match = axl.Match((player, opponent), turns=GAME_ROUNDS)
        match.play()
        score, _ = match.final_score()
        total_score += score
        num_opponents += 1
    
    # Return average score per game
    return total_score / num_opponents if num_opponents > 0 else 0

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

def calculate_population_diversity(population):
    """Calculate genetic diversity as average hamming distance between chromosomes"""
    if len(population) < 2:
        return 0
    
    total_distance = 0
    comparisons = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            hamming_dist = sum(1 for a, b in zip(population[i], population[j]) if a != b)
            total_distance += hamming_dist
            comparisons += 1
    
    return total_distance / comparisons if comparisons > 0 else 0

# -------------------------
# Main Co-evolutionary GA Loop
# -------------------------
def run_coevolutionary_ga(seed=1408):
    """Run co-evolutionary genetic algorithm where population plays against itself"""
    random.seed(seed)
    
    print("Axelrod Co-evolutionary Genetic Algorithm")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Matings per generation: {NUM_MATINGS}")
    print(f"Game rounds: {GAME_ROUNDS}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print("Mode: CO-EVOLUTION (population plays against itself)")
    print("=" * 60)
    
    # Initialize population with random chromosomes
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]
    
    # Track statistics
    best_scores = []
    avg_scores = []
    std_scores = []
    diversity_scores = []
    
    # Evolution loop
    for generation in range(GENERATIONS):
        # Evaluate all individuals against the population
        scores = [evaluate_individual_coevolution(chrom, population) for chrom in population]
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        best_score = max(scores)
        diversity = calculate_population_diversity(population)
        
        best_scores.append(best_score)
        avg_scores.append(mean_score)
        std_scores.append(std_score)
        diversity_scores.append(diversity)
        
        print(f"Gen {generation+1:2d}: Best={best_score:6.1f}  Avg={mean_score:6.1f}  "
              f"Std={std_score:5.1f}  Diversity={diversity:5.1f}")
        
        # Calculate mating weights using scaling function
        mating_weights = [scaling_function(s, mean_score, std_score) for s in scores]
        
        # Create next generation through mating (Axelrod's original method)
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
    final_scores = [evaluate_individual_coevolution(chrom, population) for chrom in population]
    best_idx = final_scores.index(max(final_scores))
    best_chromosome = population[best_idx]
    
    print("\n" + "=" * 60)
    print("Co-evolution Complete!")
    print(f"Final best score (avg per game): {final_scores[best_idx]:.1f}")
    print(f"Final avg score (avg per game): {np.mean(final_scores):.1f}")
    print(f"Final diversity: {diversity_scores[-1]:.1f}")
    print("=" * 60)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    generations_x = range(1, GENERATIONS + 1)
    
    # Plot 1: Scores over time
    ax1.plot(generations_x, avg_scores, 'k-o', label='Average Score', linewidth=2)
    ax1.set_xlabel('Generations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax1.set_title("Prisoner's Dilemma\nEvolving Environment", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(False)
    
    # Plot 2: Diversity over time
    ax2.plot(generations_x, diversity_scores, 'g-o', label='Genetic Diversity', linewidth=2)
    ax2.set_xlabel('Generations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Hamming Distance', fontsize=12, fontweight='bold')
    ax2.set_title("Population Diversity", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return best_chromosome, population

# -------------------------
# Analysis Functions
# -------------------------
def analyze_best_chromosome_coevolution(chromosome, population):
    """Print analysis of the best evolved chromosome in co-evolutionary context"""
    print("\n" + "=" * 60)
    print("BEST CHROMOSOME ANALYSIS (CO-EVOLUTION)")
    print("=" * 60)
    
    # Cooperation rates against population
    print("\nCooperation Rate Against Population:")
    print("-" * 60)
    player = AxelrodGAPlayer(chromosome)
    overall_coop_count = 0
    overall_total_moves = 0
    
    for i, opponent_chrom in enumerate(population):
        if opponent_chrom == chromosome:
            continue
        
        opponent = AxelrodGAPlayer(opponent_chrom)
        match = axl.Match((player, opponent), turns=GAME_ROUNDS)
        match.play()
        
        coop_count = sum(1 for move in player.history if move == axl.Action.C)
        total_moves = len(player.history)
        
        overall_coop_count += coop_count
        overall_total_moves += total_moves
        
        #if i < 5:  # Show first 5 opponents
        coop_rate = (coop_count / total_moves * 100) if total_moves > 0 else 0
        print(f"  vs Individual {i+1:2d}:  {coop_rate:5.1f}% ({coop_count}/{total_moves} moves)")
    
    #print(f"  ... (vs {len(population)-6} more individuals)")
    overall_coop_rate = (overall_coop_count / overall_total_moves * 100) if overall_total_moves > 0 else 0
    print("-" * 60)
    print(f"  {'OVERALL':20s}: {overall_coop_rate:5.1f}% ({overall_coop_count}/{overall_total_moves} moves)")
    
    # Premise (assumed initial history)
    print("\nPremise (assumed initial 6 moves):")
    print(f"  My assumed moves:       {['C' if bit==0 else 'D' for bit in chromosome[64:67]]}")
    print(f"  Opponent assumed moves: {['C' if bit==0 else 'D' for bit in chromosome[67:70]]}")
    
    # Lookup table summary
    print("\nLookup Table Summary:")
    cooperate_genes = sum(1 for g in chromosome[:64] if g == 0)
    defect_genes = 64 - cooperate_genes
    print(f"  Cooperate responses: {cooperate_genes}/64 ({cooperate_genes/64*100:.1f}%)")
    print(f"  Defect responses:    {defect_genes}/64 ({defect_genes/64*100:.1f}%)")
    
    print("\nFull chromosome (70 bits):")
    print(f"  {chromosome}")

if __name__ == "__main__":
    best_chrom, final_pop = run_coevolutionary_ga()
    analyze_best_chromosome_coevolution(best_chrom, final_pop)