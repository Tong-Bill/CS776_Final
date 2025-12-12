# To activate virtual environment
# source 776_env/bin/activate
# Co-evolutionary version: Population plays against itself
# Creates dynamic environment where strategies evolve in response to each other

import random
import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from datetime import datetime

# -------------------------
# Constants
# -------------------------
POPULATION_SIZE = 20
GENERATIONS = 50
NUM_MATINGS = 10
GAME_ROUNDS = 151
MUTATION_RATE = 0.01
CHROMOSOME_LENGTH = 70
NUM_RUNS = 30

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
def run_coevolutionary_ga(seed=None):
    """Run co-evolutionary genetic algorithm where population plays against itself"""
    if seed is not None:
        random.seed(seed)
    
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
    final_scores = [evaluate_individual_coevolution(chrom, population) for chrom in population]
    best_idx = final_scores.index(max(final_scores))
    best_chromosome = population[best_idx]
    best_final_score = final_scores[best_idx]
    avg_final_score = np.mean(final_scores)
    final_diversity = diversity_scores[-1]
    
    return {
        'best_scores': best_scores,
        'avg_scores': avg_scores,
        'diversity_scores': diversity_scores,
        'best_chromosome': best_chromosome,
        'best_final_score': best_final_score,
        'avg_final_score': avg_final_score,
        'final_diversity': final_diversity,
        'final_population': population
    }

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
        
        coop_rate = (coop_count / total_moves * 100) if total_moves > 0 else 0
        print(f"  vs Individual {i+1:2d}:  {coop_rate:5.1f}% ({coop_count}/{total_moves} moves)")
    
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

def analyze_chromosome_data_coevolution(chromosome, population):
    """
    Analyze chromosome and return data structure (for JSON serialization)
    """
    player = AxelrodGAPlayer(chromosome)
    
    cooperation_rates = []
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
        coop_rate = (coop_count / total_moves * 100) if total_moves > 0 else 0
        
        overall_coop_count += coop_count
        overall_total_moves += total_moves
        
        cooperation_rates.append({
            'individual': i + 1,
            'rate': float(coop_rate),
            'coop_count': int(coop_count),
            'total_moves': int(total_moves)
        })
    
    overall_coop_rate = (overall_coop_count / overall_total_moves * 100) if overall_total_moves > 0 else 0
    
    cooperate_genes = sum(1 for g in chromosome[:64] if g == 0)
    defect_genes = 64 - cooperate_genes
    
    return {
        'cooperation_rates': cooperation_rates,
        'overall_cooperation': {
            'rate': float(overall_coop_rate),
            'coop_count': int(overall_coop_count),
            'total_moves': int(overall_total_moves)
        },
        'premise': {
            'my_moves': ['C' if bit==0 else 'D' for bit in chromosome[64:67]],
            'opp_moves': ['C' if bit==0 else 'D' for bit in chromosome[67:70]]
        },
        'lookup_table_summary': {
            'cooperate_responses': int(cooperate_genes),
            'defect_responses': int(defect_genes),
            'cooperate_percentage': float(cooperate_genes/64*100),
            'defect_percentage': float(defect_genes/64*100)
        }
    }

# -------------------------
# Data Saving Functions
# -------------------------
def save_results(all_results, overall_best_chromosome, overall_best_population, filename_prefix="CoEvolution_Results"):
    """
    Save all results to files for later use
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare aggregated data
    all_best_scores = [result['best_scores'] for result in all_results]
    all_avg_scores = [result['avg_scores'] for result in all_results]
    all_diversity_scores = [result['diversity_scores'] for result in all_results]
    
    avg_best_scores = np.mean(all_best_scores, axis=0).tolist()
    avg_avg_scores = np.mean(all_avg_scores, axis=0).tolist()
    avg_diversity_scores = np.mean(all_diversity_scores, axis=0).tolist()
    
    # Final scores
    final_best_scores = [result['best_final_score'] for result in all_results]
    final_avg_scores = [result['avg_final_score'] for result in all_results]
    final_diversity_scores = [result['final_diversity'] for result in all_results]
    overall_best_score = max(final_best_scores)
    
    # Create comprehensive JSON data
    json_data = {
        'metadata': {
            'timestamp': timestamp,
            'num_runs': NUM_RUNS,
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'num_matings': NUM_MATINGS,
            'game_rounds': GAME_ROUNDS,
            'mutation_rate': MUTATION_RATE,
            'chromosome_length': CHROMOSOME_LENGTH,
            'mode': 'CO-EVOLUTION'
        },
        'aggregated_scores': {
            'avg_best_scores_per_generation': avg_best_scores,
            'avg_avg_scores_per_generation': avg_avg_scores,
            'avg_diversity_per_generation': avg_diversity_scores,
            'final_best_score_mean': float(np.mean(final_best_scores)),
            'final_best_score_std': float(np.std(final_best_scores)),
            'final_avg_score_mean': float(np.mean(final_avg_scores)),
            'final_avg_score_std': float(np.std(final_avg_scores)),
            'final_diversity_mean': float(np.mean(final_diversity_scores)),
            'final_diversity_std': float(np.std(final_diversity_scores)),
            'overall_best_score': float(overall_best_score)
        },
        'best_chromosome': {
            'chromosome': overall_best_chromosome,
            'score': float(overall_best_score)
        }
    }
    
    # Analyze best chromosome for detailed stats
    chromosome_analysis = analyze_chromosome_data_coevolution(overall_best_chromosome, overall_best_population)
    json_data['best_chromosome_analysis'] = chromosome_analysis
    
    # Save JSON file
    json_filename = f"{filename_prefix}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n✓ Saved aggregated results to: {json_filename}")
    
    # Save chromosome as pickle
    pickle_filename = f"{filename_prefix}_chromosome_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(overall_best_chromosome, f)
    print(f"✓ Saved best chromosome to: {pickle_filename}")
    
    # Save formatted text output
    text_filename = f"{filename_prefix}_analysis_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CO-EVOLUTIONARY GA EXPERIMENT RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of runs: {NUM_RUNS}\n")
        f.write(f"Population size: {POPULATION_SIZE}\n")
        f.write(f"Generations: {GENERATIONS}\n")
        f.write(f"Game rounds: {GAME_ROUNDS}\n")
        f.write(f"Mode: CO-EVOLUTION\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("AGGREGATED PERFORMANCE (averaged over 30 runs)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Final best score (averaged): {json_data['aggregated_scores']['final_best_score_mean']:.1f} ")
        f.write(f"(±{json_data['aggregated_scores']['final_best_score_std']:.1f})\n")
        f.write(f"Final avg score (averaged):  {json_data['aggregated_scores']['final_avg_score_mean']:.1f} ")
        f.write(f"(±{json_data['aggregated_scores']['final_avg_score_std']:.1f})\n")
        f.write(f"Final diversity (averaged):  {json_data['aggregated_scores']['final_diversity_mean']:.1f} ")
        f.write(f"(±{json_data['aggregated_scores']['final_diversity_std']:.1f})\n")
        f.write(f"Overall best score:          {overall_best_score:.1f}\n")
        f.write("=" * 60 + "\n\n")
        
        # Write chromosome analysis
        f.write("=" * 60 + "\n")
        f.write("BEST CHROMOSOME ANALYSIS (Best across all 30 runs)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Cooperation Rates Against Population:\n")
        f.write("-" * 60 + "\n")
        for rate_data in chromosome_analysis['cooperation_rates']:
            f.write(f"  vs Individual {rate_data['individual']:2d}:  {rate_data['rate']:5.1f}% ")
            f.write(f"({rate_data['coop_count']}/{rate_data['total_moves']} moves)\n")
        
        overall = chromosome_analysis['overall_cooperation']
        f.write("-" * 60 + "\n")
        f.write(f"  {'OVERALL':20s}: {overall['rate']:5.1f}% ")
        f.write(f"({overall['coop_count']}/{overall['total_moves']} moves)\n\n")
        
        f.write("Premise (assumed initial 6 moves):\n")
        f.write(f"  My assumed moves:       {chromosome_analysis['premise']['my_moves']}\n")
        f.write(f"  Opponent assumed moves: {chromosome_analysis['premise']['opp_moves']}\n\n")
        
        lookup = chromosome_analysis['lookup_table_summary']
        f.write("Lookup Table Summary:\n")
        f.write(f"  Cooperate responses: {lookup['cooperate_responses']}/64 ({lookup['cooperate_percentage']:.1f}%)\n")
        f.write(f"  Defect responses:    {lookup['defect_responses']}/64 ({lookup['defect_percentage']:.1f}%)\n\n")
        
        f.write("Full chromosome (70 bits):\n")
        f.write(f"  {overall_best_chromosome}\n")
    
    print(f"✓ Saved formatted analysis to: {text_filename}")
    print("\nAll results saved successfully!")
    
    return json_filename, pickle_filename, text_filename

# -------------------------
# Loading Functions (for future use)
# -------------------------
def load_results(json_filename):
    """Load results from JSON file"""
    with open(json_filename, 'r') as f:
        return json.load(f)

def load_chromosome(pickle_filename):
    """Load chromosome from pickle file"""
    with open(pickle_filename, 'rb') as f:
        return pickle.load(f)

def create_player_from_chromosome(chromosome):
    """Create an AxelrodGAPlayer from a saved chromosome"""
    return AxelrodGAPlayer(chromosome)

# -------------------------
# Multiple Runs
# -------------------------
def run_multiple_experiments():
    """Run the co-evolutionary GA experiment 30 times and aggregate results"""
    print("Axelrod Co-evolutionary Genetic Algorithm")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Matings per generation: {NUM_MATINGS}")
    print(f"Game rounds: {GAME_ROUNDS}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print(f"Number of runs: {NUM_RUNS}")
    print("Mode: CO-EVOLUTION (population plays against itself)")
    print("=" * 60)
    
    all_results = []
    all_best_scores = []
    all_avg_scores = []
    all_diversity_scores = []
    
    # Store the overall best chromosome and its score
    overall_best_chromosome = None
    overall_best_score = -float('inf')
    overall_best_population = None
    
    for run in range(NUM_RUNS):
        print(f"\nRun {run + 1}/{NUM_RUNS}...")
        result = run_coevolutionary_ga(seed=None)
        all_results.append(result)
        all_best_scores.append(result['best_scores'])
        all_avg_scores.append(result['avg_scores'])
        all_diversity_scores.append(result['diversity_scores'])
        
        # Track overall best
        if result['best_final_score'] > overall_best_score:
            overall_best_score = result['best_final_score']
            overall_best_chromosome = result['best_chromosome']
            overall_best_population = result['final_population']
        
        print(f"  Final best: {result['best_final_score']:.1f}, Final avg: {result['avg_final_score']:.1f}, Diversity: {result['final_diversity']:.1f}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS ACROSS 30 RUNS")
    print("=" * 60)
    
    # Average scores per generation
    avg_best_scores = np.mean(all_best_scores, axis=0)
    avg_avg_scores = np.mean(all_avg_scores, axis=0)
    avg_diversity_scores = np.mean(all_diversity_scores, axis=0)
    
    # Final scores
    print("\nAGGREGATED PERFORMANCE")
    print("=" * 60)
    final_best_scores = [result['best_final_score'] for result in all_results]
    final_avg_scores = [result['avg_final_score'] for result in all_results]
    final_diversity_scores = [result['final_diversity'] for result in all_results]
    
    print(f"Final best score (averaged over 30 runs): {np.mean(final_best_scores):.1f} (±{np.std(final_best_scores):.1f})")
    print(f"Final avg score (averaged over 30 runs):  {np.mean(final_avg_scores):.1f} (±{np.std(final_avg_scores):.1f})")
    print(f"Final diversity (averaged over 30 runs):  {np.mean(final_diversity_scores):.1f} (±{np.std(final_diversity_scores):.1f})")
    print(f"Overall best score across all runs:       {overall_best_score:.1f}")
    print("=" * 60)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    generations_x = range(1, GENERATIONS + 1)
    
    # Plot 1: Scores over time
    ax1.plot(generations_x, avg_avg_scores, 'k-o', label='Average Score (avg)', linewidth=2)
    ax1.set_xlabel('Generations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax1.set_title(f"Prisoner's Dilemma\nEvolving Environment (averaged over {NUM_RUNS} runs)", 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(False)
    
    # Plot 2: Diversity over time
    ax2.plot(generations_x, avg_diversity_scores, 'g-o', label='Genetic Diversity (avg)', linewidth=2)
    ax2.set_xlabel('Generations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Hamming Distance', fontsize=12, fontweight='bold')
    ax2.set_title(f"Population Diversity (averaged over {NUM_RUNS} runs)", 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze the overall best chromosome
    analyze_best_chromosome_coevolution(overall_best_chromosome, overall_best_population)
    
    # Save all results to files
    save_results(all_results, overall_best_chromosome, overall_best_population)
    
    return all_results, overall_best_chromosome

if __name__ == "__main__":
    all_results, best_chromosome = run_multiple_experiments()