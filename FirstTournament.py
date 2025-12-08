# To activate virtual environment
# source 776_env/bin/activate

import axelrod as axl

first_tournament_participants_ordered_by_reported_rank = [s() for s in axl.axelrod_first_strategies]
number_of_strategies = len(first_tournament_participants_ordered_by_reported_rank)

tournament = axl.Tournament(
     players=first_tournament_participants_ordered_by_reported_rank,
     turns=200,
     repetitions=5,
     seed=1408,
)

results = tournament.play()

for name in results.ranked_names:
    print(name)