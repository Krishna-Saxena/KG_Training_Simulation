from datetime import date, datetime, timedelta

from sim_utils import *

####### Setup Sim #######
##  - Define the layers of skills
N_BEGIN_SKILLS = 6
N_INTER_SKILLS = 4
N_EXP_SKILLS = 3

SKILL_IDX_TO_IDS = {i:1000+i for i in range(N_BEGIN_SKILLS)}
SKILL_IDX_TO_IDS.update({j+N_BEGIN_SKILLS:2000+j for j in range(N_INTER_SKILLS)})
SKILL_IDX_TO_IDS.update({k+N_BEGIN_SKILLS+N_INTER_SKILLS: 3000+k for k in range(N_EXP_SKILLS)})

SKILL_IDS_TO_IDX = {v:k for k,v in SKILL_IDX_TO_IDS.items()}

##  - Create a Knowledge Graph for beginner, intermediate, and expert skills
## First, define the dependency map from beginner to intermediate skills
##		A[i, :] is the weights of how strongly the beginner skills impact the i^th intermediate skill
A = np.array([
	[0.8, 0.8, 0.5, 0.2, 0.1, 0.1],
	[0.1, 0.5, 0.5, 0.8, 0.5, 0.1],
	[0.2, 0.2, 0.5, 0.1, 0.8, 0.1],
	[0.1, 0.1, 0.8, 0.5, 0.8, 0.8]
])
# A = make_dependency_layer(N_BEGIN_SKILLS, N_INTER_SKILLS)

## Now define the uniqueness of the intermediate skills
##		U_B,(i,i) = 0.55 means that a student's ability in the i^th intermediate skill is 55% dependent on new skills and 45% dependent on beginner skills. (the breakdown of this 45% is the distribution formed after 1-normalizing i^th row of A)
U_B = make_uniqueness_vector(N_INTER_SKILLS, 0.5, 0.2)
## Now define the dependency map from intermediate to expert skills
B = np.array([
	[0.8, 0.8, 0.2, 0.2],
	[0.8, 0.8, 0.8, 0.5],
	[0.2, 0.8, 0.2, 0.8]
])
# B = make_dependency_layer(N_INTER_SKILLS, N_EXP_SKILLS)

## Finally, define the uniqueness of the expert skills
##		we assume that these skills are more specialized and therefore less dependent on prerequisites
U_C = make_uniqueness_vector(N_EXP_SKILLS, 0.85, 0.1)

##  - Initialize student and mentor characteristics
N_TRAINEES = 10
N_MENTORS = 5

TRAINEE_IDX_TO_IDS = {i: f'T{i}' for i in range(N_TRAINEES)}
TRAINEE_IDS_TO_IDX= {v:k for k,v in TRAINEE_IDX_TO_IDS.items()}

MENTOR_IDX_TO_IDS = {i: f'M{i}' for i in range(N_MENTORS)}
MENTOR_IDS_TO_IDX= {v:k for k,v in MENTOR_IDX_TO_IDS.items()}

SIM_START_DATE = date(2024, 1, 1)
N_SIM_DATES = 500

if __name__ == '__main__':
	sim_timestamp_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

	## Assume that all trainees start with lower preparation for more advanced skills
	trainee_preparation = np.vstack((
		rng.uniform(0, 0.1, (N_BEGIN_SKILLS, N_TRAINEES)),
		rng.uniform(0, 0.01, (N_INTER_SKILLS, N_TRAINEES)),
		rng.uniform(0, 0.001, (N_EXP_SKILLS, N_TRAINEES))
	))

	## Assume that mentors have different standards for what is low, intermediate, and high quality work
	mentor_low_threshold = rng.uniform(0.3, 0.7, (N_MENTORS,))
	mentor_high_threshold = rng.uniform(mentor_low_threshold, 0.95, (N_MENTORS,))
	mentor_threshold_mat = np.hstack((mentor_low_threshold[:, None], mentor_high_threshold[:, None]))

	## skill_update_mat_{i,j} := a student who gains delta ability in skill j will earn delta*skill_update_mat_{i,j} abilitiy in skill i from the practice
	skill_update_mat = make_update_mat((A, B), (U_B, U_C))

	sim_date = SIM_START_DATE

	updated_trainee_preparation = trainee_preparation.copy()
	reseed()

	with open(f'./sim_results/simulation_{sim_timestamp_str}.tsv', 'w') as f:
		f.write('date\ttrainee_tested\tmentor_testing\tskill_tested\trating_given\n')

		for _ in range(N_SIM_DATES):
			sim_skills, sim_raw_scores, sim_ratings, sim_mentors, updated_trainee_preparation = simulate_training_day(
				updated_trainee_preparation, mentor_threshold_mat, skill_update_mat
			)
			sim_date += timedelta(days=rng.poisson(2))
			for t, (m, s, r) in enumerate(zip(sim_mentors, sim_skills, sim_ratings)):
				f.write(
					f'{sim_date.strftime(DATE_FMT)}\t{TRAINEE_IDX_TO_IDS[t]}\t{MENTOR_IDX_TO_IDS[m]}\t{SKILL_IDX_TO_IDS[s]}\t{r.value}\n')