from enum import StrEnum
import numpy as np

DATE_FMT = '%Y-%m-%d'

rng = np.random.default_rng(seed=321)

def reseed(seed=321):
	global rng
	rng = np.random.default_rng(seed=seed)

class Rating(StrEnum):
	LOW = 'LOW'
	MEDIUM = 'MED'
	ADVANCED = 'ADV'

def make_dependency_layer(n_from, n_to):
	return rng.random((n_to, n_from))

def make_uniqueness_vector(n_skills, mode_uniqueness, range_uniqueness):
	return rng.uniform(
		max(mode_uniqueness-range_uniqueness/2, 0),
		min(mode_uniqueness+range_uniqueness/2, 1),
		(n_skills,)
	)

def make_prob_mat(A, axis=1):
	return A / np.linalg.norm(A, ord=1, axis=axis, keepdims=True)

def make_update_mat(dependency_mats, uniqueness_vecs):
	# TODO (1): transpose update mat (use left state_vector.T @ update_mat multiplies)
	n_skills_by_level = []
	for dependency_mat in dependency_mats:
		n_skills_by_level.append(dependency_mat.shape[1])
	n_skills_by_level.append(dependency_mats[-1].shape[0])

	uniqueness_vecs_ext = [np.ones((n_skills_by_level[0],))]
	uniqueness_vecs_ext.extend(uniqueness_vecs)

	n_skills_total = sum(n_skills_by_level)
	# make block-upper triangular update mat
	update_mat = np.zeros((n_skills_total, n_skills_total))
	start_row = 0
	for i, n_skills_i in enumerate(n_skills_by_level):
		# set the diagonal elements to the uniqueness matrices
		np.put(update_mat, [i+i*n_skills_total for i in range(start_row, start_row+n_skills_i)], uniqueness_vecs_ext[i])
		# build the rest of the row
		start_col, cum_mat = start_row+n_skills_i, np.eye(n_skills_i)
		for j in range(i+1, len(n_skills_by_level)):
			n_skills_j = n_skills_by_level[j]
			ij_dep_mat = (1 - uniqueness_vecs_ext[j])[None,:]*make_prob_mat(dependency_mats[j-1].T)
			cum_mat = cum_mat @ ij_dep_mat
			update_mat[start_row:start_row+n_skills_i, start_col:start_col+n_skills_j] = cum_mat
			start_col += n_skills_j
		start_row += n_skills_i
	return update_mat

def simulate_training_day(trainee_prep_mat, mentor_threshold_mat, update_mat):
	sim_skills = [None] * trainee_prep_mat.shape[1]
	sim_raw_scores = [None] * trainee_prep_mat.shape[1]
	sim_ratings = [None]*trainee_prep_mat.shape[1]
	sim_mentors = [None]*trainee_prep_mat.shape[1]
	updated_trainee_prep_mat_T = trainee_prep_mat.T.copy()

	# TODO (2): parallelize loop
	for t in range(trainee_prep_mat.shape[1]):
		skill_distr = make_prob_mat(trainee_prep_mat[:, t:t+1]*(trainee_prep_mat[:, t:t+1] < 0.95), 0)
		skill = rng.choice(trainee_prep_mat.shape[0], p=skill_distr.squeeze())
		mentor = rng.choice(mentor_threshold_mat.shape[0])

		curr_skill_prep = trainee_prep_mat[skill, t]
		raw_score = rng.normal(1.5 * curr_skill_prep, 0.5 * (curr_skill_prep * (1 - curr_skill_prep)) ** 0.5)
		raw_score = np.clip(raw_score, 0.0, 1.0)
		rating = Rating.LOW if raw_score < mentor_threshold_mat[mentor, 0] else \
			(Rating.MEDIUM if raw_score <= mentor_threshold_mat[mentor, 1] else Rating.ADVANCED)

		if raw_score > curr_skill_prep:
			updated_trainee_prep_mat_T[t, :] += (raw_score-curr_skill_prep)*update_mat[:, skill]

		sim_skills[t] = skill
		sim_raw_scores[t] = raw_score
		sim_ratings[t] = rating
		sim_mentors[t] = mentor

	updated_trainee_prep_mat_T.clip(0.0, 1.0)
	return sim_skills, sim_raw_scores, sim_ratings, sim_mentors, updated_trainee_prep_mat_T.T