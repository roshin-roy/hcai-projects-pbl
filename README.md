# Human-Centric AI — Project Work (SoSe 2026)

Coursework for *Human-Centric Artificial Intelligence* at TUHH. Four small web
applications, all living inside one Django project, each exploring a different
angle on how people and machine learning models work together.

**Group 40**

| Name | Matriculation number |
| --- | --- |
| Moniya Mohan | 675659 |
| Roshin Roy | 674412 |

Repository: <https://github.com/roshin-roy/hcai-projects-pbl>

---

## Running the project

The project needs Python 3.10 or newer.

```bash
git clone https://github.com/roshin-roy/hcai-projects-pbl

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>. The root URL redirects to the landing page,
which lists all four projects.

A note on the first run: Project 3 trains a text classifier on 120,000 AG News
articles the first time you open it. That takes roughly half a minute, plus
another 10–20 seconds for the deferral experiments. Everything is cached to
`project3/data/*.joblib` afterwards, so subsequent loads are instant. If the
page seems to hang on first visit, it is working — watch the terminal.

---

## Repository layout

```
pbl/                  Django settings, root URL configuration
home/                 Landing page listing the group and the four projects
demos/                Provided demo app 
project1/             Supervised learning interface
project2/             Explainability
project3/             Learning-to-defer + active learning
project4/             Preference elicitation
templates/            HTML templates, one directory per app
static/               Shared stylesheet
media/                Generated plots (created at runtime, not committed)
```

---

## Project 1 — Supervised learning interface

**Where:** `project1/`, `templates/project1/` · **URL:** `/project1/`

Upload a CSV, look at it, then train a model on it. The CSV is expected in the
format described in the assignment: first row is feature names, last column is
the target.

The app detects automatically whether the target looks like a classification or
a regression problem (few distinct values and integer-like → classification),
but the user can override that choice, because the guess is wrong often enough
to be annoying — the Iris dataset with numeric species labels is exactly the
ambiguous case.

**What the user controls:** the model, the hyperparameter to sweep is fixed per
model but the range is swept automatically, the train/test split proportion,
and the scoring function. **What happens automatically:** feature scaling (only
for the models that need it, and fitted on the training split only), the sweep
itself, and picking the best configuration.

Five models are available for each problem type:

| Classification | Swept parameter | Regression | Swept parameter |
| --- | --- | --- | --- |
| Logistic Regression | `C` | Ridge Regression | `alpha` |
| K-Nearest Neighbors | `n_neighbors` | K-Nearest Neighbors | `n_neighbors` |
| SVM (RBF) | `C` | SVR (RBF) | `C` |
| Decision Tree | `max_depth` | Decision Tree | `max_depth` |
| Random Forest | `n_estimators` | Random Forest | `n_estimators` |

Results are shown as a table of train/test scores per hyperparameter value plus
a curve, which makes overfitting visible rather than something you have to
infer from two numbers.

Uploaded files are saved with a UUID prefix so two people using the app at the
same time do not overwrite each other. Datasets that are too small, contain
missing values, or have no numeric feature columns are caught with a readable
error instead of a stack trace — a 12-row CSV used to crash the KNN sweep
because the sweep asked for more neighbours than there were training rows.

---

## Project 2 — Explainability

**Where:** `project2/`, `templates/project2/` · **URL:** `/project2/`

Built on the Palmer Penguins dataset, predicting species from the other seven
features. Everything is on a single page and everything is linked: whichever
model class and λ you pick at the top is the model used by the counterfactual
and feature-effect sections below.

**Interpretability vs. complexity.** Decision trees are trained for every
`max_leaf_nodes` from 2 to 30; logistic regressions are trained with L1
penalty across eleven values of `C`. Moving the λ slider re-selects the model
maximising `acc_test − λ·Ω(f)`. For trees, Ω is the number of leaves. For
logistic regression we use the number of features with at least one non-zero
coefficient across the three one-vs-rest classifiers — the L1 penalty drives
whole features out of the model, so this is the natural analogue of "number of
leaves", and the interface labels it exactly that way rather than the vaguer
"number of non-zero coefficients".

**Counterfactuals.** Pick a penguin and a target species, and the app samples
`N` perturbed versions of that penguin, keeps the ones the model classifies as
the target, and ranks them by MAD-weighted L1 distance. Numeric features are
perturbed with Gaussian noise scaled to each feature's standard deviation;
categorical features (island, sex) are one-hot encoded and perturbed by
flipping the whole group at once, so a generated counterfactual is never a
penguin that is 0.4 Biscoe and 0.6 Dream. If no counterfactual is found, the
search widens: more samples and larger noise, up to five rounds.

**Feature effects.** PDP and ALE for the four numerical features, computed by
hand rather than with a library, one curve per species. The two tell different
stories where features are correlated, which is the point of showing both.

On the assignment's question about derivatives: for logistic regression the
partial derivative of the predicted probability can be written down in closed
form via the sigmoid derivative. A decision tree is piecewise constant, so it
has no derivative at its split points and differencing over bins is the only
option. Our implementation uses the binned finite-difference form for both
models, so the curves are comparable and the tree case is handled correctly.

---

## Project 3 — Learning to defer, and active learning

**Where:** `project3/`, `templates/project3/` · **URL:** `/project3/`
**Report:** downloadable from the project page (`project3/static/project3/report.pdf`)

The AG News dataset, four topics, 120,000 training and 7,600 test articles. The
CSVs are committed to the repository so the project runs without network
access.

**Baseline.** TF-IDF followed by logistic regression, reaching about 92.2%
test accuracy. Deliberately a simple, fast, well-understood model — the
interesting part of this project is the deferral logic, not squeezing out
another point of accuracy.

**Simulated expert.** The expert is good at World and Sports (95%) and weak at
Business and Sci/Tech (55%), giving an overall test accuracy of about 75.7% —
well below the classifier. This is the point: an expert who is worse on average
can still improve the team, provided the system learns *where* they are better.
Wrong answers are drawn uniformly over the other three classes, and the
randomness is seeded per row so the expert gives the same answer to the same
article every time it is queried.

**Two deferral strategies** are implemented and shown side by side. The first
is a confidence threshold tuned on training data. The second is a learned
policy: a logistic regression over the classifier's probability vector,
maximum confidence, entropy, and predicted class, trained on the oracle signal
"defer if and only if the classifier is wrong and the expert is right". Both
thresholds are tuned for system accuracy rather than for deferral-classifier
accuracy, since combined performance is what actually matters.

Results are reported as a full outcome breakdown rather than a single number,
because "the system deferred 6% of the time" hides whether those deferrals
helped:

| Outcome | Meaning |
| --- | --- |
| Helpful | Deferred; classifier wrong, expert right — the deferral fixed it |
| Harmful | Deferred; classifier right, expert wrong — the deferral broke it |
| Both correct | Deferred unnecessarily, but no damage done |
| Both wrong | Deferred; neither was right — accuracy unchanged |
| Missed | Not deferred, but the expert would have been right |
| Correctly kept | Not deferred, classifier right |

Separating "harmful" from "both wrong" matters. Only the first actually costs
accuracy relative to the classifier alone, and an earlier version of this code
conflated them, which made the policy look considerably better behaved than it
was.

Both strategies land around 92.5–92.6% system accuracy at a 4–6% deferral rate
— a small but real gain over the classifier alone, and the per-class deferral
rates confirm the policy sends disproportionately many Business and Sci/Tech
articles to the expert, which is the wrong direction given this expert's
profile and is discussed in the report.

**Active learning.** Here we drop the assumption that expert labels exist.
Starting from nothing, the system queries the expert on selected training
articles and estimates their per-class competence from the answers.
Uncertainty sampling (query the lowest-confidence articles) is compared against
random sampling over five seeds, up to a budget of 3,000 queries in rounds of
300. The reasoning for uncertainty sampling: deferral is only ever relevant
where the classifier might be wrong, so spending the budget in the uncertain
region is where knowing the expert's competence actually changes a decision.

**Optional human-in-the-loop.** A separate page where you label articles
yourself instead of the simulation. Competence is estimated against the true
training labels, so every answer counts from the first one. The live
performance readout underneath uses the *simulated* expert on the test set —
you cannot hand-label 7,600 articles — and the page says so explicitly rather
than presenting it as a measurement of your own team.

---

## Project 4 — Preference elicitation

**Where:** `project4/`, `templates/project4/` · **URL:** `/project4/`
**Report:** downloadable from the project page (`project4/static/project4/report.pdf`)

A study design comparing two ways of asking someone what films they like. The
study is not run — the deliverable is the protocol and a working interface that
could run it.

**Features.** From the IMDB 5000 metadata we build an interpretable feature
vector per film: 22 binary genre indicators, release-decade indicators,
runtime, IMDB score, log vote count, and three content-rating groups. Every
feature is chosen to be something a person could plausibly have an opinion
about, since the whole model rests on the assumption that utility is linear in
these features — "prefers long films", "prefers 90s films", and "prefers
animation" are all meaningful statements; a latent embedding dimension is not.

**Preference models.** Pairwise choices are fitted with Bradley-Terry. For the
ranking interface we extend to Plackett-Luce, which factorises a full ranking
into a sequence of choices: the winner is chosen from all ten films, then the
runner-up from the remaining nine, and so on. Bradley-Terry falls out as the
two-item special case, so both interfaces are fitted under one consistent
model. Both are fitted by maximum likelihood with L2 regularisation, which is
doing real work here — there are around 35 features and only a handful of
interactions to learn from.

**The interfaces.** Design 1 presents two films and asks which you would rather
watch, twelve times. Design 2 presents ten films and asks you to rank them, three
times. Each participant does both, in an order randomised per session to
counterbalance learning and fatigue effects. A consent page precedes the tasks,
ranking submissions are validated both in the browser and on the server so a
partial or duplicated ranking cannot be submitted, and each pairwise submission
is checked against the task it claims to answer, so a double-click or a
refresh cannot silently skip a comparison.

At the end the app shows what it thinks you like and dislike, estimated
separately from each interface, which makes the two methods directly
comparable within a single session.

---

## Notes on the code

**Caching.** Project 3 caches its trained models and computed metrics in
`project3/data/*.joblib`. Delete them to force a full retrain:

```bash
rm project3/data/defer_metrics.joblib project3/data/defer_policy.joblib
rm project3/data/active_metrics.joblib
rm project3/data/baseline_model.joblib project3/data/baseline_metrics.joblib
```

**Generated plots.** Matplotlib figures are written to `media/` with random
filenames and loaded as images, following the pattern in the provided demo app.
The directory is created automatically and is excluded from version control.

**Session state.** Projects 1 and 4 keep per-user state in the Django session
rather than the database, since nothing needs to outlive a browser session.
One consequence worth knowing: opening the same project in two tabs will make
them interfere, as they share a session.

**Reproducibility.** Random seeds are fixed throughout — train/test splits,
the simulated expert's errors, the active-learning runs — so the numbers quoted
above and in the reports are reproducible from a clean checkout.

