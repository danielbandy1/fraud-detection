# PaySim Financial Fraud Detection: Senior ML & Risk Interview Defense Guide

**AUTHORED BY GEMINI (extending HELIOS & ORION frameworks)**

This guide provides Daniel Bandy with the financial ML domain knowledge, quantitative depth, and architectural defense needed to command data science and machine learning engineering interviews in fintech, banking, and fraud analytics (e.g., Stripe, Plaid, Capital One, JPMorgan Chase, American Express).

---

## 1. Project Summary & Elevator Pitch (30-second summary)

> *"In this project, I developed an end-to-end fraud detection system on the 6.36-million-transaction PaySim mobile money dataset, isolating the 2.77 million high-risk transfer and cash-out transactions containing all 8,213 frauds. Rather than relying on standard random cross-validation—which creates massive lookahead leakage by letting models memorize future account trajectories—I enforced a strict forward-time split (steps 1–500 for training, 501–600 for validation, and 601–743 for held-out testing) with rolling account history features computed strictly from earlier time steps.*
>
> *Crucially, I designed and compared two distinct production paradigms: an **in-line Pre-Authorization model** (13 strictly pre-event features, achieving a test Average Precision of 0.8366 and AUROC of 0.9896 at a 3.67% holdout base rate) and an **asynchronous Post-Transaction model** (19 features incorporating post-settlement balance adjustments, achieving a test AP of 0.8803 and AUROC of 0.9956). Using a prespecified 10:1 false-negative to false-positive operational cost function tuned exclusively on validation labels, the model captures 99.1% of fraudulent transactions while maintaining an 87.8% alert precision."*

---

## 2. The Five Core Methodological Traps & How We Solved Them

### Trap 1: Temporal Data Leakage & Lookahead Bias
- **The Pitfall in Naive Implementations:**
  - Standard academic tutorials and baseline Kaggle notebooks perform random splitting: `train_test_split(df, test_size=0.2, random_state=42)`.
  - In financial transactions, accounts execute repeated transactions over time. Random splitting randomly distributes past and future transactions of the same account across train and test sets.
  - Furthermore, naive rolling graph features (e.g., "total transaction count for sender", "inbound transfer velocity for recipient") compute aggregations over the entire dataset window.
  - As a result, the training set observes signals that depend on future events, allowing the gradient booster to memorize account life-cycles rather than learning generalized fraud patterns.
- **Our Solution:**
  - **Temporal Splitting by Simulation Step:** PaySim steps represent 1-hour increments across 31 simulated days (743 total steps).
    - **Training Set (Steps 1–500 / ~21 days):** 2,645,779 eligible transactions (5,561 frauds, 0.210% fraud prevalence).
    - **Validation Set (Steps 501–600 / ~4 days):** 81,080 transactions (1,052 frauds, 1.297% fraud prevalence).
    - **Held-Out Test Set (Steps 601–743 / ~6 days):** 43,550 transactions (1,600 frauds, 3.674% fraud prevalence).
  - **Strictly Earlier-Step Feature Extraction:**
    - Rolling velocity features (`orig_tx_count`, `orig_unique_dest`, `dest_recv_count`, `orig_step_gap`) are calculated strictly from transactions occurring at `step < current_step`.
    - **Same-step isolation:** Transactions occurring within the exact same hour cannot observe one another.
    - Verified via deterministic property-based regression tests (`test_strict_prior_step_feature_isolation` and `test_future_invariance`) that inject future fraudulent records and assert past feature representations remain identical to bit-level precision.

---

### Trap 2: Post-Transaction Leakage vs. Real-Time Pre-Authorization
- **The Pitfall:**
  - In the raw PaySim schema, each record includes `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`.
  - Many published models compute features such as `orig_balance_delta = newbalanceOrig - oldbalanceOrg` and flags like `orig_drained = (newbalanceOrig == 0)`.
  - In real banking infrastructure, **Pre-Authorization fraud detection** occurs *before* funds are debited or credited. At authorization time, the post-transaction balances do not exist on the ledger.
  - Furthermore, PaySim's underlying agent simulator exhibits an artificial generator artifact: fraudulent transfers systematically drain origin accounts to exactly zero, while destination accounts often display an unupdated `newbalanceDest == oldbalanceDest`. A model utilizing post-transaction balances learns the synthetic simulator's quirks rather than real-world behavioral fraud signals.
- **Our Solution: Two Disjoint Operating Models**
  - We engineered and benchmarked two separate pipelines on the exact same temporal split:

| Architectural Property | Post-Transaction Model | Pre-Authorization Model |
|---|---|---|
| **Operational Role** | Post-clearing fraud triage / AML investigation | In-line payment authorization / Card-swipe blocking |
| **Feature Count** | 19 features | 13 features |
| **Excluded Features** | None | `newbalanceorig`, `newbalancedest`, `orig_balance_delta`, `dest_balance_delta`, `orig_drained`, `dest_no_change` |
| **Held-Out Test AP** | **0.8803** | **0.8366** |
| **Held-Out Test AUROC** | **0.9956** | **0.9896** |
| **Validation-Fitted Threshold** | 0.995684 | 0.99999998 |
| **Test Precision at Threshold** | 87.81% (220 False Positives) | 83.33% (317 False Positives) |
| **Test Recall at Threshold** | 99.06% (15 False Negatives) | 99.06% (15 False Negatives) |
| **Total Test Frauds Detected** | 1,585 of 1,600 | 1,585 of 1,600 |

- **Interview Defense:**
  *"Removing post-transaction ledger balances reduces Average Precision from 0.880 to 0.837. This 4.3 percentage point delta quantifies the exact value of post-settlement ledger feedback. The 0.8366 AP model is our true, deployment-honest benchmark for real-time payment authorization."*

---

### Trap 3: The Base Rate Coincidence & Metric Integrity
- **The Trap:**
  - Legacy random-CV benchmarks for PaySim quote an Average Precision (PR-AUC) of approximately **0.880**.
  - Our rigorous temporal split retrain on the full 2.77M rows yielded a held-out test AP of **0.8803**.
  - *Interviewer:* *"So your temporal split didn't change the performance at all? It's still 0.88."*
- **The Senior-Level Defense:**
  - **It is a pure mathematical coincidence, not corroboration.**
  - Average Precision is mathematically coupled to the prevalence (base rate) of the positive class. Unlike AUROC, which is prevalence-invariant, PR-AUC scales directly with $P(Y=1)$.
  - In PaySim, fraudulent transaction rates evolve dramatically over the simulated timeline:
    - **Training period (Steps 1–500):** 0.210% fraud base rate (1 fraud per 476 transactions).
    - **Validation period (Steps 501–600):** 1.297% fraud base rate.
    - **Test period (Steps 601–743):** **3.674% fraud base rate** (1 fraud per 27 transactions).
  - The high base rate in the final 143 hours occurs because legitimate synthetic user traffic drops off in the simulation while automated fraud scripts continue executing.
  - Because the test period base rate is 17 times higher than the training base rate, an AP of 0.8803 on steps >600 represents an entirely different precision-recall curve than an AP of 0.880 on a random 0.3% split.
  - **Key takeaway:** Quoting Average Precision or PR-AUC without explicitly stating the evaluation set's base rate is meaningless in fraud analytics.

---

### Trap 4: Decision-Theoretic Cost-Sensitive Threshold Optimization
- **The Pitfall:**
  - 95% of ML papers and tutorials set an arbitrary decision threshold of $p = 0.5$, or perform grid search to maximize the F1-score directly on the test set.
  - Maximizing test-set F1 constitutes data leakage (tuning hyperparameters on holdout data) and implicitly assumes false positives and false negatives carry identical business costs.
  - In financial fraud detection, a False Negative (allowing a $50,000 fraudulent wire transfer) is catastrophic, whereas a False Positive (prompting a 2FA verification or routing to manual analyst triage) incurs a modest operational cost.
- **Our Solution:**
  - We formulated an explicit operational cost function:
    $$\text{Cost} = 1.0 \times \text{FN} + 0.1 \times \text{FP}$$
    *(Missing a fraudulent transaction is weighted as 10 times more damaging than generating a false alert).*
  - **Strict Validation Optimization:** The threshold was tuned strictly on the validation set (steps 501–600) across all empirical candidate cutoffs, including the boundary condition of never alerting.
  - The optimal threshold was frozen at:
    - **Post-Transaction:** $\tau^* = 0.995684$
    - **Pre-Authorization:** $\tau^* = 0.99999998$
  - Only after freezing was the threshold applied once to the unseen test block (steps >600), delivering 99.06% recall with 87.8% precision.

---

### Trap 5: Score Calibration vs. Ranking Discrimination
- **The Observation:**
  - Notice the extreme values of the optimal cost thresholds: $0.9957$ and $0.99999998$.
  - Why are the thresholds pushed so close to $1.0$?
- **The Explanation:**
  - To handle the 0.21% training class imbalance, LightGBM is trained with class weighting (`scale_pos_weight` derived strictly from the training partition: $w_{\text{pos}} \approx 475.5$).
  - Class weighting corrects tree splitting gradients for rare positives, but it systematically distorts the raw output margin, shifting predicted sigmoid probabilities toward 1.0.
  - Furthermore, gradient boosted trees are non-parametric margin maximizers, not probabilistic generative models.
  - **Operational Reality:** The raw model outputs are exceptional **ranking scores** (AUROC 0.9956, AP 0.8803), but they are **not calibrated probabilities**.
  - In a production deployment, if business stakeholders require true expected loss calculations ($\mathbb{E}[\text{loss}] = p \times \text{amount}$), an isotonic regression or Platt scaling calibration layer must be fit on validation predictions before production scoring.

---

## 3. Feature Engineering Architecture

### Scope Filtering: Why CASH_OUT and TRANSFER Only?
- The full PaySim dataset contains 6,362,620 transactions across 5 transaction types: `PAYMENT`, `CASH_IN`, `CASH_OUT`, `TRANSFER`, and `DEBIT`.
- **Domain Observation:** Every single one of the 8,213 fraudulent transactions occurs exclusively in `TRANSFER` (4,097 frauds) or `CASH_OUT` (4,116 frauds).
- `PAYMENT`, `CASH_IN`, and `DEBIT` contain exactly zero fraudulent events in the ground truth.
- Keeping the other 3.59 million transactions merely dilutes the negative class with uninformative noise. We filter the population down to the 2,770,409 eligible candidate transactions before model training, drastically improving computational efficiency and training focus.

### The 19 Engineered Features
1. **Transaction Velocity & Graph Topology (Strict Prior-Step):**
   - `orig_tx_count`: Cumulative count of transactions originating from `nameOrig` up to step $t-1$.
   - `orig_unique_dest`: Distinct destination accounts targeted by `nameOrig` up to step $t-1$.
   - `orig_step_gap`: Hours elapsed since the previous transaction from `nameOrig`.
   - `dest_recv_count`: Inbound transaction velocity received by `nameDest` up to step $t-1$.
2. **Transaction Economics & Amounts:**
   - `amount`: Raw transfer value.
   - `log_amount`: Log-transformed value ($\ln(1 + \text{amount})$) to handle extreme right-skewed transaction distributions.
   - `amount_to_orig_balance`: Ratio of transaction amount to origin available balance ($\text{amount} / (\text{oldbalanceOrg} + 1)$). Values near or exceeding 1.0 indicate account draining.
3. **Temporal Dynamics:**
   - `hour`: Time of day ($0 \dots 23$), computed via `step % 24`.
   - `day`: Day of simulation ($0 \dots 30$), computed via `step // 24`.
   - `is_night`: Binary flag indicating transactions initiated between 00:00 and 06:00.
4. **Transaction Mechanics:**
   - `is_transfer`: Binary indicator ($1$ for `TRANSFER`, $0$ for `CASH_OUT`).
   - `oldbalanceorg`: Pre-existing origin balance.
   - `oldbalancedest`: Pre-existing destination balance.
5. **Post-Transaction Ledger Features (Post-Transaction Model Only):**
   - `newbalanceorig`: Origin balance after transaction settlement.
   - `newbalancedest`: Destination balance after transaction settlement.
   - `orig_balance_delta`: Absolute change in origin balance (`newbalanceorig - oldbalanceorg`).
   - `dest_balance_delta`: Absolute change in destination balance (`newbalancedest - oldbalancedest`).
   - `orig_drained`: Indicator that origin balance was completely liquidated to zero.
   - `dest_no_change`: Indicator that destination balance remained unchanged despite receiving funds (a signature artifact in PaySim synthetic fraud).

---

## 4. Deep-Dive Technical Interview Questions & Model Answers

### Q1: "Can you walk me through your PaySim fraud detection project from end to end?"
> **Model Answer:**
> *"Certainly. I set out to build a production-realistic fraud detection pipeline on PaySim's 6.36 million transaction dataset. After discovering that all 8,213 frauds are concentrated exclusively in `TRANSFER` and `CASH_OUT` transactions, I narrowed the modeling population to those 2.77 million rows.
>
> Rather than following the common mistake of random cross-validation—which creates severe temporal data leakage—I established a chronological train, validation, and test split based on simulation hours. I engineered 19 features, paying strict attention to ensure rolling velocity and graph counters only observed data from strictly earlier time steps.
>
> I then trained a deterministic LightGBM classifier with training-only class balancing. Recognizing that fraud operations cannot afford to evaluate on arbitrary 0.5 thresholds, I optimized a 10:1 false-negative to false-positive cost function exclusively on validation data.
>
> Finally, to answer the question of production viability, I built two versions: a post-transaction model scoring 0.8803 Average Precision and an in-line pre-authorization model scoring 0.8366 Average Precision without post-event balance signals. On the held-out test block, the system captures 99.1% of fraudulent transactions while maintaining 87.8% alert precision."*

---

### Q2: "Why did you choose LightGBM instead of XGBoost, CatBoost, or a Deep Neural Network?"
> **Model Answer:**
> *"There were three primary engineering reasons:
> 1. **Histogram-Based Binning Efficiency:** On 2.77 million rows with high-cardinality features, LightGBM's histogram-based binning reduces continuous features into 255 discrete bins. This achieves training convergence in under 80 seconds on standard dual-core CPU architecture while maintaining sub-2GB memory footprint.
> 2. **Tabular Inductive Bias:** Tree-based ensembles naturally handle non-linear decision boundaries and sharp step functions typical of financial business rules (e.g., `amount == oldbalanceOrg` or `newbalanceOrig == 0`) far better than unregularized neural networks, without requiring extensive normalization or power transforms.
> 3. **Leaf-Wise Tree Growth:** LightGBM builds trees leaf-wise rather than depth-wise, optimizing split loss directly. By constraining `num_leaves=31` and setting `min_child_samples=50`, we achieved high expressive capacity for rare fraud interactions while preventing overfitting on small minority clusters."*

---

### Q3: "Explain exactly how you prevented data leakage in your rolling velocity features."
> **Model Answer:**
> *"In fraud modeling, feature leakage happens in two subtle ways:
> First, **lookahead leakage**: if you compute an account's transaction count using the entire dataset, a transaction at hour 10 knows how many transactions the account will perform at hour 50.
> Second, **same-batch leakage**: if an account executes three rapid-fire transactions within the same hour (step), allowing transaction #1 to see the count after transaction #3 leaks information from simultaneous events.
>
> To eliminate this, my feature engineering function partitions data chronologically. Velocity features for step $t$ are computed using a prefix aggregation strictly over steps $1 \dots t-1$. Transactions sharing the exact same step index are quarantined with same-step isolation—their rolling counts are identical and reflect only past history.
>
> I formalized this guarantee with unit test assertions: we construct synthetic data fixtures, append future fraudulent transactions, and verify that the feature vector generated for earlier transactions is bitwise identical before and after future data insertion."*

---

### Q4: "Your test AUROC is 0.9956, but your test Average Precision is 0.8803. Why the difference, and which metric do you prioritize?"
> **Model Answer:**
> *"I prioritize **Average Precision (PR-AUC)** and operational precision at our target recall, because AUROC is dangerously misleading in severe class imbalance.
>
> AUROC plots True Positive Rate against False Positive Rate. The False Positive Rate denominator is the total number of negative samples ($TN + FP$). In our test set, there are 41,950 legitimate transactions and only 1,600 frauds. If a model generates 400 false alarms, the False Positive Rate is less than 1% ($400 / 41950 \approx 0.0095$), yielding an apparently stellar AUROC of over 0.99.
>
> However, to the fraud operations team investigating those alerts, 400 false alarms alongside 1,500 true positives means that more than 20% of their operational alert volume is wasted effort.
>
> Average Precision integrates precision over recall: precision's denominator is ($TP + FP$), directly capturing analyst alert fatigue. That's why we report AP 0.8803 and precision/recall at our cost threshold as our primary performance criteria."*

---

### Q5: "How does your Pre-Authorization model differ from your Post-Transaction model, and why is this distinction critical in banking?"
> **Model Answer:**
> *"This distinction represents the difference between **in-line prevention** and **post-clearing detection**.
>
> An in-line authorization system sits in the critical payment path. When a customer initiates a wire or transfer, the bank's fraud engine must score the request in under 50 milliseconds to approve, challenge, or decline before the transaction executes. At that moment, the transaction hasn't posted, so `newbalanceOrig` and `newbalanceDest` do not exist.
>
> A post-transaction system runs asynchronously after clearing. It has access to settled ledger balances, reversals, and downstream ledger states. It is useful for anti-money laundering (AML), SAR filing, and fraud ring investigations.
>
> If you evaluate an in-line authorization engine using post-transaction balances, you are committing post-settlement leakage. Our post-transaction model achieves AP 0.8803, but our pre-authorization model—using only the 13 pre-event features—achieves AP 0.8366. Presenting both proves we understand where ML models sit within actual banking infrastructure."*

---

### Q6: "How did you select your decision threshold, and why did you avoid optimizing directly on the test set?"
> **Model Answer:**
> *"Tuning a threshold on test data is a form of data snooping that produces over-optimistic evaluation metrics that degrade in production.
>
> Instead, I implemented an explicit business cost function: $C = 1.0 \times \text{FN} + 0.1 \times \text{FP}$. We value preventing fraud ten times more than the administrative cost of a false alert.
>
> I swept the entire validation prediction score space (steps 501–600), evaluating empirical score cutoffs and including the boundary case of 'never alerting'. The cost curve found an optimal threshold at $\tau = 0.995684$ for post-transaction and $\tau = 0.99999998$ for pre-auth.
>
> That threshold was frozen and stored in metadata. Only then was it applied to the unseen test partition (steps 601–743). On the test set, it yielded 1,585 true positives, 220 false positives, and only 15 missed frauds—delivering 99.06% fraud recall at an 87.8% precision rate."*

---

### Q7: "Your validation-selected threshold is 0.9957. Does that mean a transaction needs a 99.57% probability of being fraud to get flagged?"
> **Model Answer:**
> *"No, and that is a crucial point regarding model calibration.
>
> Because we applied class weighting (`scale_pos_weight`) during training to compensate for the 0.21% fraud base rate, LightGBM's raw log-odds outputs are shifted significantly upward. As a result, the output scores are non-linear ranking scores rather than well-calibrated posterior probabilities.
>
> A threshold of 0.9957 simply represents the 95th–99th percentile cutoff on the model's empirical score distribution where the marginal cost of an additional false alarm exceeds the marginal benefit of an additional caught fraud.
>
> If downstream systems require true Bayesian risk estimates—for example, calculating Expected Value Loss = $P(\text{Fraud}) \times \text{Amount}$—I would calibrate the model by fitting an isotonic regression or Platt sigmoid on the validation set before production scoring."*

---

### Q8: "How would you deploy this model in a real-world production banking architecture?"
> **Model Answer:**
> *"A production deployment requires decoupling feature serving from model inference:
>
> 1. **Streaming Feature Pipeline:** We would ingest core payment events via Apache Kafka. A streaming processing engine like Apache Flink would maintain stateful sliding windows for rolling velocity metrics (`orig_tx_count`, `dest_recv_count`, `orig_step_gap`) and store low-latency aggregations in an in-memory feature store like Redis or Dragonfly.
> 2. **Inference Microservice:** We export the trained LightGBM model to ONNX or compile it via Treelite for optimized C++ execution. The scoring service is wrapped in a containerized Go or FastAPI microservice, achieving sub-10ms P99 inference latency.
> 3. **Fallback & Rules Engine:** In-line scoring sits alongside a deterministic rule engine (e.g., hard velocity limits, OFAC sanction screening). If the ML service times out (>40ms), the system falls back gracefully to deterministic rule policies.
> 4. **Drift & Performance Monitoring:** We continuously track Population Stability Index (PSI) on incoming features and monitor prediction score distributions. As fraud analysts review flagged cases, ground truth labels are fed back into an asynchronous pipeline to track rolling precision-recall curves and trigger automated model retraining when performance drops below validation baselines."*

---

## 5. Summary Reference Card for Interviews

| Dimension | Specification | Interview Value |
|---|---|---|
| **Dataset** | PaySim Synthetic Financial Transactions | 6,362,620 total rows; 2,770,409 filtered (`TRANSFER` / `CASH_OUT`) |
| **Total Frauds** | 8,213 frauds (100% inside `TRANSFER` & `CASH_OUT`) | Explains why filtering improves signal-to-noise ratio |
| **Split Strategy** | Temporal step-based holdout | Train: 1–500 (2.65M rows, 0.21% fraud)<br>Val: 501–600 (81K rows, 1.30% fraud)<br>Test: 601–743 (43.5K rows, 3.67% fraud) |
| **Feature Store** | 19 features (Post-Auth) / 13 features (Pre-Auth) | Strictly prior-step extraction; zero lookahead or same-step leakage |
| **Post-Auth Model** | Test AP: **0.8803** \| Test AUROC: **0.9956** | Precision: 87.81% \| Recall: 99.06% at $\tau = 0.9957$ |
| **Pre-Auth Model** | Test AP: **0.8366** \| Test AUROC: **0.9896** | Precision: 83.33% \| Recall: 99.06% at $\tau = 0.99999998$ |
| **Cost Basis** | Cost = $1.0 \times \text{FN} + 0.1 \times \text{FP}$ | Tuned on validation only; frozen before test evaluation |
| **Training Speed** | 75.0s – 77.1s runtime | Peak RSS ~1.9 GiB; dual-core CPU deterministic reproducibility |
| **Reproducibility** | Full hash-bound artifacts & 58-test suite | `bash reproduce.sh` passes 100% clean |

---

AUTHORED BY GEMINI (extending HELIOS & ORION frameworks)
